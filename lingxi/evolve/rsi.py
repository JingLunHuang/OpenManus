"""遞迴自我改進（RSI）閘門。依據《The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement》
（arXiv:2609.11873，上海交大、清華大學等）的改進迴圈框架實作。

論文定義的改進迴圈：「AI 系統用經驗對某個目標提出修改，依接受規則評估候選，把被接受的改變保留在狀態裡，
並從更新後的狀態開始下一輪。」九個要素在靈犀裡的對應：

  AI System    靈犀本身（每個版本 vN 是一個可追蹤的系統）
  System State evolve/versions/vN.json：harness 參數 ＋ 學到的手冊 ＋ 策略參數（state.py）
  Experience   上一輪之後的真實執行（黑匣子）＋ LightMem/FluxMem 記憶圖（improver.observe）
  Target       本輪修改的對象：上下文佈局（KV 快取）/ 檢索參數 / 手冊庫
  Improver     依診斷產生候選：參數小步調整；PEMS 收斂的程序技能匯出成手冊（improver.Improver）
  Strategy     決定往哪裡找：目標優先序、每個參數的方向與步長，隨結果更新並被繼承（improver.Strategy）
  Verifier     受保護評測：路由 / 上下文重放 / 記憶檢索（protected.py）
  Improvement  通過接受規則的候選：受保護分數提升 ≥ δ（手冊：路由無回歸 ＋ 覆蓋全部來源任務）
  Successor    整合所有被接受的改變 → 新版本 vN+1，之後的每次執行都從它開始

論文點出的三個挑戰與對策：
  安全繼承   版本歷史 ＋ 一鍵回滾；多個改變整合後重新評測全部套件，任一套件退步就只保留增益最大的單一改變
  自主歸屬   每一輪在 ledger 裡寫明哪些決策由 AI 做、哪些是外部固定的，並據此標出自主等級
  可靠驗證   評測集與改進器隔離、epoch 凍結、查詢預算；另用真實軌跡做開發評測，兩者不一致時標記「目標漂移」

自主等級（論文 L1–L5）：
  L4 部署自適應 —— 用真實執行的經驗修改持久狀態，接受規則由外部規定（本模組的常態）
  L5 只到「部分」—— 策略參數（方向、步長、目標先驗）會被改寫，但改寫規則、評測器、接受規則都不能被改寫；
     這是刻意的：論文指出評測器一旦進入可修改範圍，就會出現自我評估與目標漂移。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lingxi.evolve.improver import (Candidate, Improver, Strategy, curriculum, dev_context_score, diagnose,
                                    effective_harness, observe)
from lingxi.evolve.protected import SUITES, BudgetExceeded, ProtectedEvaluator
from lingxi.evolve.state import StateStore
from lingxi.knowledge.playbook import Playbook, PlaybookLibrary

TARGET_SUITE = {"context": "context", "retrieval": "retrieval", "playbook": "routing"}


@dataclass
class RoundReport:
    round: int
    epoch: str
    base_version: int
    new_version: int | None = None
    baseline: dict[str, float] = field(default_factory=dict)
    final: dict[str, float] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    hci: dict[str, float | None] = field(default_factory=dict)
    drift: list[str] = field(default_factory=list)
    curriculum: list[str] = field(default_factory=list)
    autonomy: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, int] = field(default_factory=dict)
    experience_runs: int = 0
    note: str = ""
    ts: float = field(default_factory=time.time)


def hci(score: float, v0: float) -> float | None:
    """Headroom-Closed Index（論文的跨基準正規化）：相對 v0 補上了多少「到滿分的剩餘空間」，0 = v0、100 = 滿分。"""
    return None if v0 >= 1.0 else round(100 * (score - v0) / (1.0 - v0), 1)


class RSILoop:
    def __init__(self, settings, graph=None):
        self.settings = settings
        self.store = StateStore(settings.evolve_dir)
        self.ledger = settings.evolve_dir / "ledger.jsonl"
        self.strategy_path = settings.evolve_dir / "strategy.json"
        self.evaluator = ProtectedEvaluator([settings.path(d) for d in settings.playbook_dirs],
                                            settings.evolve.query_budget, settings.retrieval.backend,
                                            settings.retrieval.embedding_dim)
        self.graph = graph

    # ---------- 紀錄 ----------
    def rounds(self) -> list[dict[str, Any]]:
        if not self.ledger.is_file():
            return []
        return [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _strategy(self, parent) -> Strategy:
        if self.strategy_path.is_file():
            return Strategy(json.loads(self.strategy_path.read_text(encoding="utf-8")))
        return Strategy(parent.strategy)

    def _scores(self, state, harness) -> dict[str, float]:
        """同一個 epoch 裡，同一個版本的分數只評一次（省查詢預算）。"""
        if state.scores.get("epoch") == self.evaluator.epoch:
            return {s: state.scores[s] for s in SUITES}
        scores = self.evaluator.score_all(harness, self._files(state.version))
        state.scores = {**scores, "epoch": self.evaluator.epoch}
        self.store.save(state)
        return scores

    def _files(self, version: int) -> list[Path]:
        return sorted(self.store.playbook_dir(version).glob("*.toml"))

    def _v0(self) -> dict[str, float]:
        v0 = self.store.get(0) or self.store.current()
        return self._scores(v0, effective_harness(self.settings, v0))

    # ---------- 一輪 ----------
    def round(self) -> RoundReport:
        parent = self.store.current()
        history = self.rounds()
        since = history[-1]["ts"] if history else 0.0
        report = RoundReport(round=len(history) + 1, epoch=self.evaluator.epoch, base_version=parent.version)
        base = effective_harness(self.settings, parent)
        strategy = self._strategy(parent)

        # Observe → Diagnose
        exp = observe(self.settings.runs_dir, since)
        report.experience_runs = len(exp.runs)
        diag = diagnose(exp, self.graph, set(parent.playbooks), self.settings.memory)
        report.diagnosis = {t: {k: v for k, v in d.items() if k != "nodes"} for t, d in diag.items()}
        report.curriculum = curriculum(exp)
        improver = Improver(self.settings)

        try:
            v0 = self._v0()
            report.baseline = baseline = self._scores(parent, base)
            accepted: list[Candidate] = []
            gains: dict[str, float] = {}
            # Propose → Validate → Select（每個目標最多接受一個改變：小步編輯）
            for target in strategy.order(diag):
                suite = TARGET_SUITE[target]
                for cand in improver.propose(target, parent, base, strategy, diag, parent.version + 1):
                    row = {"id": cand.id, "target": target, "suite": suite, "change": cand.change,
                           "playbook": (cand.playbook or {}).get("node"), "rationale": cand.rationale}
                    key = next(iter(cand.change), None)
                    if cand.playbook is not None:
                        coverage = self._dev_coverage(cand)
                        row["dev_coverage"] = coverage
                        if coverage < 1.0:  # 開發檢查不過就不浪費受保護評測的預算
                            row.update(accepted=False, reason=f"只覆蓋 {coverage:.0%} 的來源任務")
                            report.candidates.append(row)
                            strategy.record(target, None, False)
                            continue
                    files = self._files(parent.version) + ([self._tmp_playbook(cand)] if cand.playbook else [])
                    score = self.evaluator.score(suite, cand.harness, files)
                    gain = round(score - baseline[suite], 4)
                    ok = gain >= (0.0 if target == "playbook" else self.settings.evolve.min_gain)
                    row.update(score=round(score, 4), gain=gain, accepted=ok,
                               reason="通過接受規則" if ok else ("路由退步" if target == "playbook"
                                                              else f"提升 {gain:+.4f} 未達 δ={self.settings.evolve.min_gain}"))
                    if target == "context" and ok:
                        dev_base, dev_new = dev_context_score(exp, base), dev_context_score(exp, cand.harness)
                        if dev_base is not None and dev_new is not None:
                            row["dev_gain"] = round(dev_new - dev_base, 4)
                            if (dev_new - dev_base) * gain < 0:
                                report.drift.append(f"{cand.id}：受保護評測 {gain:+.4f}、真實軌跡 {dev_new - dev_base:+.4f}，方向不一致")
                    report.candidates.append(row)
                    strategy.record(target, key, ok)
                    if ok:
                        accepted.append(cand)
                        gains[cand.id] = gain
                        break

            # Integrate（安全繼承）：整合後重評全部套件，任何套件退步就只留增益最大的一個
            if accepted:
                chosen = accepted
                final = self._integrated(parent, base, chosen)
                if any(final[s] < baseline[s] - 1e-9 for s in SUITES):
                    best = max(accepted, key=lambda c: gains[c.id])
                    chosen = [best]
                    final = self._integrated(parent, base, chosen)
                    report.note = "整合後有套件退步，只保留增益最大的單一改變"
                if all(final[s] >= baseline[s] - 1e-9 for s in SUITES):
                    harness = dict(base)
                    for c in chosen:
                        harness.update({k: v[1] for k, v in c.change.items()})
                    new_playbooks = {c.playbook["file"]: c.playbook["toml"] for c in chosen if c.playbook}
                    successor = self.store.commit(parent, harness, new_playbooks, strategy.as_dict(),
                                                  {**final, "epoch": self.evaluator.epoch},
                                                  note="；".join(c.rationale for c in chosen))
                    report.new_version = successor.version
                    report.final = final
                    for row in report.candidates:
                        row["inherited"] = any(row["id"] == c.id for c in chosen)
            report.final = report.final or baseline
            report.hci = {s: hci(report.final[s], v0[s]) for s in SUITES}
        except BudgetExceeded as exc:
            report.note = str(exc)
        finally:
            self._tmp_cleanup()

        report.budget = {"used": self.evaluator.queries, "limit": self.evaluator.query_budget}
        report.autonomy = self._autonomy(report)
        self.strategy_path.write_text(json.dumps(strategy.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")
        return report

    def rollback(self, version: int) -> None:
        self.store.set_current(version)
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"round": len(self.rounds()) + 1, "rollback": version, "ts": time.time(),
                                "note": f"人工回滾到 v{version}"}, ensure_ascii=False) + "\n")

    # ---------- 內部 ----------
    def _integrated(self, parent, base, chosen: list[Candidate]) -> dict[str, float]:
        harness = dict(base)
        for c in chosen:
            harness.update({k: v[1] for k, v in c.change.items()})
        files = self._files(parent.version) + [self._tmp_playbook(c) for c in chosen if c.playbook]
        return self.evaluator.score_all(harness, files)

    def _tmp_dir(self) -> Path:
        d = self.settings.evolve_dir / "candidates"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tmp_playbook(self, cand: Candidate) -> Path:
        path = self._tmp_dir() / cand.playbook["file"]
        path.write_text(cand.playbook["toml"], encoding="utf-8")
        return path

    def _tmp_cleanup(self) -> None:
        for f in self._tmp_dir().glob("*.toml"):
            f.unlink()

    def _dev_coverage(self, cand: Candidate) -> float:
        """開發檢查：新手冊必須能命中它的所有來源任務（訓練集），否則根本不會被用到。"""
        book = Playbook.from_toml(self._tmp_playbook(cand))
        library = PlaybookLibrary([book])
        tasks = cand.playbook["tasks"]
        return sum(1 for t in tasks if library.match(t, [], top_k=1)) / len(tasks) if tasks else 0.0

    def _autonomy(self, report: RoundReport) -> dict[str, Any]:
        decisions = {
            "objective": "外部：使用者的任務與人工標註的受保護評測",
            "acceptance_rule": f"外部：受保護分數提升 ≥ {self.settings.evolve.min_gain}（手冊：路由無回歸且覆蓋全部來源任務），整合後不得退步",
            "verifier": f"外部：lingxi/evolve/protected（epoch {report.epoch}，AI 不可修改）",
            "tunable_scope": "外部：state.TUNABLE 白名單與範圍",
            "experience_selection": f"AI：選用上一輪之後的 {report.experience_runs} 次真實執行",
            "target_selection": "AI：依失敗歸因 × 歷史接受率排序目標",
            "proposal": "AI：參數方向與步長、程序技能匯出",
            "strategy_revision": "AI（規則固定）：接受則放大步長、拒絕則反向減半",
        }
        level = "L4 部署自適應" if report.experience_runs else "L2 策略自主"
        return {"level": level, "l5": "部分：只改寫策略參數；改寫規則、評測器、接受規則不在可修改範圍",
                "decisions": decisions}
