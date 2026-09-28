"""改進器（Improver）與策略（Strategy）：把經驗變成候選改進。

改進器只讀得到「經驗」（黑匣子裡的執行紀錄、記憶圖）與評測回傳的彙總分數，
不匯入 protected.py、看不到評測集——候選是從經驗診斷出來的，不是從評測集反推的。

策略（Strategy）決定「往哪裡找」：
  - 目標優先序 = 診斷出的牽連程度 × 該目標過去的接受率（Beta 後驗平均 (接受+1)/(嘗試+2)）
  - 每個參數有自己的搜尋方向與步長：被接受 → 方向不變、步長放大 1.5 倍；被拒絕 → 反向、步長減半
  - 每個候選只改一個參數（小步編輯預算，避免一次綁一堆改動、分不清是哪個起作用）
策略本身隨每一輪的結果更新並被繼承——這是「改進改進的方法」，但更新規則是寫死的（見 rsi.py 的自主等級說明）。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lingxi.evolve.state import TUNABLE, SystemState, clamp, read_setting
from lingxi.journal.recorder import Journal
from lingxi.llm.kvcache import replay
from lingxi.memory.fluxmem import common_phrases

TARGET_KEYS = {
    "context": ["kv_cache.fold_block", "agent.fold_after_turns"],
    "retrieval": ["memory.min_score", "retrieval.bm25_weight"],
}
DEFAULT_STEPS = {"kv_cache.fold_block": 2, "agent.fold_after_turns": 1, "agent.max_observation_chars": 1000,
                 "retrieval.dense_weight": 0.25, "retrieval.bm25_weight": 0.25, "memory.min_score": 0.05}
DEFAULT_DIRS = {"kv_cache.fold_block": 1, "agent.fold_after_turns": -1, "agent.max_observation_chars": -1,
                "retrieval.dense_weight": 1, "retrieval.bm25_weight": 1, "memory.min_score": -1}
MIN_STEPS = {"kv_cache.fold_block": 1, "agent.fold_after_turns": 1, "agent.max_observation_chars": 250,
             "retrieval.dense_weight": 0.05, "retrieval.bm25_weight": 0.05, "memory.min_score": 0.02}


@dataclass
class Candidate:
    id: str
    target: str  # context | retrieval | playbook
    harness: dict[str, Any]
    change: dict[str, Any] = field(default_factory=dict)
    playbook: dict[str, Any] | None = None  # {"node", "file", "toml", "tasks"}
    rationale: str = ""


@dataclass
class Experience:
    runs: list[dict[str, Any]] = field(default_factory=list)

    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.runs if r["status"] != "success"]


def observe(runs_dir: Path, since: float = 0.0) -> Experience:
    """經驗選擇：讀取上一輪之後的執行紀錄（自上次改進以來的部署經驗）。"""
    exp = Experience()
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "events.jsonl").exists()):
        events = Journal.load(run_dir)
        if not events or events[0].ts <= since:
            continue
        finish = next((e for e in reversed(events) if e.kind == "run.finish"), None)
        memory = Counter(e.data.get("action") for e in events if e.kind == "memory")
        recall = next((e for e in events if e.kind == "memory" and e.data.get("action") == "recall"), None)
        recalled = sum(len(recall.data.get(k, [])) for k in ("semantic", "episodic", "procedural")) if recall else 0
        outputs = [len(e.data.get("summary", "")) + len(e.data.get("detail", "")) + 40
                   for e in events if e.kind == "outcome"]
        exp.runs.append({
            "run": run_dir.name, "task": events[0].data.get("task", ""), "ts": events[0].ts,
            "status": finish.data.get("status") if finish else "unfinished",
            "steps": finish.step if finish else 0,
            "prefix_reuse": ((finish.data.get("kv") or {}).get("prefix_reuse") if finish else None),
            "memory": dict(memory), "recalled": recalled,
            "memory_size": (recall.data.get("stats") or {}) if recall else {},
            "outputs": outputs,
            "guards": sum(1 for e in events if e.kind == "guard"),
        })
    return exp


def diagnose(exp: Experience, graph, exported: set[str], memory_cfg) -> dict[str, dict[str, Any]]:
    """失敗歸因到可改進的目標（論文例子：優先看被失敗測試牽連的程式位置）。"""
    diag: dict[str, dict[str, Any]] = {}
    reuse = [r["prefix_reuse"] for r in exp.runs if r["prefix_reuse"] is not None]
    avg = sum(reuse) / len(reuse) if reuse else None
    diag["context"] = {"implication": 0.2 + (max(0.0, 0.8 - avg) if avg is not None else 0.0),
                       "reasons": [f"近 {len(reuse)} 次執行平均 KV 前綴重用率 {avg:.0%}" if avg is not None
                                   else "沒有新的執行紀錄，以固定基線評估"]}
    expand = sum(r["memory"].get("expand", 0) for r in exp.runs)
    prune = sum(r["memory"].get("prune", 0) for r in exp.runs)
    empty = sum(1 for r in exp.runs if r["recalled"] == 0 and sum(
        v for k, v in r["memory_size"].items() if k in ("semantic", "episodic", "procedural")) > 0)
    n = max(len(exp.runs), 1)
    diag["retrieval"] = {"implication": 0.1 + (expand + prune + empty) / n,
                         "expand": expand, "prune": prune, "empty_recall": empty,
                         "reasons": [f"連結不足（擴充）{expand} 次、過度連結（剪枝）{prune} 次、"
                                     f"記憶非空卻零召回 {empty} 次"]}
    mature = []
    if graph is not None:
        for node in graph.active("procedural"):
            d = node.data
            if d.get("converged") and len(d.get("support", [])) >= memory_cfg.min_support and node.id not in exported:
                mature.append(node)
    diag["playbook"] = {"implication": float(len(mature)), "nodes": mature,
                        "reasons": [f"{len(mature)} 個 PEMS 已收斂、尚未匯出的程序技能"]}
    return diag


def curriculum(exp: Experience, limit: int = 5) -> list[str]:
    """經驗獲取建議（L3 的部分能力）：下一輪最值得重練的任務＝最近失敗或未完成的。需要人或排程實際去跑。"""
    return [r["task"] for r in sorted(exp.failed(), key=lambda r: -r["ts"])][:limit]


def dev_context_score(exp: Experience, harness: dict[str, Any], budget_chars: int = 60000) -> float | None:
    """開發評測：用真實執行軌跡（每步工具輸出字數）重放。只用來偵測目標漂移，不參與接受決策。"""
    traces = [r["outputs"] for r in exp.runs if len(r["outputs"]) >= 3]
    if not traces:
        return None
    scores = [1.0 - replay(t, int(harness["agent.fold_after_turns"]), int(harness["kv_cache.fold_block"]),
                           int(harness["agent.max_observation_chars"]))["billed_chars"] / budget_chars for t in traces]
    return sum(scores) / len(scores)


class Strategy:
    def __init__(self, raw: dict[str, Any] | None = None):
        raw = raw or {}
        self.targets: dict[str, dict[str, int]] = raw.get("targets", {})
        self.steps: dict[str, float] = {**DEFAULT_STEPS, **raw.get("steps", {})}
        self.dirs: dict[str, int] = {**DEFAULT_DIRS, **raw.get("dirs", {})}

    def priority(self, target: str, implication: float) -> float:
        t = self.targets.get(target, {"tried": 0, "accepted": 0})
        return implication * (t["accepted"] + 1) / (t["tried"] + 2)

    def order(self, diag: dict[str, dict[str, Any]]) -> list[str]:
        ranked = sorted(diag, key=lambda t: -self.priority(t, diag[t]["implication"]))
        return [t for t in ranked if diag[t]["implication"] > 0]

    def record(self, target: str, key: str | None, accepted: bool) -> None:
        t = self.targets.setdefault(target, {"tried": 0, "accepted": 0})
        t["tried"] += 1
        t["accepted"] += int(accepted)
        if key is None:
            return
        if accepted:
            self.steps[key] = self.steps[key] * 1.5
        else:
            self.dirs[key] = -self.dirs[key]
            self.steps[key] = max(MIN_STEPS[key], self.steps[key] * 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {"targets": self.targets, "steps": {k: round(v, 3) for k, v in self.steps.items()}, "dirs": self.dirs}


def effective_harness(settings, state: SystemState) -> dict[str, Any]:
    return {k: state.harness.get(k, read_setting(settings, k)) for k in TUNABLE}


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # JSON 字串跳脫與 TOML basic string 相容


def export_playbook(node, version: int) -> tuple[str, list[str]] | None:
    """程序技能 → 站點手冊 TOML（只有步驟與陷阱，不推測 URL 模板）。沒有可用觸發詞時回傳 None。"""
    d = node.data
    tasks = d.get("tasks", [])
    phrases = common_phrases(tasks, min_len=2, limit=3)
    # 優先用具體的長片段當觸發詞：「北京」「航班」這種短詞很容易搶走別本手冊的任務（受保護路由評測會擋下來）
    triggers = [p for p in phrases if len(p) >= 3] or phrases
    if not triggers:
        return None
    lines = [f"# 由 RSI 自動產生（靈犀 v{version}）：來自程序技能 {node.id}《{node.title}》",
             f"# 依據：{len(d.get('support', []))} 次成功經歷，PEMS {d.get('pems_final', 0):.3f}"
             f"（{d.get('rounds', 0)} 輪，{'已' if d.get('converged') else '未'}收斂）；通過受保護路由評測無回歸才被接受",
             f"id = {_toml_str('learned-' + node.id.lower())}",
             f"title = {_toml_str('（學到的）' + node.title)}"]
    if d.get("profile"):
        lines.append(f"profile = {_toml_str(d['profile'])}")
    lines.append("triggers = [" + ", ".join(_toml_str(t) for t in triggers) + "]")
    lines.append("tips = [" + ", ".join(_toml_str(p) for p in d.get("pitfalls", [])) + "]")
    lines += ["", "[route]", "steps = [" + ", ".join(_toml_str(s) for s in d.get("steps", [])) + "]", ""]
    return "\n".join(lines), tasks


class Improver:
    def __init__(self, settings):
        self.settings = settings
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"c{int(time.time()) % 100000}-{self._n}"

    def propose(self, target: str, parent: SystemState, base: dict[str, Any], strategy: Strategy,
                diag: dict[str, dict[str, Any]], version: int) -> list[Candidate]:
        if target == "playbook":
            out = []
            for node in diag["playbook"].get("nodes", []):
                exported = export_playbook(node, version)
                if exported is None:
                    continue
                toml, tasks = exported
                out.append(Candidate(self._id(), "playbook", dict(base),
                                     playbook={"node": node.id, "file": f"{node.id}.toml", "toml": toml, "tasks": tasks},
                                     rationale=f"程序技能 {node.id}《{node.title}》PEMS 已收斂，"
                                               f"有 {len(node.data.get('support', []))} 次成功經歷支撐"))
            return out
        out = []
        for key in TARGET_KEYS[target]:
            direction = strategy.dirs[key]
            if target == "retrieval" and key == "memory.min_score":
                d = diag["retrieval"]
                if d["expand"] + d["empty_recall"] > d["prune"]:
                    direction = -1  # 召回不足：放寬門檻
                elif d["prune"] > d["expand"] + d["empty_recall"]:
                    direction = 1  # 召回太雜：收緊門檻
            value = clamp(key, base[key] + direction * strategy.steps[key])
            if value == base[key]:
                value = clamp(key, base[key] - direction * strategy.steps[key])  # 碰到邊界就試另一個方向
            if value == base[key]:
                continue
            harness = {**base, key: value}
            out.append(Candidate(self._id(), target, harness, change={key: [base[key], value]},
                                 rationale=f"{'；'.join(diag[target]['reasons'])} → 試 {key} {base[key]} → {value}"))
        return out
