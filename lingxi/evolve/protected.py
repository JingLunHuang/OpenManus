"""受保護評測（Verifier）：RSI 論文「可靠驗證」挑戰的對策。

  1. 資料與改進器隔離：評測集隨倉庫發佈在 lingxi/evolve/protected/，只有這個模組讀取；
     改進器（improver.py）不匯入它、拿不到逐題結果，只看得到每個套件的彙總分數。
  2. 評測週期（epoch）凍結：評測集內容的雜湊就是 epoch；內容一變，epoch 就變，舊分數全部作廢重算。
  3. 查詢預算：同一個 epoch 裡最多評測 query_budget 次，防止「反覆試到分數好看為止」式的評測器利用。
  4. 預算對等：三個套件都是確定性的重放，基線與候選在完全相同的條件下評分。

三個套件：
  routing   手冊路由（21 題，人工標註）→ 準確率。新手冊不能搶走不該命中的任務。
  context   上下文重放（3 條固定種子軌跡）→ 1 − 平均「有效計費量」／體積預算；
            有效計費量 = 未命中部分 ＋ cache_price × 命中部分（DashScope 隱式快取命中價 20%），
            同時懲罰「前綴不穩」與「提示太長」，不會為了重用率把上下文無限撐大。
  retrieval 記憶檢索（18 條記憶、15 個查詢）→ MRR@5 − 0.05 × 平均每題混入的無關記憶數。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from lingxi.knowledge.playbook import Playbook, PlaybookLibrary
from lingxi.llm.kvcache import replay

_DATA = Path(__file__).parent / "protected"
SUITES = ("routing", "context", "retrieval")


class BudgetExceeded(RuntimeError):
    pass


class ProtectedEvaluator:
    def __init__(self, base_playbook_dirs: list[Path], query_budget: int = 60, backend: str = "auto",
                 embedding_dim: int = 512):
        self.__suites = {name: json.loads((_DATA / f"{name}.json").read_text(encoding="utf-8")) for name in SUITES}
        digest = hashlib.sha1()
        for name in SUITES:
            digest.update((_DATA / f"{name}.json").read_bytes())
        self.epoch = digest.hexdigest()[:12]
        self.base_dirs = base_playbook_dirs
        self.query_budget = query_budget
        self.queries = 0
        self.backend = backend
        self.embedding_dim = embedding_dim

    def _spend(self) -> None:
        if self.queries >= self.query_budget:
            raise BudgetExceeded(f"本評測週期（epoch {self.epoch}）的評測次數已用完（{self.query_budget} 次）")
        self.queries += 1

    def score(self, suite: str, harness: dict[str, Any], playbook_files: list[Path] | None = None) -> float:
        self._spend()
        data = self.__suites[suite]
        if suite == "routing":
            return self._routing(data, playbook_files or [])
        if suite == "context":
            return self._context(data, harness)
        return self._retrieval(data, harness)

    def score_all(self, harness: dict[str, Any], playbook_files: list[Path] | None = None) -> dict[str, float]:
        return {s: round(self.score(s, harness, playbook_files), 4) for s in SUITES}

    # ---------- routing ----------
    def _routing(self, data, files: list[Path]) -> float:
        library = PlaybookLibrary.load(self.base_dirs)
        library.playbooks += [Playbook.from_toml(f) for f in files]
        hits = 0
        for case in data["cases"]:
            top = library.match(case["task"], [], top_k=1)
            got = top[0].id if top else None
            hits += got == case["expect"]
        return hits / len(data["cases"])

    # ---------- context ----------
    def _context(self, data, harness: dict[str, Any]) -> float:
        fold_after = int(harness.get("agent.fold_after_turns", 6))
        if fold_after < data["min_fold_after"]:
            return 0.0  # 資訊保留下限：最近幾輪的觀察必須完整保留
        scores = []
        for trace in data["traces"]:
            rng = random.Random(trace["seed"])
            outputs = [rng.randint(*data["tool_output_chars"]) for _ in range(trace["steps"])]
            r = replay(outputs, fold_after, int(harness.get("kv_cache.fold_block", 4)),
                       int(harness.get("agent.max_observation_chars", 6000)), tools_chars=data["tools_chars"],
                       system_chars=data["system_chars"], brief_chars=data["brief_chars"],
                       cache_price=data["cache_price"])
            scores.append(1.0 - r["billed_chars"] / data["budget_chars"])
        return sum(scores) / len(scores)

    # ---------- retrieval ----------
    def _retrieval(self, data, harness: dict[str, Any]) -> float:
        from lingxi.retrieval import Doc, HashEmbedder, HybridIndex

        index = HybridIndex(HashEmbedder(self.embedding_dim), self.backend,
                            float(harness.get("retrieval.dense_weight", 1.0)),
                            float(harness.get("retrieval.bm25_weight", 0.5)))
        index.add([Doc(d["id"], d["text"], {"layer": d["layer"], "site": d["site"]}) for d in data["corpus"]])
        min_score = float(harness.get("memory.min_score", 0.25))
        total = 0.0
        for q in data["queries"]:
            hits = index.search(q["q"], k=5, min_score=min_score)
            ids = [h.doc.id for h in hits]
            rr = next((1.0 / (i + 1) for i, x in enumerate(ids) if x in q["relevant"]), 0.0)
            noise = sum(1 for x in ids if x not in q["relevant"])
            total += rr - 0.05 * noise
        return total / len(data["queries"])
