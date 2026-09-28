"""FluxMem 異質記憶圖：三層節點 ＋ 型別化的邊，檢索走 LlamaIndex 混合索引。

  語義層 semantic   —— 站點知識與事實（「出發城市輸入後必須點選聯想候選」「MU5101 ¥680」）
  情節層 episodic   —— 一次執行的完整軌跡（任務、每一步的動作簽名與成敗、最終狀態）
  程序層 procedural —— 從多次相似經歷蒸餾出的可重用技能（步驟 ＋ 陷阱 ＋ PEMS 成熟度）

  ground   語義 → 情節：這條知識/事實出自、支撐了哪次經歷
  distill  情節 → 程序：這個技能是從哪些經歷蒸餾出來的
  steplink 任意 → 任意：執行中臨時建立的「本步用到了哪些記憶」（不落盤，見 fluxmem.Subgraph）

節點從不物理刪除：被更新取代的標為 superseded（保留取代鏈），被證明有害的標為 retired，
讓「記憶怎麼演化」本身也可以被復盤。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lingxi.retrieval import Doc, Embedder, HybridIndex, cosine

LAYERS = ("semantic", "episodic", "procedural")
PREFIX = {"semantic": "S", "episodic": "E", "procedural": "P"}


@dataclass
class MemNode:
    id: str
    layer: str
    text: str
    title: str = ""
    site: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    run_id: str = ""
    sources: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    uses: int = 0
    helpful: int = 0
    harmful: int = 0
    status: str = "active"  # active | superseded | retired
    superseded_by: str = ""

    @property
    def active(self) -> bool:
        return self.status == "active"

    def search_text(self) -> str:
        if self.layer == "episodic":
            return f"{self.title}\n{' → '.join(s['a'] for s in self.data.get('steps', []))}"
        if self.layer == "procedural":
            return f"{self.title}\n" + "\n".join(self.data.get("steps", []))
        return self.text

    def age(self, now: float | None = None) -> str:
        seconds = (now or time.time()) - self.created
        if seconds < 3600:
            return f"{max(1, int(seconds // 60))} 分鐘前"
        if seconds < 86400:
            return f"{int(seconds // 3600)} 小時前"
        return f"{int(seconds // 86400)} 天前"


@dataclass
class MemEdge:
    type: str  # ground | distill
    src: str
    dst: str
    weight: float = 1.0
    created: float = field(default_factory=time.time)


class MemoryGraph:
    def __init__(self, path: Path | None, embedder: Embedder, backend: str = "auto",
                 dense_weight: float = 1.0, sparse_weight: float = 0.5):
        self.path = path
        self.embedder = embedder
        self.nodes: dict[str, MemNode] = {}
        self.edges: list[MemEdge] = []
        self.counters = {layer: 0 for layer in LAYERS}
        self.meta: dict[str, Any] = {"runs_since_sleep": 0, "sleeps": 0, "ingested": []}
        self.index = HybridIndex(embedder, backend, dense_weight, sparse_weight)

    # ---------- 持久化 ----------
    @classmethod
    def load(cls, path: Path | None, embedder: Embedder, backend: str = "auto",
             dense_weight: float = 1.0, sparse_weight: float = 0.5) -> "MemoryGraph":
        g = cls(path, embedder, backend, dense_weight, sparse_weight)
        if path and path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            g.nodes = {n["id"]: MemNode(**n) for n in raw.get("nodes", [])}
            g.edges = [MemEdge(**e) for e in raw.get("edges", [])]
            g.counters.update(raw.get("counters", {}))
            g.meta.update(raw.get("meta", {}))
            g.index.add([g._doc(n) for n in g.nodes.values() if n.active])
        return g

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "counters": self.counters, "meta": self.meta,
                   "nodes": [asdict(n) for n in self.nodes.values()], "edges": [asdict(e) for e in self.edges]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---------- 寫入 ----------
    def new_id(self, layer: str) -> str:
        self.counters[layer] += 1
        return f"{PREFIX[layer]}{self.counters[layer]}"

    def add(self, node: MemNode) -> MemNode:
        if not node.id:
            node.id = self.new_id(node.layer)
        self.nodes[node.id] = node
        if node.active:
            self.index.add([self._doc(node)])
        return node

    def touch(self, node: MemNode) -> None:
        """節點內容改變後重新索引。"""
        node.updated = time.time()
        if node.active:
            self.index.add([self._doc(node)])
        else:
            self.index.remove([node.id])

    def link(self, type_: str, src: str, dst: str, weight: float = 1.0) -> None:
        if not any(e.type == type_ and e.src == src and e.dst == dst for e in self.edges):
            self.edges.append(MemEdge(type_, src, dst, weight))

    def supersede(self, old: MemNode, new: MemNode) -> None:
        old.status, old.superseded_by = "superseded", new.id
        self.touch(old)
        for e in self.edges:  # 邊跟著改指向新節點，連通性不因更新而斷裂
            if e.src == old.id:
                e.src = new.id
            if e.dst == old.id:
                e.dst = new.id

    def retire(self, node: MemNode, reason: str) -> None:
        node.status = "retired"
        node.data["retired_reason"] = reason
        self.touch(node)

    # ---------- 查詢 ----------
    def get(self, node_id: str) -> MemNode | None:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str, type_: str, outgoing: bool = True) -> list[MemNode]:
        ids = [e.dst if outgoing else e.src for e in self.edges
               if e.type == type_ and (e.src if outgoing else e.dst) == node_id]
        return [self.nodes[i] for i in dict.fromkeys(ids) if i in self.nodes and self.nodes[i].active]

    def search(self, query: str, layer: str, k: int = 5, min_score: float = 0.0, site: str | None = None,
               dense_only: bool = False) -> list[tuple[MemNode, float]]:
        where: dict[str, Any] = {"layer": layer}
        if site:
            where["site"] = site
        hits = self.index.search(query, k=k * 3 if dense_only else k, where=where)
        scored = [(self.nodes[h.doc.id], h.dense if dense_only else h.score) for h in hits]
        scored = [(n, s) for n, s in scored if s >= min_score and n.active]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def similarity(self, a: MemNode, b: MemNode) -> float:
        va, vb = self.index.vectors.get(a.id), self.index.vectors.get(b.id)
        if va is None or vb is None:
            va, vb = self.embedder.embed([a.search_text(), b.search_text()])
        return cosine(va, vb)

    def active(self, layer: str) -> list[MemNode]:
        return [n for n in self.nodes.values() if n.layer == layer and n.active]

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {layer: len(self.active(layer)) for layer in LAYERS}
        out["superseded"] = sum(1 for n in self.nodes.values() if n.status == "superseded")
        out["retired"] = sum(1 for n in self.nodes.values() if n.status == "retired")
        out["edges"] = {t: sum(1 for e in self.edges if e.type == t) for t in ("ground", "distill")}
        out["backend"] = self.index.backend
        out["embedder"] = self.embedder.name
        out.update({k: v for k, v in self.meta.items() if k != "ingested"})
        return out

    def _doc(self, n: MemNode) -> Doc:
        return Doc(n.id, n.search_text(), {"layer": n.layer, "site": n.site})
