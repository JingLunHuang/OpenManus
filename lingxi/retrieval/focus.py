"""檢索加速的兩個落點：長網頁精讀聚焦、搜尋結果快取。

focus_text()   web_read 讀長網頁時，不再把 12,000 字整塊丟給模型：
               切塊 → 混合檢索挑出與目標最相關的幾段（永遠保留第 1 段的頁首脈絡）→ 按原順序拼回去。
               送出的字數通常降到三分之一左右，模型呼叫更快、更便宜，而且一次就能看到分散在全文各處的相關段落。
SearchCache    web_search 的結果按查詢建索引存在 memory/search_cache.jsonl；
               同一查詢（或向量相似度 ≥ 門檻且「數字完全相同」的查詢）在時效內直接回傳，幾毫秒取代數秒的瀏覽器搜尋。
               數字必須相同：「6月26日 上海到北京機票」絕不會命中「6月27日」的快取。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from lingxi.hanzi import unify
from lingxi.retrieval.embed import HashEmbedder
from lingxi.retrieval.index import Doc, HybridIndex
from lingxi.retrieval.text import split_text

_NUM = re.compile(r"\d+")


def focus_text(text: str, query: str, rs, chunk_chars: int = 800) -> tuple[str, dict[str, Any]]:
    chunks = split_text(text, chunk_chars, 100)
    index = HybridIndex(HashEmbedder(rs.embedding_dim), rs.backend, rs.dense_weight, rs.bm25_weight)
    index.add([Doc(str(i), c) for i, c in enumerate(chunks)])
    hits = index.search(query, k=rs.read_focus_chunks)
    keep = sorted({0} | {int(h.doc.id) for h in hits})
    body = "\n…\n".join(f"〔第 {i + 1}/{len(chunks)} 段〕{chunks[i]}" for i in keep)
    return body, {"chunks": len(chunks), "kept": len(keep), "chars_total": len(text), "chars_sent": len(body)}


def _norm(q: str) -> str:
    return re.sub(r"\s+", "", unify(q.lower()))


class SearchCache:
    _instances: dict[Path, "SearchCache"] = {}

    @classmethod
    def for_settings(cls, settings) -> "SearchCache | None":
        if settings.retrieval.search_cache_hours <= 0:
            return None
        path = settings.memory_dir / "search_cache.jsonl"
        if path not in cls._instances:
            cls._instances[path] = cls(path, settings.retrieval)
        return cls._instances[path]

    def __init__(self, path: Path, rs):
        self.path = path
        self.rs = rs
        self.rows: dict[str, dict[str, Any]] = {}
        self.index = HybridIndex(HashEmbedder(rs.embedding_dim), rs.backend, 1.0, 0.0)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[_norm(row["query"])] = row
            self.index.add([Doc(k, r["query"]) for k, r in self.rows.items()])

    def lookup(self, query: str) -> dict[str, Any] | None:
        max_age = self.rs.search_cache_hours * 3600
        key = _norm(query)
        row = self.rows.get(key)
        if row is None:
            nums = _NUM.findall(key)
            for hit in self.index.search(query, k=3):
                cand = self.rows.get(hit.doc.id)
                if cand and hit.dense >= self.rs.search_cache_min_score and _NUM.findall(hit.doc.id) == nums:
                    row = cand
                    break
        if row is None or time.time() - row["ts"] > max_age:
            return None
        return row

    def store(self, query: str, provider: str, hits: list[dict[str, Any]]) -> None:
        row = {"query": query, "provider": provider, "hits": hits, "ts": time.time()}
        key = _norm(query)
        self.rows[key] = row
        self.index.add([Doc(key, query)])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
