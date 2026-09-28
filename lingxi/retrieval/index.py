"""混合檢索索引：稠密向量 ＋ 稀疏 BM25。

打分沿用 FluxMem Stage I 的加權相加（dense_weight=1.0、bm25_weight=0.5）：

    score = w_d · max(cos, 0) ＋ w_s · BM25 / BM25_上界

BM25 除以「這個查詢可能得到的最高分」Σ idf·(k1+1)，得到 [0, 1) 的絕對分數；
不用 min-max 正規化，是因為只有一個候選時 min-max 會把它硬拉成滿分，無法判斷「到底相不相關」。

兩個後端，排序語義完全相同（稠密部分都是精確餘弦）：
  llamaindex —— 節點、元資料篩選、檢索流程交給 llama-index-core 的 VectorStoreIndex，
                向量庫換成自訂的 MatrixVectorStore（實作 LlamaIndex 的 BasePydanticVectorStore 介面）：
                全部向量放在一個 numpy 矩陣裡，一次矩陣乘法 ＋ argpartition 取前 k 名。
                LlamaIndex 內建的 SimpleVectorStore 逐筆迴圈算相似度，5000 筆時比純 Python 還慢，因此不用它；
                需要擴充到百萬級時，同一個介面可換成 Qdrant / FAISS / Milvus 等 LlamaIndex 向量庫整合
  builtin    —— 純 Python 暴力餘弦，沒裝 LlamaIndex 時自動退回
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from lingxi.retrieval.embed import Embedder, cosine
from lingxi.retrieval.text import tokens


@dataclass
class Doc:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    dense: float
    sparse: float


def llamaindex_available() -> bool:
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str) -> str:
    if backend == "builtin":
        return "builtin"
    if backend == "llamaindex" and not llamaindex_available():
        raise RuntimeError("retrieval.backend = llamaindex，但沒有安裝：pip install -e \".[rag]\"")
    return "llamaindex" if llamaindex_available() else "builtin"


class BM25:
    """可增刪的 BM25（k1=1.5, b=0.75）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.length: dict[str, int] = {}
        self.df: Counter = Counter()
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.total_len = 0

    def add(self, doc_id: str, toks: list[str]) -> None:
        self.remove(doc_id)
        tf = Counter(toks)
        self.tf[doc_id] = tf
        self.length[doc_id] = len(toks)
        self.total_len += len(toks)
        for t in tf:
            self.df[t] += 1
            self.postings[t].add(doc_id)

    def remove(self, doc_id: str) -> None:
        tf = self.tf.pop(doc_id, None)
        if tf is None:
            return
        self.total_len -= self.length.pop(doc_id)
        for t in tf:
            self.df[t] -= 1
            self.postings[t].discard(doc_id)

    def idf(self, t: str) -> float:
        n = len(self.tf)
        return math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def scores(self, query: list[str], allowed: set[str] | None = None) -> dict[str, float]:
        if not self.tf:
            return {}
        avg = self.total_len / len(self.tf)
        out: dict[str, float] = defaultdict(float)
        for t in set(query):
            idf = self.idf(t)
            for doc_id in self.postings.get(t, ()):
                if allowed is not None and doc_id not in allowed:
                    continue
                f = self.tf[doc_id][t]
                dl = self.length[doc_id]
                out[doc_id] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
        return out

    def ceiling(self, query: list[str]) -> float:
        return sum(self.idf(t) * (self.k1 + 1) for t in set(query)) or 1.0


class HybridIndex:
    def __init__(self, embedder: Embedder, backend: str = "auto", dense_weight: float = 1.0,
                 sparse_weight: float = 0.5):
        self.embedder = embedder
        self.backend = resolve_backend(backend)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.docs: dict[str, Doc] = {}
        self.vectors: dict[str, list[float]] = {}
        self.bm25 = BM25()
        self._li = None  # llama_index VectorStoreIndex

    def __len__(self) -> int:
        return len(self.docs)

    # ---------- 寫入 ----------
    def add(self, docs: list[Doc]) -> None:
        docs = [d for d in docs if d.text.strip()]
        if not docs:
            return
        self.remove([d.id for d in docs if d.id in self.docs])
        vecs = self.embedder.embed([d.text for d in docs])
        for d, v in zip(docs, vecs):
            self.docs[d.id] = d
            self.vectors[d.id] = v
            self.bm25.add(d.id, tokens(d.text))
        if self.backend == "llamaindex":
            self._li_insert(docs, vecs)

    def remove(self, ids: list[str]) -> None:
        ids = [i for i in ids if i in self.docs]
        for i in ids:
            self.docs.pop(i)
            self.vectors.pop(i)
            self.bm25.remove(i)
        if ids and self._li is not None:
            self._li.delete_nodes(ids, delete_from_docstore=True)

    # ---------- 查詢 ----------
    def search(self, query: str, k: int = 5, where: dict[str, Any] | None = None,
               min_score: float = 0.0) -> list[Hit]:
        if not self.docs or not query.strip():
            return []
        allowed = {i for i, d in self.docs.items() if _matches(d.meta, where)} if where else None
        if allowed is not None and not allowed:
            return []
        pool = max(k * 4, 20)
        qvec = self.embedder.embed([query])[0]
        dense = self._li_dense(query, qvec, pool, where, allowed) if self.backend == "llamaindex" \
            else self._builtin_dense(qvec, pool, allowed)
        qtoks = tokens(query)
        raw = self.bm25.scores(qtoks, allowed)
        ceiling = self.bm25.ceiling(qtoks)
        sparse = {i: s / ceiling for i, s in sorted(raw.items(), key=lambda kv: -kv[1])[:pool]}
        hits = []
        for i in set(dense) | set(sparse):
            d = dense.get(i)
            if d is None:
                d = cosine(qvec, self.vectors[i])
            s = sparse.get(i, 0.0)
            score = self.dense_weight * max(d, 0.0) + self.sparse_weight * s
            if score >= min_score:
                hits.append(Hit(self.docs[i], round(score, 4), round(d, 4), round(s, 4)))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def _builtin_dense(self, qvec, pool, allowed) -> dict[str, float]:
        ids = allowed if allowed is not None else self.docs.keys()
        scored = sorted(((cosine(qvec, self.vectors[i]), i) for i in ids), reverse=True)[:pool]
        return {i: s for s, i in scored}

    # ---------- LlamaIndex 後端 ----------
    def _li_insert(self, docs: list[Doc], vecs: list[list[float]]) -> None:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import TextNode

        nodes = []
        for d, v in zip(docs, vecs):
            meta = {k: val for k, val in d.meta.items() if isinstance(val, (str, int, float, bool))}
            nodes.append(TextNode(id_=d.id, text=d.text, embedding=v, metadata=meta,
                                  excluded_embed_metadata_keys=list(meta), excluded_llm_metadata_keys=list(meta)))
        if self._li is None:
            from llama_index.core import StorageContext

            storage = StorageContext.from_defaults(vector_store=_matrix_store_cls()())
            self._li = VectorStoreIndex(nodes, storage_context=storage, embed_model=_LXEmbedding.wrap(self.embedder))
        else:
            self._li.insert_nodes(nodes)

    def _li_dense(self, query, qvec, pool, where, allowed) -> dict[str, float]:
        from llama_index.core.schema import QueryBundle
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

        filters = None
        scalar = {k: v for k, v in (where or {}).items() if isinstance(v, (str, int, float, bool))}
        if scalar:
            filters = MetadataFilters(filters=[MetadataFilter(key=k, value=v) for k, v in scalar.items()])
        retriever = self._li.as_retriever(similarity_top_k=pool, filters=filters)
        out = {}
        for nws in retriever.retrieve(QueryBundle(query_str=query, embedding=qvec)):
            if allowed is None or nws.node.node_id in allowed:
                out[nws.node.node_id] = float(nws.score or 0.0)
        return out


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, want in where.items():
        have = meta.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


_MATRIX_STORE = None


def _matrix_store_cls():
    """建立 MatrixVectorStore 類別（延遲到第一次使用，未安裝 LlamaIndex 時不會匯入它）。"""
    global _MATRIX_STORE
    if _MATRIX_STORE is not None:
        return _MATRIX_STORE
    import numpy as np
    from llama_index.core.vector_stores.types import (BasePydanticVectorStore, FilterOperator,
                                                      VectorStoreQueryResult)
    from pydantic import PrivateAttr

    class MatrixVectorStore(BasePydanticVectorStore):
        """numpy 矩陣向量庫：向量按列存放，刪除時標記空位、空位過半才壓縮。"""

        stores_text: bool = False
        _matrix: Any = PrivateAttr(default=None)
        _ids: list = PrivateAttr(default_factory=list)
        _row: dict = PrivateAttr(default_factory=dict)
        _meta: dict = PrivateAttr(default_factory=dict)
        _alive: Any = PrivateAttr(default=None)

        @property
        def client(self) -> Any:
            return None

        def add(self, nodes, **kwargs) -> list[str]:
            vecs = np.asarray([n.get_embedding() for n in nodes], dtype=np.float32)
            if self._matrix is None:
                self._matrix = np.empty((0, vecs.shape[1]), dtype=np.float32)
                self._alive = np.empty(0, dtype=bool)
            start = self._matrix.shape[0]
            self._matrix = np.vstack([self._matrix, vecs])
            self._alive = np.concatenate([self._alive, np.ones(len(nodes), dtype=bool)])
            for offset, n in enumerate(nodes):
                if n.node_id in self._row:
                    self._alive[self._row[n.node_id]] = False
                self._row[n.node_id] = start + offset
                self._meta[n.node_id] = dict(n.metadata)
                self._ids.append(n.node_id)
            return [n.node_id for n in nodes]

        def delete(self, ref_doc_id: str, **kwargs) -> None:
            self.delete_nodes([ref_doc_id])

        def delete_nodes(self, node_ids=None, filters=None, **kwargs) -> None:
            for nid in node_ids or []:
                row = self._row.pop(nid, None)
                if row is not None:
                    self._alive[row] = False
                    self._meta.pop(nid, None)
            if self._alive is not None and len(self._alive) > 64 and self._alive.mean() < 0.5:
                keep = np.flatnonzero(self._alive)
                self._matrix = self._matrix[keep]
                self._ids = [self._ids[i] for i in keep]
                self._alive = np.ones(len(keep), dtype=bool)
                self._row = {nid: i for i, nid in enumerate(self._ids)}

        def clear(self) -> None:
            self._matrix, self._alive = None, None
            self._ids, self._row, self._meta = [], {}, {}

        def query(self, query, **kwargs) -> VectorStoreQueryResult:
            if self._matrix is None or not self._row:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            mask = self._alive.copy()
            if query.filters is not None:
                for nid, row in self._row.items():
                    if mask[row] and not _li_filter_ok(self._meta[nid], query.filters, FilterOperator):
                        mask[row] = False
            q = np.asarray(query.query_embedding, dtype=np.float32)
            sims = self._matrix @ q
            sims[~mask] = -np.inf
            k = min(query.similarity_top_k, int(mask.sum()))
            if k <= 0:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            return VectorStoreQueryResult(similarities=[float(sims[i]) for i in top],
                                          ids=[self._ids[i] for i in top])

    _MATRIX_STORE = MatrixVectorStore
    return MatrixVectorStore


def _li_filter_ok(meta: dict[str, Any], filters, ops) -> bool:
    results = []
    for f in filters.filters:
        have = meta.get(f.key)
        if f.operator == ops.EQ:
            results.append(have == f.value)
        elif f.operator == ops.NE:
            results.append(have != f.value)
        elif f.operator == ops.IN:
            results.append(have in (f.value or []))
        elif f.operator == ops.NIN:
            results.append(have not in (f.value or []))
        else:
            raise NotImplementedError(f"MatrixVectorStore 不支援篩選運算子 {f.operator}")
    cond = str(getattr(filters, "condition", "and")).lower()
    return any(results) if cond.endswith("or") else all(results)


class _LXEmbedding:
    """把靈犀的 Embedder 包成 LlamaIndex 的 BaseEmbedding（延遲建立類別，避免未安裝時匯入失敗）。"""

    _cls = None

    @classmethod
    def wrap(cls, embedder: Embedder):
        if cls._cls is None:
            from llama_index.core.embeddings import BaseEmbedding
            from pydantic import PrivateAttr

            class LingXiEmbedding(BaseEmbedding):
                _inner: Any = PrivateAttr()

                def __init__(self, inner, **kw):
                    super().__init__(model_name=inner.name, embed_batch_size=64, **kw)
                    self._inner = inner

                def _get_query_embedding(self, query: str) -> list[float]:
                    return self._inner.embed([query])[0]

                async def _aget_query_embedding(self, query: str) -> list[float]:
                    return self._get_query_embedding(query)

                def _get_text_embedding(self, text: str) -> list[float]:
                    return self._inner.embed([text])[0]

                def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                    return self._inner.embed(texts)

            cls._cls = LingXiEmbedding
        return cls._cls(embedder)
