"""檢索層：LlamaIndex 混合檢索（向量 ＋ BM25），供記憶召回、搜尋快取、長網頁精讀共用。"""

from lingxi.retrieval.embed import ApiEmbedder, Embedder, HashEmbedder, cosine
from lingxi.retrieval.index import Doc, Hit, HybridIndex, llamaindex_available
from lingxi.retrieval.text import estimate_tokens, split_text, tokens


def make_embedder(settings, cache_dir=None) -> Embedder:
    """依 [retrieval] 設定建立向量化器；embedding = "api" 但沒有金鑰時退回雜湊向量。"""
    r = settings.retrieval
    if r.embedding == "api":
        key = settings.llm.resolve_api_key()
        if key:
            cache = (cache_dir / "embeddings.jsonl") if cache_dir else None
            return ApiEmbedder(settings.llm.base_url, key, r.embedding_model, r.embedding_dim, cache)
    return HashEmbedder(r.embedding_dim)


__all__ = ["ApiEmbedder", "Doc", "Embedder", "HashEmbedder", "Hit", "HybridIndex", "cosine",
           "estimate_tokens", "llamaindex_available", "make_embedder", "split_text", "tokens"]
