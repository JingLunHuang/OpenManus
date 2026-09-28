"""向量化（Embedding）。

  HashEmbedder —— 預設。對 tokens() 做帶正負號的特徵雜湊（feature hashing），確定性、離線、零費用；
                  對「同一站點 / 同一類任務」這種字面重合度高的經驗召回已經足夠，測試也用它。
  ApiEmbedder  —— OpenAI 相容的 /embeddings 端點（DashScope text-embedding-v4 等）。
                  向量按文字雜湊快取在磁碟上，同一段記憶只付一次費用，重建索引不再呼叫 API。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

from lingxi.retrieval.text import tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def cosine(a: list[float], b: list[float]) -> float:
    """兩個向量都已 L2 正規化時，點積即餘弦相似度。"""
    return sum(x * y for x, y in zip(a, b))


def max_previous_similarity(vecs: list[list[float]]) -> list[float]:
    """第 i 個向量與它之前所有向量的最大相似度（第一個為 0）。有 numpy 時一次矩陣乘法算完。"""
    if not vecs:
        return []
    try:
        import numpy as np

        m = np.asarray(vecs, dtype=np.float32)
        sims = m @ m.T
        sims[np.triu_indices(len(vecs))] = -np.inf  # 只看「之前」的向量
        out = sims.max(axis=1)
        out[0] = 0.0
        return [float(x) for x in out]
    except ImportError:
        return [max((cosine(vecs[i], vecs[j]) for j in range(i)), default=0.0) for i in range(len(vecs))]


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok, count in Counter(tokens(text)).items():
            h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
            weight = (1.0 + math.log(count)) * (1.5 if len(tok) > 1 else 1.0)  # 雙字與整詞比單字更有鑑別力
            v[h % self.dim] += weight if (h >> 40) & 1 else -weight
        return normalize(v)


class ApiEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str = "text-embedding-v4", dim: int = 512,
                 cache_path: Path | None = None, batch: int = 10):
        self.name = f"api:{model}"
        self.dim = dim
        self.model = model
        self.batch = batch  # DashScope 每批最多 10 條
        self._client_args = {"base_url": base_url, "api_key": api_key}
        self._client = None
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path and cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["k"]] = row["v"]

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.dim}|{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(**self._client_args)
            fresh = []
            for i in range(0, len(missing), self.batch):
                chunk = missing[i:i + self.batch]
                resp = self._client.embeddings.create(model=self.model, input=[t for _, t in chunk],
                                                      dimensions=self.dim)
                for (k, _), item in zip(chunk, resp.data):
                    vec = normalize(list(item.embedding))
                    self._cache[k] = vec
                    fresh.append({"k": k, "v": vec})
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    for row in fresh:
                        f.write(json.dumps(row) + "\n")
        return [self._cache[k] for k in keys]
