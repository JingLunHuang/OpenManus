"""檢索共用的斷詞與切塊。

斷詞：中文取「單字 ＋ 相鄰雙字」，英數取整詞（保留 MU5101、12:30、example.com 這類完整記號）。
一律先 unify()，所以繁體查詢能命中簡體內容、「搜尋」能命中「搜索」。
不依賴 jieba / nltk / tiktoken 等需要下載資料的斷詞器，離線可用。
"""

from __future__ import annotations

import re

from lingxi.hanzi import unify

_CJK = "㐀-鿿豈-﫿"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-z0-9]+(?:[._:/-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[。！？；!?;\n])")


def _is_cjk(ch: str) -> bool:
    return "㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿"


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(unify((text or "").lower())):
        run = m.group()
        if _is_cjk(run[0]):
            out.extend(run)
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            out.append(run)
    return out


def estimate_tokens(text: str) -> int:
    """粗估 token 數：中文約 1 字 1 token，英數約 4 字元 1 token。只用於門檻判斷與報表。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + max(0, len(text) - cjk) // 4


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


def split_text(text: str, chunk_chars: int = 600, overlap: int = 80) -> list[str]:
    """按句子切塊。安裝了 LlamaIndex 時交給它的 SentenceSplitter（仍以字元計長、用中文標點斷句），
    否則用內建的同規則切法；兩者輸出的塊長度上限一致。"""
    if len(text) <= chunk_chars:
        return [text] if text.strip() else []
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=chunk_chars, chunk_overlap=overlap, tokenizer=list,
                                    chunking_tokenizer_fn=sentences)
        return [c for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        pass
    chunks: list[str] = []
    buf = ""
    for sent in sentences(text):
        while len(sent) > chunk_chars:  # 超長句硬切
            chunks.append((buf + sent[: chunk_chars - len(buf)]).strip())
            sent = sent[chunk_chars - len(buf):]
            buf = ""
        if len(buf) + len(sent) > chunk_chars and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap:] if overlap else ""
        buf += sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks
