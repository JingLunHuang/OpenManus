"""檢索層：LlamaIndex 與純 Python 後端排序一致、繁簡通吃、長網頁聚焦、搜尋快取不誤命中。"""

from __future__ import annotations

import time

import pytest

from lingxi.hanzi import fold
from lingxi.retrieval import Doc, HashEmbedder, HybridIndex, llamaindex_available, split_text, tokens
from lingxi.retrieval.focus import SearchCache, focus_text
from lingxi.settings import RetrievalSettings

BACKENDS = ["builtin"] + (["llamaindex"] if llamaindex_available() else [])

DOCS = [
    Doc("s1", "攜程機票：出發城市輸入後必須點選聯想候選才會生效", {"layer": "semantic", "site": "ctrip.com"}),
    Doc("s2", "12306 查餘票需要先登入，未登入只能看車次", {"layer": "semantic", "site": "12306.cn"}),
    Doc("e1", "在測試頁查詢 6月26日 從上海到北京的航班", {"layer": "episodic", "site": "127.0.0.1"}),
    Doc("s3", "攜程日期欄是 div 日曆，直接點日期格", {"layer": "semantic", "site": "ctrip.com"}),
]


@pytest.mark.parametrize("backend", BACKENDS)
def test_hybrid_search_filters_upserts_and_removes(backend):
    ix = HybridIndex(HashEmbedder(256), backend)
    ix.add(DOCS)
    top = ix.search("攜程 聯想候選", 3)
    assert top[0].doc.id == "s1" and top[0].sparse > 0 and top[0].dense > 0
    assert [h.doc.id for h in ix.search("查詢航班", 3, where={"layer": "episodic"})] == ["e1"]
    assert all(h.doc.meta["site"] == "ctrip.com" for h in ix.search("攜程", 5, where={"site": "ctrip.com"}))
    ix.add([Doc("s3", "攜程日期欄已改版為輸入框", {"layer": "semantic", "site": "ctrip.com"})])  # 同 id 覆蓋
    assert len(ix) == 4 and "改版" in ix.search("攜程 日期欄", 1)[0].doc.text
    ix.remove(["s1"])
    assert "s1" not in {h.doc.id for h in ix.search("聯想候選", 5)}


def test_backends_rank_identically():
    if len(BACKENDS) < 2:
        pytest.skip("沒有安裝 llama-index-core")
    docs = [Doc(f"d{i}", t, {"layer": "semantic"}) for i, t in enumerate(
        ["上海 天氣 晴", "北京 機票 價格", "上海 北京 航班 價格", "高鐵 餘票 查詢", "攜程 日曆 點選", "酒店 會員 優惠"] * 5)]
    results = []
    for backend in BACKENDS:
        ix = HybridIndex(HashEmbedder(256), backend)
        ix.add(docs)
        results.append([(h.doc.id, h.score) for h in ix.search("上海到北京的機票價格", 6)])
    assert results[0] == results[1]


def test_traditional_query_matches_simplified_text():
    ix = HybridIndex(HashEmbedder(256), "builtin")
    ix.add([Doc("cn", fold("出發城市輸入後必須點選聯想候選")), Doc("x", "天氣晴朗")])
    assert ix.search("出發城市 聯想候選", 1)[0].doc.id == "cn"
    assert set(tokens("搜尋")) == set(tokens(fold("搜尋")))


def test_split_text_respects_limit_and_keeps_content():
    text = "東方航空 MU5101 起飛。" * 200
    chunks = split_text(text, 300, 40)
    assert len(chunks) > 5 and all(len(c) <= 320 for c in chunks)
    assert "MU5101" in chunks[-1]


def test_focus_text_picks_the_relevant_paragraph_far_into_the_page():
    filler = "本站提供國內外機票、酒店預訂服務，客服電話全天候為您服務。" * 8
    key = "退改簽規則：起飛前 24 小時以上免費退票，24 小時內收取票面價 20% 的手續費。"
    text = "\n".join([filler] * 55 + [key] + [filler] * 10)
    body, info = focus_text(text, "退票手續費是多少", RetrievalSettings(backend="builtin"))
    assert text.find(key) > 12000  # 舊做法一次只讀前 12,000 字，讀不到
    assert "20% 的手續費" in body
    assert info["chars_sent"] < info["chars_total"] / 2 and "〔第 1/" in body  # 永遠保留頁首脈絡


def test_search_cache_hits_same_query_but_never_a_different_date(tmp_path):
    rs = RetrievalSettings(backend="builtin")
    cache = SearchCache(tmp_path / "c.jsonl", rs)
    cache.store("6月26日 上海到北京 機票", "browser:bing", [{"title": "t", "url": "https://e.com", "snippet": "s"}])
    assert cache.lookup("6月26日  上海到北京 機票")  # 空白不同仍命中
    assert cache.lookup("6月27日 上海到北京 機票") is None  # 數字不同絕不命中
    reloaded = SearchCache(tmp_path / "c.jsonl", rs)  # 落盤後重新載入
    assert reloaded.lookup("6月26日 上海到北京 機票")["provider"] == "browser:bing"
    reloaded.rows[next(iter(reloaded.rows))]["ts"] = time.time() - 13 * 3600  # 過期
    assert reloaded.lookup("6月26日 上海到北京 機票") is None
