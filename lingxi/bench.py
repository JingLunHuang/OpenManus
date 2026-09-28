"""加速基準（lingxi bench）：全部離線、確定性，不呼叫任何模型。

  1. KV 快取  ：同一批軌跡在「逐輪摺疊」與「分段摺疊」下的前綴重用率、有效計費量
  2. 檢索後端 ：5,000 條記憶上 LlamaIndex（MatrixVectorStore）與純 Python 的寫入 / 查詢耗時，以及前 5 名是否一致
  3. 精讀聚焦 ：一篇 2 萬字的長網頁，web_read 送給模型的字數，以及關鍵段落有沒有被選中
  4. 搜尋快取 ：命中快取的查詢耗時（對照：瀏覽器搜尋一次通常要數秒）
軌跡用的是與受保護評測不同的種子，基準數字不會洩漏評測集。
"""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path
from typing import Any

from lingxi.llm.kvcache import replay
from lingxi.retrieval import Doc, HashEmbedder, HybridIndex, llamaindex_available


def bench_kv(settings) -> dict[str, Any]:
    rng = random.Random(2026)
    traces = [[rng.randint(300, 5200) for _ in range(n)] for n in (12, 24, 40)]
    fold_after = settings.agent.fold_after_turns
    max_obs = settings.agent.max_observation_chars

    def run(block: int) -> dict[str, float]:
        rows = [replay(t, fold_after, block, max_obs) for t in traces]
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}

    before, after = run(1), run(settings.kv_cache.fold_block)
    return {"fold_block": settings.kv_cache.fold_block, "before": before, "after": after,
            "billed_saving": 1 - after["billed_chars"] / before["billed_chars"]}


def bench_retrieval(n: int = 5000, queries: int = 20) -> dict[str, Any]:
    rng = random.Random(7)
    words = "北京 上海 機票 價格 天氣 航班 酒店 攜程 日曆 登入 驗證碼 搜尋 報告 程式 資料 分析 會員 優惠 高鐵 餘票".split()
    docs = [Doc(f"d{i}", " ".join(rng.choices(words, k=12)), {"layer": "semantic"}) for i in range(n)]
    qs = [" ".join(rng.choices(words, k=3)) for _ in range(queries)]
    out: dict[str, Any] = {"docs": n}
    tops = {}
    for backend in ("builtin", "llamaindex"):
        if backend == "llamaindex" and not llamaindex_available():
            continue
        index = HybridIndex(HashEmbedder(512), backend)
        t0 = time.perf_counter()
        index.add(docs)
        t1 = time.perf_counter()
        tops[backend] = [[h.doc.id for h in index.search(q, 5)] for q in qs]
        t2 = time.perf_counter()
        out[backend] = {"add_s": round(t1 - t0, 2), "search_ms": round((t2 - t1) / queries * 1000, 1)}
    if len(tops) == 2:
        out["same_top5"] = tops["builtin"] == tops["llamaindex"]
        out["speedup"] = round(out["builtin"]["search_ms"] / max(out["llamaindex"]["search_ms"], 0.01), 1)
    return out


def bench_focus(settings) -> dict[str, Any]:
    from lingxi.retrieval.focus import focus_text

    rng = random.Random(3)
    filler = ["本站提供國內外機票、酒店、火車票預訂服務，客服電話全天候為您服務。",
              "會員積分可以兌換禮品，詳情請見會員中心的說明頁面。",
              "為保障您的帳戶安全，請勿將驗證碼告訴他人。",
              "熱門目的地推薦：東京、大阪、首爾、曼谷、新加坡，機票低至三折起。"]
    paras = [rng.choice(filler) * 10 for _ in range(60)]
    key = "退改簽規則：起飛前 24 小時以上免費退票，24 小時內收取票面價 20% 的手續費，起飛後不可退。"
    paras.insert(47, key)  # 關鍵段落藏在全文四分之三處，逐段讀要讀到第 2 塊之後才看得到
    text = "\n".join(paras)
    body, info = focus_text(text, "這張機票的退票手續費是多少", settings.retrieval)
    offset = text.find(key)
    return {**info, "baseline_chars": min(len(text), 12000), "found_key": key[:20] in body,
            "key_offset": offset, "baseline_found": offset + len(key) <= 12000}


def bench_search_cache(settings) -> dict[str, Any]:
    from lingxi.retrieval.focus import SearchCache

    with tempfile.TemporaryDirectory() as d:
        cache = SearchCache(Path(d) / "c.jsonl", settings.retrieval)
        for i in range(200):
            cache.store(f"查詢 {i} 號主題 最新消息", "browser:bing", [{"title": "t", "url": "https://e.com", "snippet": ""}])
        t0 = time.perf_counter()
        hit = cache.lookup("查詢 42 號主題 最新消息")
        exact_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        near = cache.lookup("查詢 42號主題最新消息")
        miss = cache.lookup("查詢 43 號主題 最新消息 以後")
        near_ms = (time.perf_counter() - t0) * 1000 / 2
    return {"entries": 200, "exact_hit": hit is not None, "exact_ms": round(exact_ms, 2),
            "near_hit": near is not None, "different_numbers_miss": miss is None or "43" in miss["query"],
            "lookup_ms": round(near_ms, 2)}


def run_all(settings) -> dict[str, Any]:
    return {"kv": bench_kv(settings), "retrieval": bench_retrieval(), "focus": bench_focus(settings),
            "search_cache": bench_search_cache(settings)}


def render(r: dict[str, Any]) -> str:
    kv, rt, fc, sc = r["kv"], r["retrieval"], r["focus"], r["search_cache"]
    lines = ["靈犀加速基準（離線、確定性，不呼叫模型）", "",
             f"① KV 快取：分段摺疊 fold_block={kv['fold_block']}",
             f"   前綴重用率   {kv['before']['reuse']:.1%} → {kv['after']['reuse']:.1%}",
             f"   平均請求     {kv['before']['avg_chars']:,.0f} → {kv['after']['avg_chars']:,.0f} 字元",
             f"   有效計費量   {kv['before']['billed_chars']:,.0f} → {kv['after']['billed_chars']:,.0f}"
             f"（命中按 20% 計價，省 {kv['billed_saving']:.1%}）", "",
             f"② 檢索後端（{rt['docs']:,} 條記憶，混合檢索 dense＋BM25）"]
    for b in ("builtin", "llamaindex"):
        if b in rt:
            lines.append(f"   {b:<11} 寫入 {rt[b]['add_s']:.2f}s · 每次查詢 {rt[b]['search_ms']:.1f} ms")
    if "speedup" in rt:
        lines.append(f"   LlamaIndex 查詢快 {rt['speedup']}× · 前 5 名與純 Python 完全一致：{'是' if rt['same_top5'] else '否'}")
    lines += ["", "③ web_read 長網頁聚焦",
              f"   原文 {fc['chars_total']:,} 字 → 送給模型 {fc['chars_sent']:,} 字（舊做法一次送 {fc['baseline_chars']:,} 字）",
              f"   挑出 {fc['kept']}/{fc['chunks']} 段；位於第 {fc['key_offset']:,} 字的關鍵段落"
              f"{'有被選中' if fc['found_key'] else '沒有被選中'}"
              f"（舊做法第一次{'讀得到' if fc['baseline_found'] else '讀不到，要再用 offset 讀第二塊'}）", "",
              "④ 搜尋快取",
              f"   {sc['entries']} 筆快取 · 完全相同查詢 {sc['exact_ms']:.2f} ms · 相近寫法 {'命中' if sc['near_hit'] else '未命中'}"
              f"（{sc['lookup_ms']:.2f} ms）· 數字不同的查詢不會誤命中：{'是' if sc['different_numbers_miss'] else '否'}"]
    return "\n".join(lines)
