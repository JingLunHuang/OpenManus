"""網頁搜尋：可插拔的搜尋提供方，按配置順序逐個嘗試。

  browser:bing / browser:baidu —— 複用靈犀自己的瀏覽器，在獨立標籤頁裡開啟搜尋結果頁並解析，
                                  零額外依賴，也不會打斷當前頁面
  ddgs                        —— 可選依賴 ddgs（DuckDuckGo）
  llm                         —— 模型自帶聯網搜尋（如 DashScope 的 enable_search）

搜到的結果會作為"線索"返回；要獲取可靠資料仍需 web_open 開啟來源頁核實。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from lingxi.hanzi import to_trad, to_trad_many
from lingxi.llm.messages import Message
from lingxi.skills.base import Outcome, Skill
from lingxi.utils import clip

if TYPE_CHECKING:
    from lingxi.kernel.context import RunContext


@dataclass
class Hit:
    title: str
    url: str
    snippet: str = ""


_ENGINES = {
    "bing": (
        "https://cn.bing.com/search?q={q}&setlang=zh-CN",
        """() => [...document.querySelectorAll('#b_results > li.b_algo')].map(li => ({
            title: (li.querySelector('h2') || {}).innerText || '',
            url: (li.querySelector('h2 a') || {}).href || '',
            snippet: (li.querySelector('.b_caption p, .b_lineclamp2, .b_paractl') || {}).innerText || ''
        }))""",
    ),
    "baidu": (
        "https://www.baidu.com/s?wd={q}",
        """() => [...document.querySelectorAll('#content_left .result, #content_left .result-op')].map(d => ({
            title: (d.querySelector('h3') || {}).innerText || '',
            url: (d.querySelector('h3 a') || {}).href || d.getAttribute('mu') || '',
            snippet: (d.querySelector('[class*="content-right"], .c-abstract, [class*="summary"]') || {}).innerText || ''
        }))""",
    ),
}


async def _browser_search(ctx: "RunContext", engine: str, query: str, limit: int) -> list[Hit]:
    url_tpl, js = _ENGINES[engine]
    async with ctx.browser.side_page() as page:  # 獨立標籤頁，不打斷當前頁面
        await page.goto(url_tpl.format(q=quote_plus(query)), wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)
        rows = await page.evaluate(js)
    return [Hit(r["title"].strip(), r["url"], r["snippet"].strip()) for r in rows if r.get("url")][:limit]


async def _ddgs_search(query: str, limit: int) -> list[Hit]:
    from ddgs import DDGS  # 可選依賴

    rows = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=limit)))
    return [Hit(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows]


async def _llm_search(ctx: "RunContext", query: str) -> str:
    reply = await ctx.llm.chat(
        [Message.user(f"請聯網搜尋並回答，列出資訊來源網址：{query}")],
        extra_body={"enable_search": True},
    )
    return reply.content


class SearchParams(BaseModel):
    query: str = Field(description="搜尋關鍵詞")
    limit: int = Field(6, ge=1, le=10, description="返回條數")


class WebSearch(Skill):
    name = "web_search"
    description = "用搜尋引擎查詢線索，返回標題、連結和摘要。需要可靠資料時請再用 web_open 開啟來源頁面核實。"
    Params = SearchParams

    async def run(self, ctx, p: SearchParams) -> Outcome:
        from lingxi.retrieval.focus import SearchCache

        cache = SearchCache.for_settings(ctx.settings)
        cached = cache.lookup(p.query) if cache else None
        if cached:
            hits = [Hit(**h) for h in cached["hits"]][: p.limit]
            age = max(1, int((time.time() - cached["ts"]) // 60))
            lines = [f"{i}. {clip(h.title, 60)}\n   {h.url}\n   {clip(h.snippet, 160)}" for i, h in enumerate(hits, 1)]
            lines.append(f"（本地搜尋快取：{age} 分鐘前「{cached['query']}」的結果；需要最新資料請用 web_open 開來源頁）")
            ctx.emit("cache", kind="search", query=p.query, matched=cached["query"], age_minutes=age)
            return Outcome(ok=True, summary=f"「{p.query}」命中本地快取 {len(hits)} 條（{age} 分鐘前）",
                           detail="\n".join(lines), progress=["results"], data={"hits": cached["hits"], "cached": True})
        errors = []
        for provider in ctx.settings.search.providers:
            try:
                if provider.startswith("browser:"):
                    hits = await _browser_search(ctx, provider.split(":", 1)[1], p.query, p.limit)
                elif provider == "ddgs":
                    hits = await _ddgs_search(p.query, p.limit)
                elif provider == "llm":
                    text = to_trad(await _llm_search(ctx, p.query))
                    return Outcome(ok=True, summary=f"模型聯網搜尋「{p.query}」", detail=text, progress=["results"])
                else:
                    errors.append(f"{provider}: 未知的搜尋提供方")
                    continue
            except Exception as exc:
                errors.append(f"{provider}: {clip(str(exc), 100)}")
                continue
            if hits:
                # 搜尋結果的標題與摘要一律轉成繁體呈現（網址保留原樣）
                conv = to_trad_many([h.title for h in hits] + [h.snippet for h in hits])
                hits = [Hit(conv[i], h.url, conv[len(hits) + i]) for i, h in enumerate(hits)]
                lines = [f"{i}. {clip(h.title, 60)}\n   {h.url}\n   {clip(h.snippet, 160)}" for i, h in enumerate(hits, 1)]
                if cache:
                    cache.store(p.query, provider, [h.__dict__ for h in hits])
                return Outcome(ok=True, summary=f"「{p.query}」找到 {len(hits)} 條結果（{provider}）",
                               detail="\n".join(lines), progress=["results"],
                               data={"hits": [h.__dict__ for h in hits]})
            errors.append(f"{provider}: 沒有結果")
        return Outcome.fail(f"所有搜尋提供方都失敗了：「{p.query}」", "\n".join(errors))
