"""真實瀏覽器測試：在本地夾具頁上驗證感知層與「定位 → 執行 → 驗證」。

這些正是課程裡在攜程上踩過的坑，這裡全部離線復現：
  - 日曆格子是 div + addEventListener，browser-use 的元素列表裡看不到；
  - 城市輸入必須從聯想候選裡點選，否則失焦回滾；
  - 登入遮罩擋住整個頁面。
"""

import asyncio

import pytest

from conftest import ScriptedLLM, needs_browser
from lingxi.hanzi import fold, to_trad, unify
from lingxi.journal.recorder import Journal
from lingxi.kernel.context import RunContext
from lingxi.kernel.profiles import PROFILES
from lingxi.skills.web import WebClick, WebOpen, WebType

pytestmark = [pytest.mark.browser, needs_browser]


def make_ctx(settings, profile="web_query"):
    return RunContext(task="t", settings=settings, journal=Journal(), llm=ScriptedLLM([]),
                      profile=PROFILES[profile], briefed_task="t")


def run(coro):
    return asyncio.run(coro)


def test_snapshot_sees_div_calendar_with_month_context(settings, site):
    async def body():
        ctx = make_ctx(settings)
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/flight.html"})
            snap = await ctx.browser.snapshot()
            # 夾具頁模擬攜程，是簡體；用 fold() 比對，繁體寫法也能對上
            inputs = {fold(e.label): e for e in snap.elements if e.kind == "input"}
            assert inputs[fold("出發城市")].value == "新加坡"  # 無 placeholder 也能拿到旁邊的標籤
            assert fold("到達城市") in inputs

            out = await WebClick().invoke(ctx, {"target": "出發日期"})
            assert out.ok and out.verified, out.summary
            snap = await ctx.browser.snapshot()
            dates = [e for e in snap.elements if e.kind == "date"]
            june = [e for e in dates if e.date["month"] == 6]
            july = [e for e in dates if e.date["month"] == 7]
            assert len(june) == 28 and len(july) == 31  # 6月1、2日不可選（不可點選，不列出）
            assert all(e.date["year"] == 2027 for e in dates)
            assert "〔2027年6月〕" in snap.render()
        finally:
            await ctx.close()

    run(body())


def test_click_date_by_natural_language_is_verified(settings, site):
    async def body():
        ctx = make_ctx(settings)
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/flight.html"})
            await WebClick().invoke(ctx, {"target": "出發日期"})
            out = await WebClick().invoke(ctx, {"target": "6月26日"})
            assert out.ok and out.verified
            assert out.grounding["method"] == "date"
            page = await ctx.browser.page()
            assert await page.inner_text("#date-text") == "2027-06-26"
        finally:
            await ctx.close()

    run(body())


def test_type_reads_back_value_and_lists_suggestions(settings, site):
    async def body():
        ctx = make_ctx(settings)
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/flight.html"})
            typed = await WebType().invoke(ctx, {"target": "出發城市", "text": "上海"})
            assert typed.ok and typed.verified, typed.detail
            assert "出現聯想候選" in typed.detail and "上海(SHA)" in typed.detail

            picked = await WebClick().invoke(ctx, {"target": "上海(SHA)"})
            assert picked.ok and picked.verified
            page = await ctx.browser.page()
            assert await page.eval_on_selector("#from", "el => el.dataset.code") == "SHA"
        finally:
            await ctx.close()

    run(body())


def test_type_without_choosing_suggestion_is_caught(settings, site):
    """復現"輸入了上海，出發城市卻還是新加坡"：失焦回滾後，簡報裡能看到真實值。"""

    async def body():
        ctx = make_ctx(settings)
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/flight.html"})
            await WebType().invoke(ctx, {"target": "出發城市", "text": "上海"})
            await WebClick().invoke(ctx, {"target": "搜尋"})  # 沒選候選就去點別處
            snap = await ctx.browser.snapshot()
            city = next(e for e in snap.elements if fold(e.label) == fold("出發城市"))
            assert city.value == "新加坡"
            assert fold("出發城市=新加坡") in fold(snap.render())
        finally:
            await ctx.close()

    run(body())


def test_overlay_is_detected_and_close_button_suggested(settings, site):
    async def body():
        ctx = make_ctx(settings)
        try:
            opened = await WebOpen().invoke(ctx, {"url": f"{site}/flight.html?login=1"})
            assert opened.verified  # 登入框裡的"獲取驗證碼"不應被誤判為反爬挑戰
            snap = await ctx.browser.snapshot()
            assert not snap.challenge
            assert snap.overlay and snap.overlay["coveredCount"] >= 3
            close_ids = snap.overlay["closeIds"]
            assert close_ids, snap.render()
            assert "浮層遮擋" in snap.render()
            out = await WebClick().invoke(ctx, {"target": f"#{close_ids[0]}"})
            assert out.verified
            assert (await ctx.browser.snapshot()).overlay is None
        finally:
            await ctx.close()

    run(body())


def test_ambiguous_target_returns_candidates_instead_of_guessing(settings, site):
    async def body():
        ctx = make_ctx(settings)
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/flight.html"})
            await WebClick().invoke(ctx, {"target": "出發日期"})
            out = await WebClick().invoke(ctx, {"target": "26"})
            assert not out.ok and "多個元素" in out.summary and "#" in out.detail
        finally:
            await ctx.close()

    run(body())


def test_open_fixes_stale_dates_and_warms_up_site(settings, site):
    async def body():
        ctx = make_ctx(settings)
        ctx.browser.warmups["127.0.0.1"] = f"{site}/flight.html?warm=1"
        try:
            out = await WebOpen().invoke(ctx, {"url": f"{site}/flight.html?depdate=2023-06-26"})
            assert "已把過期日期順延" in out.detail and "depdate: 2023-06-26" in out.detail
            assert "建立會話" in out.detail
            page = await ctx.browser.page()
            assert "depdate=2023" not in page.url
        finally:
            await ctx.close()

    run(body())


def test_simplified_site_is_presented_in_traditional(settings, site):
    """簡體網站（flight-cn.html，由 unify() 即時產生）：

    - 快照裡的標題、標籤、正文摘要全部以繁體呈現，並標示原網頁為簡體；
    - 模型用繁體描述目標（出發城市／搜尋按鈕）照樣能定位；
    - 用繁體輸入「廣州」時，會自動轉寫成簡體送進網頁，網站自己的聯想搜尋因此對得上。
    """

    async def body():
        ctx = make_ctx(settings)
        try:
            opened = await WebOpen().invoke(ctx, {"url": f"{site}/flight-cn.html"})
            # 網頁原文經 unify() 產生（用詞已是大陸說法）；呈現時只轉字形、保留網站原本的用詞
            assert to_trad(unify("模擬機票預訂")) in opened.summary
            snap = await ctx.browser.snapshot()
            assert snap.script == "simplified"
            page_text = snap.render()
            assert "原網頁為簡體中文" in page_text and "出發城市" in page_text and "請選擇日期" in page_text
            # 呈現給模型與人看的文字裡，不再有網頁原本的簡體字樣
            assert fold("出發城市") not in page_text and fold("請選擇日期") not in page_text

            typed = await WebType().invoke(ctx, {"target": "到達城市", "text": "廣州"})
            assert typed.ok and typed.verified and typed.grounding["label"] == "到達城市"
            assert "已自動轉寫成簡體輸入" in typed.detail and "廣州(CAN)" in typed.detail
            page = await ctx.browser.page()
            assert await page.eval_on_selector("#to", "el => el.value") == fold("廣州")  # 網頁實際收到簡體

            picked = await WebClick().invoke(ctx, {"target": "廣州(CAN)"})
            assert picked.ok and picked.verified
            assert await page.eval_on_selector("#to", "el => el.dataset.code") == "CAN"

            clicked = await WebClick().invoke(ctx, {"target": "搜尋按鈕"})
            assert clicked.ok and clicked.grounding["label"] == to_trad(unify("搜尋"))  # 網站寫的是「搜索」
        finally:
            await ctx.close()

    run(body())


def test_web_read_focuses_on_relevant_paragraphs_of_a_long_page(settings, site):
    """長網頁：只把與目標最相關的段落送給模型，而且藏在第 12,000 字之後的關鍵段落一次就讀到。"""
    from conftest import reply

    from lingxi.skills.web import WebRead

    async def body():
        ctx = make_ctx(settings)
        ctx.llm = ScriptedLLM([reply(content='{"found": true, "facts": ["24 小時內退票收取 20% 手續費"], '
                                              '"answer": "20%", "corroborates": [], "contradicts": []}')])
        try:
            await WebOpen().invoke(ctx, {"url": f"{site}/long.html"})
            out = await WebRead().invoke(ctx, {"goal": "退票手續費是多少"})
            prompt = ctx.llm.requests[0][0][-1].content
            assert out.ok and out.data["focus"]["chars_total"] > 12000
            assert "20% 的手續費" in prompt and "最相關" in prompt
            assert len(prompt) < 9000  # 舊做法一次送 12,000 字，而且讀不到這段
            assert any(e.kind == "focus" for e in ctx.journal.events)
            assert "full=true" in out.detail
        finally:
            await ctx.close()

    run(body())
