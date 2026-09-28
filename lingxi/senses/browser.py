"""瀏覽器會話：直接基於 Playwright，不依賴 browser-use。

職責只有"感知與執行的原語"，不包含任何決策：
  snapshot()  —— 注入 dom_snapshot.js，得到結構化頁面觀察
  probe()     —— 極輕量的狀態探針（URL / DOM 變更計數 / 焦點 / 標籤頁數），用於動作前後對比
  settle()    —— 等待頁面"安靜下來"：載入完成 + DOM 變更停止
  goto()      —— 帶"會話預熱"的導航：首次造訪某站點時先開啟其首頁（繞開深鏈直達被風控的問題）
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lingxi.hanzi import fold, variants
from lingxi.senses.page import PageSnapshot
from lingxi.settings import BrowserSettings
from lingxi.utils import site_key  # noqa: F401  （沿用舊的匯入路徑）

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

log = logging.getLogger("lingxi.browser")

SNAPSHOT_JS = (Path(__file__).parent / "dom_snapshot.js").read_text(encoding="utf-8")

def _both_scripts(pattern: str) -> str:
    """把正規表示式裡每個有簡體寫法的字展開成「繁、簡兩種寫法」的字元類別，讓同一條規則同時比對繁、簡網頁。"""
    return "".join(f"[{c}{fold(c)}]" if fold(c) != c else c for c in pattern)


_CLOSE_WORDS = ["×", "✕", "✖", "關閉", "close", "跳過", "我知道了", "稍後", "暫不", "不再提示", "取消"]
_CHALLENGE = ["安全驗證", "人機驗證", "滑動.{0,6}驗證", "拖動.{0,6}滑塊", "訪問過於頻繁", "請求過於頻繁"]
SNAPSHOT_OPTIONS = {
    "closeWords": sorted({v for w in _CLOSE_WORDS for v in variants(w)}),
    "challenge": "|".join(_both_scripts(p) for p in _CHALLENGE)
    + "|captcha|are you a robot|verify you are human|unusual traffic",
    "daySuffix": "|".join(dict.fromkeys(variants("日") + variants("號"))),
}


_PROBE_JS = """() => ({
  url: location.href,
  title: document.title,
  mut: (window.__lx || {mut: 0}).mut,
  focused: document.activeElement && document.activeElement.getAttribute
    ? document.activeElement.getAttribute("data-lx-id") : null,
})"""

_QUIET_JS = """(opts) => new Promise((resolve) => {
  const box = window.__lx || { mut: 0 };
  let last = box.mut, stable = 0; const t0 = Date.now();
  const iv = setInterval(() => {
    const m = (window.__lx || box).mut;
    if (m === last) stable += 100; else { stable = 0; last = m; }
    if (stable >= opts.quiet || Date.now() - t0 > opts.max) { clearInterval(iv); resolve(m); }
  }, 100);
})"""

_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
window.chrome = window.chrome || { runtime: {} };
"""


class BrowserSession:
    def __init__(self, settings: BrowserSettings):
        self.settings = settings
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._owns_browser = True
        self.last_snapshot: PageSnapshot | None = None
        self.warmups: dict[str, str] = {}  # 站點 → 預熱首頁（來自手冊）
        self.warmed: set[str] = set()
        self.visited: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._context is not None

    async def start(self) -> None:
        if self._context is not None:
            return
        from playwright.async_api import async_playwright

        s = self.settings
        self._pw = await async_playwright().start()
        if s.cdp_url:
            # 接管使用者已登入的真實 Chrome：天然攜帶 Cookie 與真實指紋
            self._browser = await self._pw.chromium.connect_over_cdp(s.cdp_url)
            self._owns_browser = False
            self._context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
        else:
            args = ["--disable-blink-features=AutomationControlled"] if s.stealth else []
            self._browser = await self._pw.chromium.launch(
                headless=s.headless, args=args, executable_path=s.executable_path or None
            )
            opts: dict[str, Any] = {
                "viewport": {"width": s.viewport_width, "height": s.viewport_height},
                "locale": s.locale,
                "timezone_id": s.timezone,
            }
            if s.user_agent:
                opts["user_agent"] = s.user_agent
            self._context = await self._browser.new_context(**opts)
        if s.stealth:
            await self._context.add_init_script(_STEALTH_JS)
        self._context.set_default_timeout(s.action_timeout_ms)
        self._context.set_default_navigation_timeout(s.navigation_timeout_ms)
        self._context.on("page", self._on_new_page)
        pages = self._context.pages
        self._page = pages[-1] if pages else await self._context.new_page()

    def _on_new_page(self, page: "Page") -> None:
        # 點選開啟的新標籤頁自動成為當前頁（與人的直覺一致）
        self._page = page

    async def page(self) -> "Page":
        await self.start()
        if self._page is None or self._page.is_closed():
            pages = [p for p in self._context.pages if not p.is_closed()]
            self._page = pages[-1] if pages else await self._context.new_page()
        return self._page

    @property
    def pages(self) -> list["Page"]:
        return [p for p in self._context.pages if not p.is_closed()] if self._context else []

    async def switch(self, index: int) -> "Page":
        pages = self.pages
        if not 0 <= index < len(pages):
            raise IndexError(f"標籤頁序號超出範圍（共 {len(pages)} 個）")
        self._page = pages[index]
        await self._page.bring_to_front()
        return self._page

    async def new_page(self) -> "Page":
        await self.start()
        self._page = await self._context.new_page()
        return self._page

    @asynccontextmanager
    async def side_page(self):
        """臨時標籤頁（如搜尋）：用完即關，並恢復原來的當前頁。"""
        await self.start()
        current = self._page
        page = await self._context.new_page()
        try:
            yield page
        finally:
            await page.close()
            self._page = current if current is not None and not current.is_closed() else None

    # ---------- 導航 ----------
    async def goto(self, url: str) -> dict[str, Any]:
        page = await self.page()
        key = site_key(url)
        warmed_note = None
        warm = self.warmups.get(key)
        if warm and key not in self.warmed and site_key(page.url or "") != key and warm.rstrip("/") != url.rstrip("/"):
            try:
                await page.goto(warm, wait_until="domcontentloaded")
                await self.settle(max_ms=2500)
                warmed_note = warm
            except Exception as exc:  # 預熱失敗不影響正式導航
                log.debug("warm-up failed: %s", exc)
        self.warmed.add(key)
        response = await page.goto(url, wait_until="domcontentloaded")
        await self.settle()
        first_visit = page.url not in self.visited
        self.visited.add(page.url)
        return {
            "status": response.status if response else None,
            "url": page.url,
            "warmed": warmed_note,
            "first_visit": first_visit,
        }

    # ---------- 感知原語 ----------
    async def settle(self, quiet_ms: int = 400, max_ms: int = 3000) -> None:
        page = await self.page()
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=max_ms)
        except Exception:
            pass
        try:
            await page.evaluate(_QUIET_JS, {"quiet": quiet_ms, "max": max_ms})
        except Exception:
            # 導航中執行上下文被銷燬：再等一次載入即可
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=max_ms)
            except Exception:
                pass

    async def snapshot(self, **opts: Any) -> PageSnapshot:
        page = await self.page()
        options = {
            **SNAPSHOT_OPTIONS,
            "digestChars": self.settings.digest_chars * 4,
            "maxElements": 500,
            **opts,
        }
        for attempt in range(3):
            try:
                data = await page.evaluate(SNAPSHOT_JS, options)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                log.debug("snapshot retry: %s", exc)
                await self.settle(max_ms=1500)
        self.last_snapshot = PageSnapshot.from_js(data)
        return self.last_snapshot

    async def probe(self) -> dict[str, Any]:
        page = await self.page()
        try:
            data = await page.evaluate(_PROBE_JS)
        except Exception:
            data = {"url": page.url, "title": "", "mut": -1, "focused": None}
        data["tabs"] = len(self.pages)
        return data

    async def screenshot_for_vision(self) -> tuple[bytes, tuple[int, int]]:
        """視口截圖，scale='css' 保證截影象素 == CSS 畫素，視覺座標可直接用於點選。"""
        page = await self.page()
        png = await page.screenshot(type="png", scale="css", full_page=False)
        vp = page.viewport_size or {"width": self.settings.viewport_width, "height": self.settings.viewport_height}
        return png, (vp["width"], vp["height"])

    async def html(self) -> str:
        return await (await self.page()).content()

    async def close(self) -> None:
        try:
            if self._owns_browser and self._browser:
                await self._browser.close()
        finally:
            if self._pw:
                await self._pw.stop()
            self._pw = self._browser = self._context = self._page = None
