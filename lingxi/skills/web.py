"""瀏覽器技能族：每個動作都走「定位 → 執行 → 驗證」三段式。

OpenManus 的 browser_use 工具執行完就返回"Clicked element at index 36"，
不管頁面有沒有真的變化；課程裡"輸入上海，出發城市卻還是新加坡"就是這麼漏掉的。

靈犀的每個動作都會：
  定位：grounding.resolve()，記錄定位依據（編號 / 語義 / 日期 / 視覺+DOM 共識）
  執行：Playwright 定位器 → JS 點選 → 座標點選，逐級嘗試
  驗證：對比動作前後的探針（URL、DOM 變更數、焦點、標籤頁），輸入類動作回讀輸入框的值；
        沒有觀察到變化就明確標註"未驗證"，讓模型知道要複查。
"""

from __future__ import annotations

import base64
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field

from lingxi.hanzi import to_trad, to_trad_many, unify, variants
from lingxi.llm.messages import Message
from lingxi.senses.grounding import Ambiguous, NotFound, Resolution, normalize, resolve
from lingxi.skills.base import Outcome, Skill
from lingxi.utils import clip, extract_json

if TYPE_CHECKING:
    from lingxi.kernel.context import RunContext

TARGET_DOC = "目標元素：優先用頁面簡報裡的編號（如 #45），也可以用文字描述（如 '出發城市'、'搜尋'、'6月26日'）"

_KEY_ALIASES = {
    "esc": "Escape", "escape": "Escape", "enter": "Enter", "return": "Enter", "tab": "Tab",
    "space": "Space", "backspace": "Backspace", "delete": "Delete", "del": "Delete",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "pageup": "PageUp", "pagedown": "PageDown", "home": "Home", "end": "End",
    "ctrl": "Control", "control": "Control", "cmd": "Meta", "command": "Meta", "alt": "Alt", "shift": "Shift",
}


def normalize_keys(keys: str) -> str:
    parts = [p.strip() for p in re.split(r"\s*\+\s*", keys.strip()) if p.strip()]
    out = []
    for p in parts:
        alias = _KEY_ALIASES.get(p.lower())
        out.append(alias or (p.upper() if len(p) == 1 else p[0].upper() + p[1:]))
    return "+".join(out)


def fix_stale_dates(url: str, params: set[str], today: date) -> tuple[str, list[str]]:
    """URL 查詢參數裡過期的日期順延到未來（同月同日），返回新 URL 與修改說明。"""
    parsed = urlparse(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    lowered = {p.lower() for p in params}
    changes, new_query = [], []
    for key, value in query:
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if key.lower() in lowered and m:
            y, mo, d = map(int, m.groups())
            try:
                when = date(y, mo, d)
                while when < today:
                    y += 1
                    when = date(y, mo, d)
            except ValueError:
                when = None
            if when and when.isoformat() != value:
                changes.append(f"{key}: {value} → {when.isoformat()}")
                value = when.isoformat()
        new_query.append((key, value))
    if not changes:
        return url, []
    return urlunparse(parsed._replace(query=urlencode(new_query))), changes


def _candidates(cands) -> str:
    return " · ".join(f"#{e.id} {clip(e.label, 24)}" + ("（被遮擋）" if e.covered else "") for e in cands)


def _text_locator(page, text: str):
    """在網頁上找一段文字：原文、逐字簡體、大陸用語三種寫法任一命中即可。"""
    forms = variants(text)
    loc = page.get_by_text(forms[0], exact=False)
    for form in forms[1:]:
        loc = loc.or_(page.get_by_text(form, exact=False))
    return loc.first


async def _locate(ctx: "RunContext", target: str, intent) -> Resolution | Outcome:
    snap = await ctx.browser.snapshot()
    try:
        return await resolve(ctx, snap, target, intent)
    except Ambiguous as amb:
        return Outcome.fail(f"「{target}」匹配到多個元素，請用編號指定", "候選：" + _candidates(amb.candidates))
    except NotFound as nf:
        return Outcome.fail(str(nf), "請對照最新的頁面簡報換一種描述，或先滾動 / 關閉遮擋的浮層。")


async def _effect(ctx: "RunContext", before: dict[str, Any]) -> tuple[bool, str, list[str]]:
    await ctx.browser.settle()
    after = await ctx.browser.probe()
    parts, progress = [], []
    if after["tabs"] > before["tabs"]:
        parts.append("開啟了新標籤頁")
    if after["url"] != before["url"]:
        parts.append(f"頁面跳轉到 {clip(after['url'], 120)}")
        if after["url"] not in ctx.browser.visited:
            progress.append("new_url")
            ctx.browser.visited.add(after["url"])
    elif after["mut"] > before["mut"] >= 0:
        parts.append(f"頁面發生變化（{after['mut'] - before['mut']} 處 DOM 變更）")
    if after.get("focused") and after.get("focused") != before.get("focused"):
        parts.append(f"焦點移到 #{after['focused']}")
    return bool(parts), "，".join(parts), progress


async def _evidence(ctx: "RunContext", tag: str, force: bool = False) -> list[str]:
    """失敗/未驗證時必留截圖；開啟 debug_snapshots 時每步都留截圖 + HTML。"""
    if not (force or ctx.settings.agent.debug_snapshots) or not ctx.browser_started:
        return []
    out = []
    try:
        page = await ctx.browser.page()
        shot = await page.screenshot(type="jpeg", quality=70)
        if (p := ctx.journal.save_artifact(f"{tag}.jpg", shot)):
            out.append(p)
        if ctx.settings.agent.debug_snapshots and (p := ctx.journal.save_artifact(f"{tag}.html", await page.content())):
            out.append(p)
    except Exception:
        pass
    return out


async def _press(ctx: "RunContext", res: Resolution) -> str:
    page = await ctx.browser.page()
    timeout = min(ctx.settings.browser.action_timeout_ms, 4000)
    if res.element and res.method != "vision":
        loc = page.locator(f'[data-lx-id="{res.element.id}"]').first
        try:
            await loc.scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass
        try:
            await loc.click(timeout=timeout)
            return "定位器點選"
        except Exception as first_error:
            try:
                await loc.evaluate("el => el.click()")
                return "JS 點選"
            except Exception:
                box = await loc.bounding_box()
                if box:
                    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    return "座標點選"
                raise first_error
    if res.point:
        await page.mouse.click(*res.point)
        return "視覺座標點選"
    raise NotFound("沒有可執行的點選目標")


def _outcome_from_effect(action: str, res: Resolution, how: str, changed: bool, desc: str,
                         progress: list[str], artifacts: list[str]) -> Outcome:
    label = res.element.label if res.element and res.element.label else "目標"
    who = f"#{res.element.id}「{clip(label, 30)}」" if res.element else f"座標 {res.point}"
    summary = f"{action}{who}（{how}）" + (f"：{desc}" if changed else "，但頁面沒有可觀察的變化")
    detail = res.note
    if not changed:
        detail = (detail + "\n" if detail else "") + "可能的原因：元素被浮層遮擋、需要先展開父級選單、或點選的是非互動區域。"
    return Outcome(ok=True, summary=summary, detail=detail, verified=changed, grounding=res.evidence(),
                   progress=progress + (["verified_action"] if changed else []), artifacts=artifacts)


# ======================= 技能定義 =======================

class OpenParams(BaseModel):
    url: str = Field(description="要開啟的網址")
    new_tab: bool = Field(False, description="是否在新標籤頁開啟")


class WebOpen(Skill):
    name = "web_open"
    description = "在瀏覽器中開啟網址。首次造訪手冊中登記過的站點時，會先自動造訪其首頁建立會話，再跳轉到目標地址。"
    Params = OpenParams

    async def run(self, ctx, p: OpenParams) -> Outcome:
        url = p.url.strip()
        if not re.match(r"^[a-zA-Z][\w+.-]*://", url):
            url = "https://" + url
        notes = []
        if ctx.profile.date_preference == "future":
            url, changes = fix_stale_dates(url, ctx.date_params, date.today())
            if changes:
                notes.append("已把過期日期順延：" + "；".join(changes))
        if p.new_tab:
            await ctx.browser.new_page()
        try:
            info = await ctx.browser.goto(url)
        except Exception as exc:
            return Outcome.fail(f"開啟失敗：{clip(str(exc), 200)}", artifacts=await _evidence(ctx, "open_fail", True))
        if info.get("warmed"):
            notes.append(f"已先造訪 {info['warmed']} 建立會話")
        snap = await ctx.browser.snapshot()
        blocked = snap.suspect_blocked or snap.challenge
        status = info.get("status")
        summary = f"已開啟「{clip(snap.title, 40)}」{snap.url}" + (f"（HTTP {status}）" if status and status >= 400 else "")
        return Outcome(
            ok=status is None or status < 400,
            summary=summary,
            detail="\n".join(notes + snap.hints()),
            verified=not blocked,
            progress=["new_url"] if info.get("first_visit") and not blocked else [],
            artifacts=await _evidence(ctx, "open", force=blocked),
            data={"url": snap.url},
        )


class ClickParams(BaseModel):
    target: str = Field(description=TARGET_DOC)


class WebClick(Skill):
    name = "web_click"
    description = "點選頁面元素（按鈕、連結、日期格、選項等），並報告點選後頁面是否真的發生了變化。"
    Params = ClickParams

    async def run(self, ctx, p: ClickParams) -> Outcome:
        res = await _locate(ctx, p.target, "click")
        if isinstance(res, Outcome):
            res.artifacts = await _evidence(ctx, "click_miss", True)
            return res
        before = await ctx.browser.probe()
        try:
            how = await _press(ctx, res)
        except Exception as exc:
            return Outcome.fail(f"點選執行失敗：{clip(str(exc), 200)}", grounding=res.evidence(),
                                artifacts=await _evidence(ctx, "click_fail", True))
        changed, desc, progress = await _effect(ctx, before)
        artifacts = await _evidence(ctx, "click", force=not changed)
        return _outcome_from_effect("點選", res, how, changed, desc, progress, artifacts)


class TypeParams(BaseModel):
    target: str = Field(description=TARGET_DOC)
    text: str = Field(description="要輸入的文字")
    submit: bool = Field(False, description="輸入後是否按回車提交")
    clear: bool = Field(True, description="輸入前是否清空原有內容")


_READBACK_JS = "el => (el.value !== undefined && el.value !== null ? el.value : el.innerText) || ''"


class WebType(Skill):
    name = "web_type"
    description = ("在輸入框中輸入文字。輸入後會回讀輸入框的實際值來驗證；"
                   "如果出現聯想下拉（如城市候選），會列出候選編號，需要時請再用 web_click 選擇。")
    Params = TypeParams

    async def run(self, ctx, p: TypeParams) -> Outcome:
        res = await _locate(ctx, p.target, "type")
        if isinstance(res, Outcome):
            res.artifacts = await _evidence(ctx, "type_miss", True)
            return res
        page = await ctx.browser.page()
        # 網頁是簡體時，先把（繁體的）輸入轉寫成簡體與大陸用語，網站的聯想搜尋才對得上
        snap_before = ctx.browser.last_snapshot
        transliterated = snap_before is not None and snap_before.script == "simplified" and unify(p.text) != p.text
        typed = unify(p.text) if transliterated else p.text
        try:
            if res.element and res.method != "vision":
                loc = page.locator(f'[data-lx-id="{res.element.id}"]').first
                try:
                    await loc.click(timeout=3000)
                except Exception:
                    await loc.focus()
                editable = await loc.evaluate("el => el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'")
                if editable:
                    if p.clear:
                        await loc.fill("")
                    await loc.press_sequentially(typed, delay=35)
                else:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.type(typed, delay=35)
                reader = loc
            else:
                await page.mouse.click(*res.point)
                await page.keyboard.press("Control+A")
                await page.keyboard.type(typed, delay=35)
                reader = page.locator(":focus").first
        except Exception as exc:
            return Outcome.fail(f"輸入失敗：{clip(str(exc), 200)}", grounding=res.evidence(),
                                artifacts=await _evidence(ctx, "type_fail", True))

        await ctx.browser.settle(quiet_ms=300, max_ms=1500)
        try:
            actual = await reader.evaluate(_READBACK_JS)
        except Exception:
            actual = ""
        verified = normalize(p.text) in normalize(actual)
        lines = [f"輸入框當前的值：「{clip(to_trad(actual), 60)}」" + ("" if verified else "（與期望不一致！）")]
        if transliterated:
            lines.append("（網頁為簡體，已自動轉寫成簡體輸入）")

        snap = await ctx.browser.snapshot()
        options = [e for e in snap.elements if e.kind == "option" and e.inView and not e.covered]
        if options:
            want = normalize(p.text)
            options.sort(key=lambda e: 0 if want and want in normalize(e.label) else 1)
            lines.append("出現聯想候選：" + _candidates(options[:8]) + "。如果需要從中選擇才能生效，請用 web_click 點選對應編號。")

        progress: list[str] = ["verified_action"] if verified else []
        summary = f"在 #{res.element.id if res.element else '?'}「{clip(res.element.label if res.element else '', 24)}」輸入「{clip(p.text, 30)}」"
        if p.submit:
            before = await ctx.browser.probe()
            await page.keyboard.press("Enter")
            changed, desc, more = await _effect(ctx, before)
            progress += more
            lines.append("已按回車：" + (desc if changed else "頁面沒有變化"))
        return Outcome(ok=True, summary=summary + ("" if verified else "，但回讀值不一致"), detail="\n".join(lines),
                       verified=verified, grounding=res.evidence(), progress=progress,
                       artifacts=await _evidence(ctx, "type", force=not verified))


class SelectParams(BaseModel):
    target: str = Field(description="下拉框：" + TARGET_DOC)
    option: str = Field(description="要選擇的選項文字")


class WebSelect(Skill):
    name = "web_select"
    description = "在下拉框中選擇選項。原生 <select> 直接選擇；自定義下拉會先展開再點選選項。"
    Params = SelectParams

    async def run(self, ctx, p: SelectParams) -> Outcome:
        res = await _locate(ctx, p.target, "select")
        if isinstance(res, Outcome):
            return res
        page = await ctx.browser.page()
        if res.element and res.method != "vision":
            loc = page.locator(f'[data-lx-id="{res.element.id}"]').first
            if await loc.evaluate("el => el.tagName") == "SELECT":
                picked_error = None
                for option in variants(p.option):
                    try:
                        await loc.select_option(label=option)
                        break
                    except Exception as exc:
                        picked_error = exc
                else:
                    try:
                        await loc.select_option(value=p.option)
                    except Exception:
                        return Outcome.fail(f"選擇失敗：{clip(str(picked_error), 160)}", grounding=res.evidence())
                chosen = to_trad(await loc.evaluate("el => el.selectedOptions[0] ? el.selectedOptions[0].text : ''"))
                ok = normalize(p.option) in normalize(chosen)
                return Outcome(ok=True, summary=f"下拉框當前選中「{chosen}」", verified=ok,
                               grounding=res.evidence(), progress=["verified_action"] if ok else [])
        await _press(ctx, res)
        await ctx.browser.settle(max_ms=1500)
        opt = await _locate(ctx, p.option, "click")
        if isinstance(opt, Outcome):
            return Outcome.fail(f"已展開下拉，但沒找到選項「{p.option}」", opt.detail)
        before = await ctx.browser.probe()
        how = await _press(ctx, opt)
        changed, desc, progress = await _effect(ctx, before)
        return _outcome_from_effect("選擇", opt, how, changed, desc, progress, [])


class KeyParams(BaseModel):
    keys: str = Field(description="按鍵，如 Enter、Escape、Tab、ArrowDown、Control+A")


class WebKey(Skill):
    name = "web_key"
    description = "向當前頁面傳送鍵盤按鍵（關閉彈窗常用 Escape，確認聯想候選常用 ArrowDown + Enter）。"
    Params = KeyParams

    async def run(self, ctx, p: KeyParams) -> Outcome:
        page = await ctx.browser.page()
        keys = normalize_keys(p.keys)
        before = await ctx.browser.probe()
        try:
            await page.keyboard.press(keys)
        except Exception as exc:
            return Outcome.fail(f"按鍵 {keys} 失敗：{clip(str(exc), 160)}")
        changed, desc, progress = await _effect(ctx, before)
        return Outcome(ok=True, summary=f"按下 {keys}" + (f"：{desc}" if changed else "，頁面沒有變化"),
                       verified=changed, progress=progress)


class ScrollParams(BaseModel):
    direction: Literal["down", "up", "top", "bottom", "to_text"] = Field("down", description="滾動方向；to_text 表示滾動到某段文字")
    pixels: int | None = Field(None, description="滾動畫素，預設約 0.8 屏")
    text: str | None = Field(None, description="direction=to_text 時要找的文字")


class WebScroll(Skill):
    name = "web_scroll"
    description = "滾動頁面，或滾動到包含指定文字的位置。"
    Params = ScrollParams

    async def run(self, ctx, p: ScrollParams) -> Outcome:
        page = await ctx.browser.page()
        if p.direction == "to_text":
            if not p.text:
                return Outcome.fail("direction=to_text 時必須提供 text")
            try:
                await _text_locator(page, p.text).scroll_into_view_if_needed(timeout=4000)
            except Exception:
                return Outcome.fail(f"頁面上沒有找到文字「{p.text}」")
        else:
            h = (page.viewport_size or {"height": 800})["height"]
            amount = p.pixels or int(h * 0.8)
            js = {"down": f"window.scrollBy(0,{amount})", "up": f"window.scrollBy(0,-{amount})",
                  "top": "window.scrollTo(0,0)", "bottom": "window.scrollTo(0,document.documentElement.scrollHeight)"}
            await page.evaluate(js[p.direction])
        await ctx.browser.settle(quiet_ms=300, max_ms=1500)
        pos = await page.evaluate("() => [Math.round(scrollY), document.documentElement.scrollHeight, innerHeight]")
        below = max(0, pos[1] - pos[0] - pos[2])
        return Outcome(ok=True, summary=f"已滾動到 y={pos[0]}（頁面高 {pos[1]}，下方還有 {below}px）")


_TEXT_JS = """() => {
  const root = document.querySelector("main,[role=main],article") || document.body;
  return (root.innerText || "").split("\\n").map(s => s.replace(/\\s+/g, " ").trim()).filter(Boolean).join("\\n");
}"""


class ReadParams(BaseModel):
    goal: str = Field(description="閱讀目標：想從當前頁面獲取什麼資訊")
    offset: int = Field(0, description="從正文第幾個字開始讀（逐段閱讀時用）")
    full: bool = Field(False, description="長網頁預設只讀與目標最相關的段落；設為 true 則從 offset 起逐段閱讀全文")


class WebRead(Skill):
    name = "web_read"
    description = ("精讀當前頁面正文，圍繞目標提取事實，並與【發現板】上已有的事實交叉比對："
                   "被新來源印證的標為多方印證，說法不一致的標為有矛盾。")
    Params = ReadParams
    chunk_chars = 12000

    async def run(self, ctx, p: ReadParams) -> Outcome:
        page = await ctx.browser.page()
        text = await page.evaluate(_TEXT_JS)
        rs = ctx.settings.retrieval
        focus = None
        if not p.full and p.offset == 0 and len(text) > rs.read_focus_chars:
            # 長網頁：用檢索挑出與目標最相關的段落，而不是把前 12,000 字整塊送給模型
            from lingxi.retrieval.focus import focus_text

            chunk, focus = focus_text(text, p.goal, rs)
            chunk = to_trad(chunk)
            ctx.emit("focus", **focus)
        else:
            chunk = to_trad(text[p.offset: p.offset + self.chunk_chars])
        if not chunk.strip():
            return Outcome.fail("當前頁面沒有可讀的正文（可能還在載入或被攔截）")
        known = ctx.findings.evidence_text(3000)
        where = (f"正文（共 {len(text)} 字，已按目標挑出最相關的 {focus['kept']}/{focus['chunks']} 段）" if focus
                 else f"正文（第 {p.offset}–{p.offset + len(chunk)} 字，共 {len(text)} 字）")
        prompt = (
            f"目標：{p.goal}\n\n請閱讀下面的網頁正文，只輸出 JSON：\n"
            '{"found": true 或 false, "facts": ["與目標相關的簡潔事實，保留數字、名稱、時間、價格"], '
            '"answer": "圍繞目標的簡短回答", "corroborates": ["本頁也支持的已有發現編號，如 F2"], '
            '"contradicts": [{"id": "F3", "note": "本頁的說法與它哪裡不同"}]}\n'
            "不要編造正文裡沒有的資訊；已有發現被本頁支持時只填 corroborates，不要在 facts 裡重複；一律使用繁體中文。\n\n"
            + (f"【已有發現】\n{known}\n\n" if known else "")
            + f"網頁：{page.url}\n{where}：\n{chunk}"
        )
        reply = await ctx.llm.chat([Message.system("你是嚴謹的網頁資訊提取與查證員。"), Message.user(prompt)])
        data = extract_json(reply.content)
        data = data if isinstance(data, dict) else {"answer": reply.content}
        facts = to_trad_many([str(f) for f in data.get("facts", []) if f])
        answer = to_trad(str(data.get("answer", "")))
        source = clip(page.url, 80)
        added = ctx.findings.add(facts, source=source, step=ctx.step)
        backed = [str(r) for r in data.get("corroborates", []) or [] if ctx.findings.corroborate(r, source)]
        clashes = []
        for item in data.get("contradicts", []) or []:
            if isinstance(item, dict) and ctx.findings.contradict(item.get("id", ""), source, str(item.get("note", ""))):
                clashes.append(f"{item.get('id')}：{item.get('note', '')}")

        lines = [f"回答：{answer}"] + [f"· {f}" for f in facts]
        if backed:
            lines.append("本頁印證了：" + "、".join(backed))
        if clashes:
            lines.append("⚠ 本頁與已有發現矛盾：" + "；".join(clashes) + "（請再找一個來源確認哪個正確）")
        if focus:
            lines.append(f"（正文共 {len(text)} 字，只讀了與目標最相關的 {focus['kept']}/{focus['chunks']} 段；"
                         "若沒找到需要的資訊，可用 full=true 從頭逐段閱讀）")
        else:
            rest = len(text) - (p.offset + len(chunk))
            if rest > 0:
                lines.append(f"（正文還剩 {rest} 字未讀，如需繼續請用 full=true、offset={p.offset + len(chunk)}）")
        progress = ["facts"] if (added or backed) else []
        return Outcome(ok=True, summary=f"提取 {len(facts)} 條（新增 {added}），印證 {len(backed)} 條，矛盾 {len(clashes)} 條",
                       detail="\n".join(lines), progress=progress,
                       data={"facts": facts, "corroborates": backed, "contradicts": clashes, "focus": focus})


class NavParams(BaseModel):
    action: Literal["back", "forward", "reload", "tabs", "switch_tab", "close_tab"] = Field(description="導航動作")
    tab: int | None = Field(None, description="switch_tab 時的標籤頁序號（從 0 開始，見 tabs 的輸出）")


class WebNav(Skill):
    name = "web_nav"
    description = "瀏覽器導航：後退、前進、重新整理、列出標籤頁、切換或關閉標籤頁。"
    Params = NavParams

    async def run(self, ctx, p: NavParams) -> Outcome:
        page = await ctx.browser.page()
        if p.action == "tabs":
            lines = [f"[{i}] {clip(await pg.title(), 40)} {pg.url}" for i, pg in enumerate(ctx.browser.pages)]
            return Outcome(ok=True, summary=f"共 {len(lines)} 個標籤頁", detail="\n".join(lines))
        if p.action == "switch_tab":
            if p.tab is None:
                return Outcome.fail("switch_tab 需要 tab 序號")
            pg = await ctx.browser.switch(p.tab)
            return Outcome(ok=True, summary=f"已切換到標籤頁 [{p.tab}] {pg.url}")
        if p.action == "close_tab":
            await page.close()
            current = await ctx.browser.page()
            return Outcome(ok=True, summary=f"已關閉當前標籤頁，現在位於 {current.url}")
        before = await ctx.browser.probe()
        await {"back": page.go_back, "forward": page.go_forward, "reload": page.reload}[p.action]()
        changed, desc, progress = await _effect(ctx, before)
        return Outcome(ok=True, summary=f"{p.action}：{desc or '頁面沒有變化'}", verified=changed, progress=progress)


class WaitParams(BaseModel):
    seconds: float = Field(2.0, description="最多等待的秒數（1~15）")
    text: str | None = Field(None, description="等待這段文字出現在頁面上")


class WebWait(Skill):
    name = "web_wait"
    description = "等待頁面載入，或等待某段文字出現（適合非同步載入的結果列表）。"
    Params = WaitParams

    async def run(self, ctx, p: WaitParams) -> Outcome:
        page = await ctx.browser.page()
        seconds = min(max(p.seconds, 0.5), 15)
        if p.text:
            try:
                await _text_locator(page, p.text).wait_for(timeout=seconds * 1000)
                return Outcome(ok=True, summary=f"文字「{p.text}」已出現", verified=True)
            except Exception:
                return Outcome(ok=False, summary=f"等待 {seconds}s 後仍未出現「{p.text}」", verified=False)
        await page.wait_for_timeout(seconds * 1000)
        await ctx.browser.settle(max_ms=1500)
        return Outcome(ok=True, summary=f"已等待 {seconds}s")


class LookParams(BaseModel):
    question: str = Field(description="想透過截圖確認的問題，如'日曆裡現在選中的是哪天？'")


class WebLook(Skill):
    name = "web_look"
    description = "看一眼當前頁面截圖並回答問題。當文字簡報不足以判斷頁面狀態（圖片、圖表、複雜控制元件）時使用。"
    Params = LookParams

    async def run(self, ctx, p: LookParams) -> Outcome:
        if ctx.vision is None:
            return Outcome.fail("未配置視覺模型（[llm.vision]），無法看圖")
        png, _ = await ctx.browser.screenshot_for_vision()
        artifact = ctx.journal.save_artifact("look.png", png)
        answer = to_trad(await ctx.vision.ask(png, p.question))
        if ctx.settings.llm.supports_vision:
            ctx.pending_image = base64.b64encode(png).decode()
        return Outcome(ok=True, summary=clip(answer, 200), detail=answer, artifacts=[artifact] if artifact else [])


def web_skills() -> list[Skill]:
    return [WebOpen(), WebClick(), WebType(), WebSelect(), WebKey(), WebScroll(), WebRead(), WebNav(), WebWait(), WebLook()]
