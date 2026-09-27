"""元素定位（Grounding）：把模型給出的 target 解析成頁面上的一個確定元素。

target 可以是：
  "#45" / "45"      —— 快照編號（最穩）
  "出發城市"         —— 自然語言描述
  "6月26日" / "26號" —— 日期（會和日期格的"日 + 月份上下文"對齊）

解析是一條"證據鏈"，每一步都把依據記錄下來：
  1. 編號直達（id）
  2. 語義打分（text / date）：標籤、值、元素語義類別、遮擋、可用性一起打分
     —— 分數接近的多個候選不會被瞎猜，而是返回給模型讓它用編號指定
  3. 視覺 + DOM 共識（vision+dom）：視覺模型給出座標 → elementFromPoint 反查 DOM 元素
     → 校驗該元素文字與目標是否一致。一致才按 DOM 元素操作；不一致則降級為座標點選並明確告知。

最後一步是靈犀相對"視覺兜底"方案的關鍵區別：視覺結果必須被 DOM 交叉驗證。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lingxi.hanzi import to_trad, unify
from lingxi.knowledge.temporal import month_day
from lingxi.senses.page import Element, PageSnapshot

if TYPE_CHECKING:
    from lingxi.kernel.context import RunContext

Intent = Literal["click", "type", "select"]

# 描述裡的泛稱（"搜尋按鈕"→"搜尋"）；兩岸用語並列，統一 unify 後再比對
_GENERIC = sorted({unify(g) for g in [
    "按鈕", "輸入框", "文字框", "文本框", "搜尋框", "搜索框", "下拉框", "下拉選單", "連結", "鏈接", "選項",
    "標籤", "圖示", "圖標", "框", "區域", "button", "input", "link", "the ",
]}, key=len, reverse=True)

_POINT_JS = """
([x, y]) => {
  window.__lx = window.__lx || { seq: 100000, mut: 0 };
  const hit = document.elementFromPoint(x, y);
  if (!hit) return null;
  let n = hit;
  for (let i = 0; i < 6 && n && n.getAttribute; i++, n = n.parentElement) {
    if (n.hasAttribute("data-lx-id")) break;
  }
  const el = (n && n.getAttribute && n.hasAttribute("data-lx-id")) ? n : hit;
  if (!el.hasAttribute("data-lx-id")) el.setAttribute("data-lx-id", String(++window.__lx.seq));
  const text = (el.innerText || el.value || el.getAttribute("aria-label") || el.getAttribute("placeholder") || "")
    .replace(/\\s+/g, " ").trim().slice(0, 80);
  return { id: Number(el.getAttribute("data-lx-id")), label: text, tag: el.tagName.toLowerCase() };
}
"""


@dataclass
class Resolution:
    method: str
    element: Element | None = None
    point: tuple[float, float] | None = None
    score: float = 0.0
    candidates: list[Element] = field(default_factory=list)
    note: str = ""

    def evidence(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "id": self.element.id if self.element else None,
            "label": self.element.label if self.element else None,
            "point": [round(v) for v in self.point] if self.point else None,
            "score": round(self.score, 1),
            "note": self.note,
        }


class Ambiguous(Exception):
    def __init__(self, candidates: list[Element]):
        super().__init__("ambiguous")
        self.candidates = candidates


class NotFound(Exception):
    pass


def normalize(text: str) -> str:
    """比較用正規化：繁簡與兩岸用語統一（「出發」的簡體寫法、「搜尋」與「搜索」視為相同），去掉空白與標點，轉小寫。"""
    return re.sub(r"[\s\-_:：,，。.·|/\\()（）\[\]【】\"'“”‘’!?！？]+", "", unify((text or "").lower()))


def _strip_generic(target: str) -> str:
    t = unify(target.lower())
    for g in _GENERIC:
        t = t.replace(g, "")
    return t.strip() or unify(target)


def _bigram_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)} or {s}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb)


def score(e: Element, target: str, intent: Intent, md: tuple[int | None, int | None]) -> float:
    month, day = md
    s = 0.0
    if day and e.kind == "date" and e.date:
        if e.date.get("day") != day:
            return 0.0
        s = 90.0
        em = e.date.get("month") or (int(e.date["iso"][5:7]) if e.date.get("iso") else None)
        if month and em:
            s = 100.0 if em == month else 5.0
        elif month and not em:
            s = 70.0
    else:
        best = 0.0
        for t in {normalize(target), normalize(_strip_generic(target))}:
            if not t:
                continue
            for field_text, weight in ((normalize(e.label), 1.0), (normalize(e.value or ""), 0.85)):
                if not field_text:
                    continue
                if field_text == t:
                    v = 100.0
                elif t in field_text:
                    v = 60 + 30 * len(t) / len(field_text)
                elif field_text in t and len(field_text) >= 2:
                    v = 50 + 30 * len(field_text) / len(t)
                else:
                    sim = _bigram_sim(field_text, t)
                    v = sim * 60 if sim >= 0.5 else 0.0
                best = max(best, v * weight)
        s = best
    if s <= 0:
        return 0.0
    if intent == "type":
        s += 25 if e.kind == "input" else (5 if e.kind == "select" else -50)
    elif intent == "select":
        s += 25 if e.kind in ("select", "input") else 0
    elif e.kind == "option":
        s += 5
    if e.covered:
        s -= 25
    if not e.inView:
        s -= 5
    if "disabled" in e.states or (e.date and e.date.get("disabled")):
        s -= 40
    return s


def rank(snap: PageSnapshot, target: str, intent: Intent) -> list[tuple[float, Element]]:
    md = month_day(target)
    scored = [(score(e, target, intent, md), e) for e in snap.elements]
    return sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])


def parse_id(target: str, allow_bare: bool = False) -> int | None:
    """'#45' / '編號45' / 'id:45' 是顯式編號；純數字'26'預設先當日期/文字理解。"""
    m = re.fullmatch(r"\s*(?:#|編號|id\s*[:：]?)\s*(\d{1,6})\s*", target, re.I)
    if not m and allow_bare:
        m = re.fullmatch(r"\s*(\d{1,6})\s*", target)
    return int(m.group(1)) if m else None


def match_snapshot(snap: PageSnapshot, target: str, intent: Intent) -> Resolution | None:
    """純函式：只用快照做定位（可單測）。返回 None 表示需要進一步手段；歧義時拋 Ambiguous。"""
    ranked = rank(snap, target, intent)
    if not ranked:
        return None
    best_score, best = ranked[0]
    if best_score < 55:
        return None
    rivals = [e for s, e in ranked[1:4] if s >= best_score - 10]
    if rivals:
        # 同名元素（如頁尾裡還有一個"搜尋"）：只有當其他同名元素都在視口外或被遮擋時，才取可見的那個。
        # 日期格永遠不走這條捷徑——兩個月裡的"26"文字可能完全相同，但含義不同。
        same = [e for e in rivals if normalize(e.label) == normalize(best.label)]
        if (best.kind != "date" and len(same) == len(rivals) and best.inView and not best.covered
                and all(not e.inView or e.covered for e in rivals)):
            return Resolution("text", best, score=best_score, note=f"另有 {len(same)} 個不可見的同名元素")
        raise Ambiguous([best] + rivals)
    method = "date" if best.kind == "date" and month_day(target)[1] else "text"
    return Resolution(method, best, score=best_score)


async def resolve(ctx: "RunContext", snap: PageSnapshot, target: str, intent: Intent) -> Resolution:
    target = (target or "").strip()
    if not target:
        raise NotFound("target 為空")

    async def by_id(element_id: int) -> Resolution:
        el = snap.find(element_id)
        if el:
            return Resolution("id", el, score=100)
        page = await ctx.browser.page()
        if await page.locator(f'[data-lx-id="{element_id}"]').count():
            return Resolution("id", Element(id=element_id, tag="?", kind="other", label=""), score=80,
                              note="元素不在最新快照中（可能在視口外），按編號直達")
        raise NotFound(f"編號 #{element_id} 已失效（頁面可能已重新整理），請參考最新的頁面簡報")

    explicit = parse_id(target)
    if explicit is not None:
        return await by_id(explicit)

    found = match_snapshot(snap, target, intent)
    if found:
        return found

    bare = parse_id(target, allow_bare=True)
    if bare is not None and snap.find(bare):
        return await by_id(bare)

    vision = ctx.vision
    if vision is None:
        raise NotFound(f"頁面上沒有找到與「{target}」匹配的元素")
    return await vision_consensus(ctx, target, intent)


async def vision_consensus(ctx: "RunContext", target: str, intent: Intent) -> Resolution:
    page = await ctx.browser.page()
    png, size = await ctx.browser.screenshot_for_vision()
    verb = "輸入內容的" if intent == "type" else "點選"
    point = await ctx.vision.locate(png, f"請找到需要{verb}的介面元素：{target}", size)
    if point is None:
        raise NotFound(f"DOM 與視覺模型都沒有找到「{target}」")
    hit = await page.evaluate(_POINT_JS, [point.x, point.y])
    if hit:
        hit["label"] = to_trad(hit.get("label") or "")  # 與快照一致：網頁文字一律以繁體呈現
    if not hit:
        return Resolution("vision", point=(point.x, point.y), score=40, note="座標處沒有 DOM 元素")
    md = month_day(target)
    label_norm, target_norm = normalize(hit["label"]), normalize(_strip_generic(target))
    agree = (
        (md[1] is not None and re.match(rf"^{md[1]}(?!\d)", hit["label"] or "") is not None)
        or (target_norm and (target_norm in label_norm or (label_norm and label_norm in target_norm)))
        or _bigram_sim(label_norm, target_norm) >= 0.5
        or (intent == "type" and hit["tag"] in ("input", "textarea"))
    )
    el = Element(id=hit["id"], tag=hit["tag"], kind="other", label=hit["label"])
    if agree:
        return Resolution("vision+dom", el, point=(point.x, point.y), score=75,
                          note=f"視覺座標反查到 DOM 元素「{hit['label'][:30]}」，二者一致")
    return Resolution("vision", el, point=(point.x, point.y), score=45,
                      note=f"視覺座標處的元素文字為「{hit['label'][:30]}」，與目標不完全一致，按座標點選")
