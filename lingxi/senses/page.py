"""頁面快照的資料模型與"簡報"渲染。

模型看到的不是一長串 [index]<tag>text，而是按語義分組、帶狀態、帶風險提示的頁面簡報：

【頁面】攜程機票 · https://… · 下方還有 2300px
【提示】⚠ 有浮層遮擋了 18 個元素 …
【日期格】〔2026年6月〕#45 26 ¥560 · #46 27 …
【輸入框】#12 出發城市=上海 · #13 到達城市=(空)★焦點
【按鈕】#20 搜尋
【正文摘要】…
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from lingxi.hanzi import detect_script, to_trad_many

KIND_TITLES: list[tuple[str, str]] = [
    ("date", "日期格"),
    ("input", "輸入框"),
    ("option", "下拉/聯想選項"),
    ("select", "下拉框"),
    ("check", "勾選項"),
    ("tab", "標籤頁"),
    ("button", "按鈕"),
    ("link", "連結"),
    ("other", "其他可點選"),
]


def compact_text(text: str, short: int = 12) -> str:
    """把連續的短行（選單項、日曆數字、表頭）合併成一行，正文長句保持獨立成行。"""
    out: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= short:
            buf.append(line)
            continue
        if buf:
            out.append(" ".join(buf))
            buf = []
        out.append(line)
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)


@dataclass
class Element:
    id: int
    tag: str
    kind: str
    label: str
    value: str | None = None
    role: str = ""
    type: str = ""
    href: str = ""
    rect: dict[str, int] = field(default_factory=dict)
    inView: bool = True
    covered: bool = False
    states: list[str] = field(default_factory=list)
    date: dict[str, Any] | None = None

    @property
    def center(self) -> tuple[float, float]:
        r = self.rect
        return r.get("x", 0) + r.get("w", 0) / 2, r.get("y", 0) + r.get("h", 0) / 2

    def display(self) -> str:
        if self.kind == "date" and self.date:
            text = f"#{self.id} {self.date['day']}"
            rest = self.label.split(" ", 1)[1] if " " in self.label else ""
            if rest and not rest.isdigit():
                text += f" {rest[:12]}"
        elif self.kind in ("input", "select"):
            shown = self.value if self.value else "(空)"
            text = f"#{self.id} {self.label or self.type or self.tag}={shown}"
        else:
            text = f"#{self.id} {self.label or self.tag}"
        marks = []
        if "focused" in self.states:
            marks.append("★焦點")
        if "checked" in self.states or "selected" in self.states:
            marks.append("✓")
        if "disabled" in self.states or (self.date and self.date.get("disabled")):
            marks.append("✗不可用")
        if self.covered:
            marks.append("被遮擋")
        if not self.inView:
            marks.append("↓視口外")
        return text + ("（" + "，".join(marks) + "）" if marks else "")


@dataclass
class PageSnapshot:
    url: str
    title: str
    elements: list[Element]
    viewport: dict[str, Any] = field(default_factory=dict)
    scroll: dict[str, int] = field(default_factory=dict)
    overlay: dict[str, Any] | None = None
    focused_id: int | None = None
    digest: str = ""
    text_length: int = 0
    iframes: int = 0
    suspect_blocked: bool = False
    challenge: bool = False
    mutations: int = 0
    script: str = "unknown"  # 原網頁的書寫系統：simplified / traditional / unknown

    @classmethod
    def from_js(cls, d: dict[str, Any]) -> "PageSnapshot":
        """把快照腳本的原始結果轉成 PageSnapshot。

        網頁文字（標題、元素標籤與值、正文摘要、浮層提示）在這裡一次批次轉成繁體：
        模型、介面、紀錄看到的都是繁體；操作仍然透過 data-lx-id 編號定位，不受轉換影響。
        原網頁的書寫系統記在 script 欄位，輸入文字時用來決定要不要轉寫成簡體。
        """
        elements = d.get("elements", [])
        overlay = d.get("overlay")
        raw_title, raw_digest = d.get("title", ""), d.get("digest", "")
        texts = [raw_title, raw_digest, (overlay or {}).get("hint", "")]
        for e in elements:
            texts += [e.get("label") or "", e.get("value") or ""]
        conv = to_trad_many(texts)
        title, digest, hint = conv[0], conv[1], conv[2]
        for i, e in enumerate(elements):
            e["label"] = conv[3 + 2 * i]
            if e.get("value"):
                e["value"] = conv[4 + 2 * i]
        if overlay:
            overlay = {**overlay, "hint": hint}
        return cls(
            url=d.get("url", ""),
            title=title,
            elements=[Element(**e) for e in elements],
            viewport=d.get("viewport", {}),
            scroll=d.get("scroll", {}),
            overlay=overlay,
            focused_id=d.get("focusedId"),
            digest=digest,
            text_length=d.get("textLength", 0),
            iframes=d.get("iframes", 0),
            suspect_blocked=d.get("suspectBlocked", False),
            challenge=d.get("challenge", False),
            mutations=d.get("mutations", 0),
            script=detect_script(raw_title + raw_digest[:3000]),
        )

    def find(self, element_id: int) -> Element | None:
        return next((e for e in self.elements if e.id == element_id), None)

    def fingerprint(self) -> str:
        """頁面狀態指紋：URL + 前 80 個元素的(編號, 文字, 值)。用於判斷"動作前後頁面是否變了"。"""
        h = hashlib.sha1(self.url.encode())
        for e in self.elements[:80]:
            h.update(f"{e.id}|{e.label}|{e.value}|{','.join(e.states)}".encode())
        return h.hexdigest()[:12]

    def hints(self) -> list[str]:
        out = []
        if self.overlay:
            o = self.overlay
            close = "、".join(f"#{i}" for i in o.get("closeIds", [])) or "未識別到"
            out.append(
                f"⚠ 有浮層遮擋了 {o['coveredCount']} 個元素（約佔視口 {o.get('area', '?')}%，內容：{o.get('hint', '')}）。"
                f"被遮擋的元素點不到；若浮層不是你需要的（如登入/廣告彈窗），先關閉它。可能的關閉按鈕：{close}"
            )
        if self.challenge:
            out.append(
                "⚠ 頁面出現人機驗證/頻率限制。不要嘗試破解驗證碼：請換用手冊推薦的入口、先造訪首頁建立會話，"
                "或用 ask_human 請使用者在瀏覽器裡手動完成驗證。"
            )
        if self.suspect_blocked:
            out.append("⚠ 頁面幾乎為空，疑似被反爬攔截。建議先開啟站點首頁建立會話再跳轉，或放慢操作節奏。")
        if self.iframes:
            out.append(f"頁面含 {self.iframes} 個 iframe，其中的內容不在下方元素列表裡。")
        return out

    def render(self, max_per_kind: int = 40, digest_chars: int = 1800) -> str:
        below = self.scroll.get("below", 0)
        lines = [f"【頁面】{self.title or '(無標題)'} · {self.url}" + (f" · 下方還有 {below}px" if below > 50 else "")]
        if self.script == "simplified":
            lines.append("【說明】原網頁為簡體中文，以下文字已轉為繁體顯示；web_type 輸入時會自動轉寫成簡體，照常用繁體描述即可。")
        hints = self.hints()
        if hints:
            lines.append("【提示】" + "\n      ".join(hints))

        by_kind: dict[str, list[Element]] = {}
        for e in self.elements:
            by_kind.setdefault(e.kind, []).append(e)

        for kind, title in KIND_TITLES:
            items = by_kind.get(kind, [])
            if not items:
                continue
            if kind == "date":
                lines.append(f"【{title}】" + self._render_dates(items))
                continue
            limit = max_per_kind if kind not in ("link", "other") else max(10, max_per_kind // 2)
            shown = " · ".join(e.display() for e in items[:limit])
            more = f" …另有 {len(items) - limit} 個" if len(items) > limit else ""
            lines.append(f"【{title}】{shown}{more}")

        if not self.elements:
            lines.append("【元素】當前頁面沒有識別到可互動元素。")
        if self.digest:
            digest = compact_text(self.digest)[:digest_chars]
            note = f"（正文共 {self.text_length} 字，此處為節選；需要更多請用 web_read）" if self.text_length > len(digest) else ""
            lines.append(f"【正文摘要】{note}\n{digest}")
        return "\n".join(lines)

    @staticmethod
    def _render_dates(items: list[Element]) -> str:
        groups: dict[str, list[str]] = {}
        for e in items:
            d = e.date or {}
            if d.get("iso"):
                key = f"{d['iso'][:4]}年{int(d['iso'][5:7])}月"
            elif d.get("month"):
                key = (f"{d['year']}年" if d.get("year") else "") + f"{d['month']}月"
            else:
                key = "月份未知"
            groups.setdefault(key, []).append(e.display().replace(f"#{e.id} ", f"#{e.id}:", 1))
        return " ".join(f"〔{k}〕" + " ".join(v) for k, v in groups.items())
