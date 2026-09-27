"""時間錨定（Temporal Anchoring）。

課程裡的真實故障：模型把"6月26日"解析成了 2023-06-26，構造出過期的查詢連結。
OpenManus 衍生版的常見修法是"在系統提示裡寫上今天幾號"，仍然依賴模型自己推理。

靈犀的做法是在任務進入模型之前，就把自然語言日期解析成絕對日期並寫回任務：
    "查詢 6月26日 從上海到北京的機票"
 →  "查詢 6月26日〔=2027-06-26 星期六〕 從上海到北京的機票"
模型只需要照抄，不需要推理年份。

prefer="future"  ：未寫年份且已過去的日期順延到明年（訂票、查詢類任務）
prefer="nearest" ：取離今天最近的那一年（回顧、總結類任務）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from lingxi.hanzi import fold

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
# 規則資料與正規表示式一律先 fold（逐字繁→簡），輸入也先 fold 再比對：繁體、簡體任務都能解析
_CN_DIGITS = {fold(k): v for k, v in
              {"〇": 0, "零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}.items()}
_WEEKDAY_CHAR = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_RELATIVE = {fold(k): v for k, v in
             {"大後天": 3, "後天": 2, "明天": 1, "明日": 1, "今天": 0, "今日": 0, "昨天": -1, "前天": -2}.items()}

Prefer = Literal["future", "nearest"]


def cn_to_int(text: str) -> int | None:
    """中文數字（≤99）轉整數：'二十六'→26，'十'→10，'三十一'→31；阿拉伯數字原樣返回。"""
    text = fold(text)
    if text.isdigit():
        return int(text)
    if not text or any(c not in _CN_DIGITS and c != "十" for c in text):
        return None
    if "十" in text:
        tens, _, ones = text.partition("十")
        t = _CN_DIGITS.get(tens, 1) if tens else 1
        o = _CN_DIGITS.get(ones, 0) if ones else 0
        return t * 10 + o
    return _CN_DIGITS.get(text)


@dataclass
class DateAnchor:
    text: str
    value: date
    start: int
    end: int

    @property
    def iso(self) -> str:
        return self.value.isoformat()

    @property
    def weekday(self) -> str:
        return WEEKDAYS[self.value.weekday()]

    def label(self) -> str:
        return f"{self.iso} {self.weekday}"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(fold(pattern))


_NUM = r"(\d{1,2}|[一二兩三四五六七八九十]{1,3})"
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ymd", _rx(r"(\d{4})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*[日號]?")),
    ("md", _rx(_NUM + r"\s*月\s*" + _NUM + r"(?!\d)\s*[日號]?")),
    ("rel", _rx("|".join(sorted(_RELATIVE, key=len, reverse=True)))),
    ("week", _rx(r"(下下|下個?|本|這個?|上個?)?(?:週|星期|禮拜)([一二三四五六日天])")),
    # "3日遊""7日內"不是日期
    ("day", _rx(r"(?<![\d月年/\-.])(\d{1,2})\s*[號日](?![\d遊內以前後間])")),
]


class TemporalAnchor:
    def __init__(self, now: datetime | date | None = None, prefer: Prefer = "future"):
        now = now or datetime.now()
        self.today = now.date() if isinstance(now, datetime) else now
        self.prefer = prefer

    # ---------- 年份推斷 ----------
    def _with_year(self, month: int, day: int) -> date | None:
        candidates = []
        for y in (self.today.year - 1, self.today.year, self.today.year + 1):
            try:
                candidates.append(date(y, month, day))
            except ValueError:
                continue
        if not candidates:
            return None
        if self.prefer == "future":
            upcoming = [d for d in candidates if d >= self.today]
            return min(upcoming) if upcoming else max(candidates)
        return min(candidates, key=lambda d: abs((d - self.today).days))

    def _day_only(self, day: int) -> date | None:
        """只有"26號"：本月該日，若已過（且偏好未來）則取下月。"""
        y, m = self.today.year, self.today.month
        for _ in range(3):
            try:
                d = date(y, m, day)
            except ValueError:
                d = None
            if d and (self.prefer != "future" or d >= self.today):
                return d
            m, y = (1, y + 1) if m == 12 else (m + 1, y)
        return None

    def _weekday(self, prefix: str | None, char: str) -> date:
        target = _WEEKDAY_CHAR[char]
        monday = self.today - timedelta(days=self.today.weekday())
        prefix = prefix or ""
        if prefix.startswith("下下"):
            return monday + timedelta(days=14 + target)
        if prefix.startswith("下"):
            return monday + timedelta(days=7 + target)
        if prefix.startswith("上"):
            return monday + timedelta(days=-7 + target)
        if prefix.startswith(("本", "這", fold("這"))):
            return monday + timedelta(days=target)
        # 單獨的"週五"：取今天起最近的一個
        return self.today + timedelta(days=(target - self.today.weekday()) % 7)

    # ---------- 抽取 ----------
    def find(self, text: str) -> list[DateAnchor]:
        taken: list[tuple[int, int]] = []
        anchors: list[DateAnchor] = []

        def free(s: int, e: int) -> bool:
            return all(e <= a or s >= b for a, b in taken)

        folded = fold(text)  # 長度不變：在摺疊文字上比對，再用同樣的位置切回原文
        for kind, pattern in _PATTERNS:
            for m in pattern.finditer(folded):
                if not free(m.start(), m.end()):
                    continue
                value = self._resolve(kind, m)
                if value is None:
                    continue
                taken.append((m.start(), m.end()))
                anchors.append(DateAnchor(text[m.start():m.end()], value, m.start(), m.end()))
        return sorted(anchors, key=lambda a: a.start)

    def _resolve(self, kind: str, m: re.Match[str]) -> date | None:
        try:
            if kind == "ymd":
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if kind == "md":
                month, day = cn_to_int(m.group(1)), cn_to_int(m.group(2))
                if not month or not day or month > 12 or day > 31:
                    return None
                return self._with_year(month, day)
            if kind == "rel":
                return self.today + timedelta(days=_RELATIVE[m.group(0)])
            if kind == "week":
                return self._weekday(m.group(1), m.group(2))
            if kind == "day":
                day = int(m.group(1))
                return self._day_only(day) if 1 <= day <= 31 else None
        except ValueError:
            return None
        return None

    def annotate(self, text: str) -> tuple[str, list[DateAnchor]]:
        """在每個日期表達後插入〔=YYYY-MM-DD 星期X〕，返回新文字與錨點列表。"""
        anchors = self.find(text)
        out, cursor = [], 0
        for a in anchors:
            out.append(text[cursor:a.end])
            if a.text.strip() != a.iso:
                out.append(f"〔={a.label()}〕")
            cursor = a.end
        out.append(text[cursor:])
        return "".join(out), anchors

    def now_line(self, now: datetime | None = None) -> str:
        now = now or datetime.now()
        return f"現在是 {now:%Y年%m月%d日 %H:%M}，{WEEKDAYS[now.weekday()]}。"


_MD_RE = _rx(_NUM + r"\s*月\s*" + _NUM)
_DAY_ONLY_RE = _rx(r"(\d{1,2})\s*[日號]?")


def month_day(text: str) -> tuple[int | None, int | None]:
    """供元素定位使用：'6月26日'→(6,26)，'26日'/'26號'/'26'→(None,26)，'2026-06-26'→(6,26)。"""
    text = fold(text.strip())
    m = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if m:
        return int(m.group(2)), int(m.group(3))
    m = _MD_RE.search(text)
    if m:
        return cn_to_int(m.group(1)), cn_to_int(m.group(2))
    m = _DAY_ONLY_RE.fullmatch(text)
    if m and 1 <= int(m.group(1)) <= 31:
        return None, int(m.group(1))
    return None, None
