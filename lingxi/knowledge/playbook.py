"""站點手冊（Playbook）：結構化、可執行的經驗庫。

課程裡的 RAG 方案是把一段 travel_tools.txt 整個塞進系統提示，讓模型自己去讀、自己拼 URL。
靈犀把經驗寫成宣告式的 TOML 手冊，並在進入模型之前完成三件確定性的工作：

  1. 匹配：按觸發詞 / 站點名給任務打分，挑出最相關的手冊；
  2. 填槽：從任務裡抽取"出發地、目的地、日期"等槽位，查代碼表（上海→sha），日期來自時間錨定；
  3. 編譯：把 URL 模板編譯成可以直接開啟的地址，並登記"預熱首頁"
          —— 首次造訪該站點時瀏覽器會先開啟首頁建立會話（課程裡繞開 whaleguard 反爬的關鍵經驗），
          這一步在工具層強制執行，而不是寄希望於模型記得。

新增一個場景 = 在 playbooks/ 下新增一個 .toml 檔案，不需要改任何程式碼。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lingxi.hanzi import fold, unify
from lingxi.knowledge.temporal import DateAnchor


class _KeepMissing(dict):
    """format_map 時保留未知佔位符原樣，避免步驟文字裡的花括號導致報錯。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass
class Slot:
    name: str
    kind: str = "text"  # text | date | <代碼表名，如 city>
    patterns: list[str] = field(default_factory=list)
    index: int = 0  # kind=date 時取第幾個日期
    required: bool = True


@dataclass
class Playbook:
    id: str
    title: str
    triggers: list[str]
    sites: list[str] = field(default_factory=list)
    profile: str | None = None
    date_preference: str | None = None
    date_params: list[str] = field(default_factory=list)
    warmup: str | None = None
    url: str | None = None
    steps: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    codes: dict[str, dict[str, str]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_toml(cls, path: Path) -> "Playbook":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        route = raw.get("route", {})
        slots = [Slot(name=k, **v) for k, v in raw.get("slots", {}).items()]
        return cls(
            id=raw["id"],
            title=raw.get("title", raw["id"]),
            triggers=raw.get("triggers", []),
            sites=raw.get("sites", []),
            profile=raw.get("profile"),
            date_preference=raw.get("date_preference"),
            date_params=raw.get("date_params", []),
            warmup=route.get("warmup"),
            url=route.get("url"),
            steps=route.get("steps", []),
            tips=raw.get("tips", []),
            slots=slots,
            codes=raw.get("codes", {}),
            source=str(path),
        )

    # ---------- 匹配 ----------
    def score(self, task: str) -> tuple[float, list[str]]:
        low = unify(task.lower())  # 繁簡、兩岸用語都能命中
        hits = [t for t in self.triggers if unify(t.lower()) in low]
        site_hits = [s for s in self.sites if unify(s.lower()) in low]
        if not hits and not site_hits:
            return 0.0, []
        return 3.0 * len(hits) + 6.0 * len(site_hits), hits + site_hits

    # ---------- 填槽 ----------
    def fill(self, task: str, anchors: list[DateAnchor]) -> tuple[dict[str, str], dict[str, str], list[str]]:
        """返回 (編譯用的值, 給人看的值, 缺失的槽位)。

        正規表示式與代碼表都在 fold（逐字繁→簡、長度不變）後的文字上比對，
        再按相同位置從原文取值，因此繁體或簡體寫的「從廣州」都得到同一個代碼，並各自顯示使用者的原文。
        """
        values, shown, missing = {}, {}, []
        folded_task = fold(task)
        for slot in self.slots:
            raw_value = None
            if slot.kind == "date":
                if len(anchors) > slot.index:
                    raw_value = anchors[slot.index].iso
            else:
                for pattern in slot.patterns:
                    m = re.search(fold(pattern), folded_task)
                    if m:
                        group = "v" if "v" in m.groupdict() else (1 if m.groups() else 0)
                        start, end = m.span(group)
                        raw_value = task[start:end].strip()
                        break
            if not raw_value:
                if slot.required:
                    missing.append(slot.name)
                continue
            shown[slot.name] = raw_value
            table = self.codes.get(slot.kind)
            if table is not None:
                # 正則可能多抓了字首（"查詢上海"），取被包含的最長的已知名稱
                key = fold(raw_value)
                folded_names = {fold(n): n for n in table}
                names = [key] if key in folded_names else (
                    [n for n in folded_names if n in key] or [n for n in folded_names if key in n])
                if not names:
                    missing.append(f"{slot.name}（代碼表裡沒有「{raw_value}」）")
                    continue
                name = max(names, key=len)
                at = key.find(name)
                shown[slot.name] = raw_value[at:at + len(name)] if at >= 0 else folded_names[name]
                values[slot.name] = table[folded_names[name]]
            else:
                values[slot.name] = raw_value
        return values, shown, missing


@dataclass
class CompiledPlaybook:
    playbook: Playbook
    score: float
    matched: list[str]
    values: dict[str, str]
    shown: dict[str, str]
    missing: list[str]
    url: str | None

    @property
    def id(self) -> str:
        return self.playbook.id

    @property
    def title(self) -> str:
        return self.playbook.title

    @property
    def warmup(self) -> str | None:
        return self.playbook.warmup

    def route_line(self) -> str:
        if self.url:
            return f"→ {self.url}"
        if self.missing:
            return f"（缺少槽位：{'、'.join(self.missing)}）"
        return ""

    def render(self) -> str:
        pb = self.playbook
        lines = [f"■ 手冊《{pb.title}》（命中：{'、'.join(self.matched)}）"]
        if self.shown:
            lines.append("  已識別：" + "，".join(f"{k}={v}" for k, v in self.shown.items()))
        if self.url:
            lines.append(f"  直達地址（已編譯，可直接 web_open）：{self.url}")
        elif pb.url:
            lines.append(f"  地址模板：{pb.url}（缺少 {'、'.join(self.missing)}，請先補全或改用頁面互動）")
        if pb.warmup:
            lines.append(f"  會話預熱：首次造訪該站點時，瀏覽器會自動先開啟 {pb.warmup}")
        values = _KeepMissing(url=self.url or pb.url or "", **self.shown)
        for i, step in enumerate(pb.steps, 1):
            lines.append(f"  步驟{i}：{step.format_map(values)}")
        for tip in pb.tips:
            lines.append(f"  提示：{tip}")
        return "\n".join(lines)


class PlaybookLibrary:
    def __init__(self, playbooks: list[Playbook] | None = None):
        self.playbooks = playbooks or []

    @classmethod
    def load(cls, dirs: list[Path]) -> "PlaybookLibrary":
        books = []
        for d in dirs:
            if d.is_dir():
                for path in sorted(d.glob("*.toml")):
                    books.append(Playbook.from_toml(path))
        return cls(books)

    def match(self, task: str, anchors: list[DateAnchor], top_k: int = 2, min_score: float = 3.0) -> list[CompiledPlaybook]:
        scored = []
        for pb in self.playbooks:
            s, matched = pb.score(task)
            if s < min_score:
                continue
            values, shown, missing = pb.fill(task, anchors)
            url = None
            if pb.url and not missing:
                try:
                    url = pb.url.format(**values)
                except KeyError as exc:
                    missing.append(str(exc))
            scored.append(CompiledPlaybook(pb, s + (2 if not missing else 0), matched, values, shown, missing, url))
        scored.sort(key=lambda c: -c.score)
        return scored[:top_k]

    def describe(self) -> list[dict[str, Any]]:
        return [{"id": p.id, "title": p.title, "triggers": p.triggers, "sites": p.sites, "source": p.source}
                for p in self.playbooks]
