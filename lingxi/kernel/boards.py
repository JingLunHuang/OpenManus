"""兩塊"白板"：計劃板與發現板。

它們不放在對話歷史裡，而是每一步都重新渲染進簡報。
好處：歷史被摺疊壓縮時，關鍵進度和已查到的事實不會丟；長任務不會"忘了自己做到哪"。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanItem:
    text: str
    done: bool = False
    note: str = ""


@dataclass
class PlanBoard:
    items: list[PlanItem] = field(default_factory=list)

    def set(self, texts: list[str]) -> None:
        self.items = [PlanItem(t.strip()) for t in texts if t.strip()]

    def mark(self, index: int, done: bool = True, note: str = "") -> PlanItem:
        item = self.items[index - 1]  # 對模型使用 1 起始編號
        item.done = done
        if note:
            item.note = note
        return item

    @property
    def progress(self) -> tuple[int, int]:
        return sum(i.done for i in self.items), len(self.items)

    def render(self) -> str:
        if not self.items:
            return ""
        done, total = self.progress
        lines = [f"【計劃】{done}/{total} 已完成"]
        for n, item in enumerate(self.items, 1):
            mark = "✔" if item.done else "☐"
            lines.append(f"  {mark} {n}. {item.text}" + (f" —— {item.note}" if item.note else ""))
        return "\n".join(lines)


@dataclass
class Finding:
    text: str
    source: str = ""
    step: int = 0
    # 反覆查證：其他來源印證過的（去重後的來源列表）與矛盾說明
    sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source and self.source not in self.sources:
            self.sources.insert(0, self.source)

    @property
    def status(self) -> str:
        if self.conflicts:
            return "conflict"
        return "corroborated" if len(self.sources) >= 2 else "single"

    @property
    def mark(self) -> str:
        return {"conflict": "⚠有矛盾", "corroborated": f"✔{len(self.sources)}方印證", "single": "○單一來源"}[self.status]


@dataclass
class FindingsBoard:
    """發現板：事實 + 來源。每條事實都有編號 F1、F2…，並標示印證狀態，方便反覆查證。"""

    items: list[Finding] = field(default_factory=list)

    def add(self, texts: list[str], source: str = "", step: int = 0) -> int:
        from lingxi.hanzi import unify

        added = 0
        for t in (t.strip() for t in texts if t and t.strip()):
            same = next((f for f in self.items if unify(f.text) == unify(t)), None)
            if same:  # 同一事實出現在另一個來源 → 視為印證
                if source and source not in same.sources:
                    same.sources.append(source)
                continue
            self.items.append(Finding(t, source, step))
            added += 1
        return added

    def get(self, ref: int | str) -> Finding | None:
        try:
            index = int(str(ref).upper().lstrip("F"))
        except ValueError:
            return None
        return self.items[index - 1] if 1 <= index <= len(self.items) else None

    def corroborate(self, ref: int | str, source: str) -> bool:
        f = self.get(ref)
        if f and source and source not in f.sources:
            f.sources.append(source)
            return True
        return False

    def contradict(self, ref: int | str, source: str, note: str) -> bool:
        f = self.get(ref)
        if f:
            f.conflicts.append(f"{note}（{source}）" if source else note)
            return True
        return False

    def stats(self) -> dict[str, int]:
        out = {"single": 0, "corroborated": 0, "conflict": 0}
        for f in self.items:
            out[f.status] += 1
        return out

    def as_list(self) -> list[dict]:
        return [{"id": f"F{i}", "text": f.text, "status": f.status, "sources": f.sources, "conflicts": f.conflicts}
                for i, f in enumerate(self.items, 1)]

    def evidence_text(self, max_chars: int = 6000) -> str:
        lines = []
        for i, f in enumerate(self.items, 1):
            line = f"F{i} [{f.mark}] {f.text}｜來源：{'；'.join(f.sources) or '未註明'}"
            if f.conflicts:
                line += "｜矛盾：" + "；".join(f.conflicts)
            lines.append(line)
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[-max_chars:]

    def render(self, max_chars: int = 2000) -> str:
        if not self.items:
            return ""
        kept: list[str] = []
        used = 0
        for i in range(len(self.items), 0, -1):  # 優先保留最新的
            f = self.items[i - 1]
            line = f"  F{i} {f.mark} {f.text}" + (f"（來源：{f.sources[0]}）" if f.sources else "")
            if f.conflicts:
                line += f" ⚠ {f.conflicts[-1]}"
            if used + len(line) > max_chars:
                break
            kept.append(line)
            used += len(line)
        s = self.stats()
        head = [f"【發現板】共 {len(self.items)} 條：✔ 多方印證 {s['corroborated']} · ○ 單一來源 {s['single']}"
                f" · ⚠ 有矛盾 {s['conflict']}（關鍵結論請盡量取得兩個以上獨立來源）"]
        if len(kept) < len(self.items):
            head.append(f"  · …更早的 {len(self.items) - len(kept)} 條已省略")
        return "\n".join(head + kept[::-1])
