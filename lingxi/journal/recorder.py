"""黑匣子（Journal）：一次執行的全部事件都追加到 events.jsonl。

它是整個框架唯一的"事實來源"：
- 控制台輸出、Web UI 的 SSE 推送都只是它的訂閱者；
- 截圖 / HTML 快照作為附件存進 artifacts/；
- 事後用 `lingxi replay <run>` 把它渲染成可離線檢視的時間線報告。

課程裡排障靠"手動加日誌、把網頁存下來"，這裡把它做成了框架的一等公民。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from lingxi.hanzi import to_trad, to_trad_deep

Listener = Callable[["Event"], None]
# 這些欄位是識別碼或網址，轉換會讓它們失效
_KEEP_RAW = frozenset({"url", "href", "artifacts", "run_id", "config", "point", "id"})


@dataclass
class Event:
    seq: int
    ts: float
    kind: str
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


def _slug(text: str, limit: int = 24) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", to_trad(text.strip()))
    return text[:limit].strip("-") or "run"


class Journal:
    def __init__(self, run_dir: Path | None = None, run_id: str | None = None):
        self.run_dir = run_dir
        self.run_id = run_id or (run_dir.name if run_dir else "memory")
        self.events: list[Event] = []
        self._listeners: list[Listener] = []
        self._queues: list[asyncio.Queue[Event | None]] = []
        self._fh = None
        self.closed = False
        if run_dir:
            (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            self._fh = open(run_dir / "events.jsonl", "a", encoding="utf-8")

    @classmethod
    def create(cls, runs_root: Path, task: str) -> "Journal":
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(task)}"
        run_dir = runs_root / run_id
        n = 1
        while run_dir.exists():
            n += 1
            run_dir = runs_root / f"{run_id}-{n}"
        return cls(run_dir)

    # ---------- 寫入 ----------
    def emit(self, kind: str, step: int = 0, **data: Any) -> Event:
        # 對外一律呈現繁體：事件裡的思考、摘要、答覆、網頁文字都先轉繁體再落地（網址等欄位保留原樣）
        data = to_trad_deep(data, _KEEP_RAW)
        event = Event(seq=len(self.events) + 1, ts=time.time(), kind=kind, step=step, data=data)
        self.events.append(event)
        if self._fh:
            self._fh.write(event.to_json() + "\n")
            self._fh.flush()
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # 訂閱者出錯不能影響主流程
                pass
        for q in list(self._queues):
            q.put_nowait(event)
        return event

    def save_artifact(self, name: str, content: bytes | str) -> str | None:
        """儲存附件，返回相對 run_dir 的路徑；純記憶體模式返回 None。"""
        if not self.run_dir:
            return None
        safe = re.sub(r"[^\w.\-]+", "_", name)
        path = self.run_dir / "artifacts" / f"{len(self.events):04d}_{safe}"
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return path.relative_to(self.run_dir).as_posix()

    def write_text(self, name: str, content: str) -> Path | None:
        if not self.run_dir:
            return None
        path = self.run_dir / name
        path.write_text(to_trad(content), encoding="utf-8")
        return path

    # ---------- 訂閱 ----------
    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def subscribe(self, replay: bool = True) -> asyncio.Queue[Event | None]:
        """供 SSE 使用：先回放已有事件，再即時推送。收到 None 表示結束。"""
        q: asyncio.Queue[Event | None] = asyncio.Queue()
        if replay:
            for e in self.events:
                q.put_nowait(e)
        if self.closed:
            q.put_nowait(None)
        else:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._queues:
            self._queues.remove(q)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._fh:
            self._fh.close()
            self._fh = None
        for q in self._queues:
            q.put_nowait(None)
        self._queues.clear()

    # ---------- 讀取 ----------
    @staticmethod
    def load(run_dir: Path) -> list[Event]:
        events = []
        with open(run_dir / "events.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(Event(**json.loads(line)))
        return events
