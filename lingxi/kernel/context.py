"""RunContext：一次執行裡所有元件共享的狀態。

核心、技能、鉤子都只透過它交流，彼此之間沒有繼承關係
（對比 OpenManus 的 BaseAgent → ReActAgent → ToolCallAgent → Manus 四級繼承）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from lingxi.journal.recorder import Journal
from lingxi.kernel.boards import FindingsBoard, PlanBoard
from lingxi.llm.client import ChatModel
from lingxi.settings import Settings

if TYPE_CHECKING:
    from lingxi.kernel.profiles import Profile
    from lingxi.knowledge.playbook import CompiledPlaybook
    from lingxi.knowledge.temporal import DateAnchor
    from lingxi.senses.browser import BrowserSession
    from lingxi.senses.vision import VisionOracle


class HumanChannel(Protocol):
    async def ask(self, question: str) -> str: ...


class ConsoleHuman:
    async def ask(self, question: str) -> str:
        return (await asyncio.to_thread(input, f"\n❓ {question}\n你的回答> ")).strip()


class NoHuman:
    """無人值守模式（測試 / 批處理）：直接告訴模型沒有人可以回答。"""

    async def ask(self, question: str) -> str:
        return "（當前為無人值守模式，無法獲得人工回答，請自行決定或給出階段性結論）"


@dataclass
class RunContext:
    task: str
    settings: Settings
    journal: Journal
    llm: ChatModel
    profile: "Profile"
    briefed_task: str = ""
    vision: "VisionOracle | None" = None
    human: HumanChannel = field(default_factory=NoHuman)
    step: int = 0
    plan: PlanBoard = field(default_factory=PlanBoard)
    findings: FindingsBoard = field(default_factory=FindingsBoard)
    anchors: list["DateAnchor"] = field(default_factory=list)
    playbooks: list["CompiledPlaybook"] = field(default_factory=list)
    # 這些 URL 參數裡的日期若已過期會被自動順延（來自手冊 + 通用出行參數名）
    date_params: set[str] = field(
        default_factory=lambda: {"depdate", "date", "departDate", "checkin", "checkout", "rdate", "returnDate"}
    )
    # 主模型支援看圖時，下一次簡報附帶的截圖
    pending_image: str | None = None
    scratch: dict[str, Any] = field(default_factory=dict)
    _browser: "BrowserSession | None" = None

    @property
    def browser(self) -> "BrowserSession":
        if self._browser is None:
            from lingxi.senses.browser import BrowserSession

            self._browser = BrowserSession(self.settings.browser)
            for pb in self.playbooks:
                if pb.warmup:
                    from lingxi.senses.browser import site_key

                    self._browser.warmups[site_key(pb.url or pb.warmup)] = pb.warmup
        return self._browser

    @property
    def browser_started(self) -> bool:
        return self._browser is not None and self._browser.started

    def emit(self, kind: str, **data: Any):
        return self.journal.emit(kind, step=self.step, **data)

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
