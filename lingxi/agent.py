"""對外門面：LingXi().run("任務") 一行跑起來。

    import asyncio
    from lingxi import LingXi
    result = asyncio.run(LingXi().run("查詢 6月26日 從上海到北京的機票"))
    print(result.answer)
"""

from __future__ import annotations

from lingxi.journal.console import console_listener
from lingxi.journal.recorder import Journal
from lingxi.kernel.context import HumanChannel
from lingxi.kernel.hooks import Hook
from lingxi.kernel.loop import Kernel, RunResult
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.llm.client import ChatModel, LLMClient
from lingxi.senses.vision import VisionOracle
from lingxi.settings import Settings, get_settings
from lingxi.skills import Skill, SkillSet, builtin_skills
from lingxi.skills.remote import DaytonaRun, McpHub


class LingXi:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: ChatModel | None = None,
        vision: VisionOracle | None = None,
        human: HumanChannel | None = None,
        console: bool = True,
        extra_skills: list[Skill] | None = None,
        hooks: list[Hook] | None = None,
    ):
        self.settings = settings or get_settings()
        self._llm = llm
        self._vision = vision
        self.human = human
        self.console = console
        self.extra_skills = extra_skills or []
        self.hooks = hooks

    def library(self) -> PlaybookLibrary:
        return PlaybookLibrary.load([self.settings.path(d) for d in self.settings.playbook_dirs])

    def _make_vision(self) -> VisionOracle | None:
        if self._vision is not None:
            return self._vision
        vs = self.settings.llm.vision
        if vs.enabled and vs.resolve_api_key():
            return VisionOracle(vs)
        return None

    async def run(self, task: str, journal: Journal | None = None) -> RunResult:
        s = self.settings
        journal = journal or Journal.create(s.runs_dir, task)
        if self.console:
            journal.add_listener(console_listener)

        skills = SkillSet(builtin_skills() + self.extra_skills)
        if s.daytona.resolve_api_key():
            skills.add(DaytonaRun(s.daytona))
        hub = McpHub(s.mcp_servers)
        mcp_skills, errors = await hub.connect()
        for skill in mcp_skills:
            skills.add(skill)
        for err in errors:
            journal.emit("guard", message=f"MCP 伺服器連線失敗，已跳過：{err}")

        kernel = Kernel(
            s,
            self._llm or LLMClient(s.llm),
            skills,
            vision=self._make_vision(),
            library=self.library(),
            human=self.human,
            hooks=self.hooks,
            journal=journal,
        )
        try:
            return await kernel.run(task)
        finally:
            await hub.close()
