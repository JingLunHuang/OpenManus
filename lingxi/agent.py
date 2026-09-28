"""對外門面：LingXi().run("任務") 一行跑起來。

    import asyncio
    from lingxi import LingXi
    result = asyncio.run(LingXi().run("查詢 6月26日 從上海到北京的機票"))
    print(result.answer)

每次執行會：
  1. 載入 RSI 目前生效的系統版本（harness 參數 ＋ 學到的手冊），見 lingxi/evolve；
  2. 掛上自我學習記憶（LightMem ＋ FluxMem）與 recall 技能，見 lingxi/memory；
  3. 執行結束後，累積滿 memory.sleep_every 次就自動做一次睡眠整理。
"""

from __future__ import annotations

from typing import Any

from lingxi.journal.console import console_listener
from lingxi.journal.recorder import Journal
from lingxi.kernel.context import HumanChannel
from lingxi.kernel.hooks import Hook, default_hooks
from lingxi.kernel.loop import Kernel, RunResult
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.llm.client import ChatModel, LLMClient
from lingxi.senses.vision import VisionOracle
from lingxi.settings import Settings, get_settings
from lingxi.skills import Skill, SkillSet, builtin_skills
from lingxi.skills.remote import DaytonaRun, McpHub


class _AnnounceState(Hook):
    """在黑匣子裡記下這次執行用的是哪個系統版本（RSI），復盤時才知道行為差異從何而來。"""

    def __init__(self, state):
        self.state = state

    async def on_start(self, ctx) -> None:
        ctx.emit("evolve", version=self.state.version, harness=self.state.harness, playbooks=self.state.playbooks)
        return None


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
        self.last_sleep: dict[str, Any] | None = None

    def library(self, settings: Settings | None = None, extra_dirs: list | None = None) -> PlaybookLibrary:
        s = settings or self.settings
        return PlaybookLibrary.load([s.path(d) for d in s.playbook_dirs] + list(extra_dirs or []))

    def _make_vision(self) -> VisionOracle | None:
        if self._vision is not None:
            return self._vision
        vs = self.settings.llm.vision
        if vs.enabled and vs.resolve_api_key():
            return VisionOracle(vs)
        return None

    def active_state(self):
        """RSI 目前生效的版本；evolve 關閉時回傳 None。"""
        if not self.settings.evolve.enabled:
            return None, None
        from lingxi.evolve import StateStore

        store = StateStore(self.settings.evolve_dir)
        return store, store.current()

    async def run(self, task: str, journal: Journal | None = None) -> RunResult:
        from lingxi.evolve import apply_harness

        store, state = self.active_state()
        s = apply_harness(self.settings, state.harness) if state else self.settings
        journal = journal or Journal.create(s.runs_dir, task)
        if self.console:
            journal.add_listener(console_listener)

        llm = self._llm or LLMClient(s.llm)
        skills = SkillSet(builtin_skills() + self.extra_skills)
        if s.daytona.resolve_api_key():
            skills.add(DaytonaRun(s.daytona))
        hub = McpHub(s.mcp_servers)
        mcp_skills, errors = await hub.connect()
        for skill in mcp_skills:
            skills.add(skill)
        for err in errors:
            journal.emit("guard", message=f"MCP 伺服器連線失敗，已跳過：{err}")

        hooks = list(self.hooks) if self.hooks is not None else default_hooks()
        memory = None
        if s.memory.enabled:
            from lingxi.memory import MemoryHook, MemorySystem, Recall

            # 睡眠整理的 consolidate_with=auto：只有真的連得上模型時才用模型，否則用規則
            real = isinstance(llm, LLMClient) and bool(llm.settings.resolve_api_key())
            memory = MemorySystem(s, llm=llm if real else None)
            hooks.append(MemoryHook(memory))
            skills.add(Recall(memory))
        if state is not None:
            hooks.insert(0, _AnnounceState(state))

        extra_dirs = [store.playbook_dir(state.version)] if state is not None else []
        kernel = Kernel(s, llm, skills, vision=self._make_vision(), library=self.library(s, extra_dirs),
                        human=self.human, hooks=hooks, journal=journal)
        try:
            result = await kernel.run(task)
        finally:
            await hub.close()
        if memory is not None and memory.due_for_sleep():
            self.last_sleep = await memory.sleep()
            if self.console:
                c = self.last_sleep["consolidation"]
                u = self.last_sleep["update"]
                print(f"\n🌙 睡眠整理（{self.last_sleep['mode']}）：合併 {u['merged']} 條、以新換舊 {u['updated']} 條；"
                      f"歸納出 {len(c['skills'])} 個程序技能", flush=True)
        return result
