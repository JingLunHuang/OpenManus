"""靈犀核心：Sense → Think → Act → Verify 迴圈。

一次執行的完整流程：

  預處理（確定性，不花 token）
    時間錨定：把任務裡的日期解析成絕對日期寫回任務
    手冊匹配：命中站點手冊 → 填槽 → 編譯直達 URL → 登記會話預熱
    意圖路由：選定模式（工具集 / 預算 / 指引）

  主迴圈（每一步）
    Sense   鉤子生成本步簡報：白板、警告、頁面快照（只在本步出現，不進歷史）
    Think   模型基於"系統提示 + 摺疊後的歷史 + 本步簡報"選擇工具
    Act     執行技能；瀏覽器技能內部完成 定位 → 執行 → 驗證
    Verify  彙總進展訊號：續航或扣減預算；迴圈檢測；最後一步強制 finish

  收尾
    結果寫入黑匣子（events.jsonl / result.md），釋放瀏覽器與外部連線
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from lingxi.hanzi import to_trad, to_trad_many
from lingxi.journal.recorder import Journal
from lingxi.kernel.budget import ProgressBudget
from lingxi.kernel.context import HumanChannel, NoHuman, RunContext
from lingxi.kernel.hooks import Hook, default_hooks
from lingxi.kernel.memory import RollingMemory
from lingxi.kernel.preflight import preflight
from lingxi.kernel.verify import fact_check
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.llm.client import ChatModel, LLMError
from lingxi.llm.kvcache import PrefixMeter
from lingxi.llm.messages import Message, ToolCall, Usage
from lingxi.prompts import briefing, system_prompt
from lingxi.senses.page import compact_text
from lingxi.settings import Settings
from lingxi.skills.base import Outcome, SkillSet
from lingxi.utils import clip


@dataclass
class RunResult:
    answer: str
    status: str
    steps: int
    usage: dict[str, int]
    run_dir: str | None = None
    findings: list[str] = field(default_factory=list)
    verification: dict | None = None


class Kernel:
    def __init__(
        self,
        settings: Settings,
        llm: ChatModel,
        skills: SkillSet,
        *,
        vision: Any = None,
        library: PlaybookLibrary | None = None,
        human: HumanChannel | None = None,
        hooks: list[Hook] | None = None,
        journal: Journal | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.skills = skills
        self.vision = vision
        self.library = library or PlaybookLibrary()
        self.human = human or NoHuman()
        self.hooks = hooks if hooks is not None else default_hooks()
        self.journal = journal

    # ---------------- 預處理 ----------------
    async def prepare(self, task: str, journal: Journal) -> RunContext:
        pf = await preflight(task, self.library, self.llm, use_llm=self.settings.intent.llm_fallback)
        ctx = RunContext(task=task, settings=self.settings, journal=journal, llm=self.llm, profile=pf.profile,
                         briefed_task=pf.briefed, vision=self.vision, human=self.human,
                         anchors=pf.anchors, playbooks=pf.playbooks)
        for pb in pf.playbooks:
            ctx.date_params |= set(pb.playbook.date_params)

        ctx.emit("run.start", task=task, profile=pf.profile.name, config=self.settings.source)
        ctx.emit("intent", profile=pf.profile.name, title=pf.profile.title, reason=pf.reason, scores=pf.scores)
        if pf.anchors:
            ctx.emit("temporal", anchors=[{"text": a.text, "date": a.label()} for a in pf.anchors])
        for pb in pf.playbooks:
            ctx.emit("playbook", id=pb.id, title=pb.title, route=pb.route_line(), slots=pb.shown, missing=pb.missing)
        return ctx

    # ---------------- 主迴圈 ----------------
    async def run(self, task: str) -> RunResult:
        journal = self.journal or Journal.create(self.settings.runs_dir, task)
        ctx = await self.prepare(task, journal)
        kv = self.settings.kv_cache
        budget = ProgressBudget.for_profile(ctx.profile.base_budget, self.settings.agent.hard_cap_factor)
        memory = RollingMemory(self.settings.agent.fold_after_turns, self.settings.agent.max_observation_chars,
                               fold_block=kv.fold_block if kv.mode != "off" else 1)
        extra = [s for h in self.hooks if (s := await h.on_start(ctx))]
        # 系統提示整次執行不變（經驗記憶也在這裡），是 KV 快取前綴的第一段
        system = Message.system(system_prompt(ctx, extra))
        system.cache = kv.mode == "explicit"
        allowed = [n for n in self.skills.names() if ctx.profile.allows(n)]
        meter = PrefixMeter()
        ctx.scratch["kv_meter"] = meter

        final: Outcome | None = None
        status, answer = "failed", ""
        verification: dict | None = None
        silent_replies = 0
        started = time.time()
        try:
            while budget.remaining > 0 and final is None:
                budget.spend()
                ctx.step += 1
                last = budget.is_last
                ctx.emit("step", remaining=budget.remaining, used=budget.used)

                sections = [s for h in self.hooks if (s := await h.brief(ctx))]
                brief = briefing(ctx, budget, sections, last)
                images = [ctx.pending_image] if ctx.pending_image and self.settings.llm.supports_vision else []
                ctx.pending_image = None
                history = memory.messages()
                if kv.mode == "explicit" and history:  # 第二個快取斷點：歷史的最後一則
                    history[-1] = replace(history[-1], cache=True)
                messages = [system, *history, Message.user(brief, images=images)]
                tool_choice: str | dict = "auto"
                if last and kv.stable_tools and kv.mode != "off":
                    tools = self.skills.schemas(allowed)  # 工具清單不變，前綴快取不失效
                    tool_choice = {"type": "function", "function": {"name": "finish"}}
                else:
                    tools = self.skills.schemas(["finish"] if last else allowed)
                reuse = meter.observe([m.to_openai() for m in messages], tools)

                try:
                    reply = await self.llm.chat(messages, tools=tools, tool_choice=tool_choice)
                except LLMError as exc:
                    ctx.emit("error", message=f"模型呼叫失敗：{exc}")
                    answer = f"模型呼叫失敗，任務中斷：{exc}"
                    break
                ctx.emit("think", content=reply.content,
                         calls=[{"name": c.name, "args": c.args()} for c in reply.tool_calls],
                         tokens=reply.prompt_tokens + reply.completion_tokens,
                         cached=reply.cached_tokens, reuse=round(reuse, 3))

                if not reply.tool_calls:
                    silent_replies += 1
                    if last or silent_replies >= 2:
                        # 兜底：模型堅持直接回答，就把它的文字當作最終答覆（仍然附上查證摘要，但不再退回）
                        final = Outcome(ok=True, summary="模型直接給出答覆", detail=reply.content, finish=True,
                                        data={"answer": reply.content, "status": "partial" if last else "success"})
                        final = await self._verify_gate(ctx, final, budget, last=True)
                        break
                    memory.add_turn(Message.assistant(reply.content), [],
                                    note="（系統）請透過呼叫工具推進任務；如果已經可以答覆，請呼叫 finish 提交。")
                    continue
                silent_replies = 0

                board_before = ctx.findings.as_list()
                outcomes, tool_messages = await self._act(ctx, reply.tool_calls, budget, last)
                memory.add_turn(Message.assistant(reply.content, reply.tool_calls), tool_messages)
                final = next((o for o in outcomes if o.finish), None)
                if ctx.findings.as_list() != board_before:
                    ctx.emit("findings", items=ctx.findings.as_list(), stats=ctx.findings.stats())

                for hook in self.hooks:
                    await hook.after_step(ctx, reply.tool_calls, outcomes, budget)
                signals = {s for o in outcomes for s in o.progress}
                delta, reason = budget.reward(signals)
                ctx.emit("budget", delta=delta, reason=reason, remaining=budget.remaining, granted=budget.granted)

            if final is not None:
                answer = final.data.get("answer", final.detail)
                status = final.data.get("status", "success")
                verification = final.data.get("verify")
            elif not answer:
                answer = self._salvage(ctx)
                status = "partial"
        finally:
            for hook in self.hooks:
                try:
                    await hook.on_finish(ctx, status, answer)
                except Exception as exc:  # 記憶寫入等收尾工作失敗不影響答覆
                    ctx.emit("guard", message=f"收尾鉤子 {type(hook).__name__} 失敗：{clip(str(exc), 200)}")
            usage = self._usage()
            kv_stats = {**meter.as_dict(), "cached_tokens": usage.get("cached_tokens", 0),
                        "hit_rate": round(usage.get("cached_tokens", 0) / usage["prompt_tokens"], 4)
                        if usage.get("prompt_tokens") else 0.0,
                        "mode": kv.mode, "fold_block": memory.fold_block}
            ctx.emit("run.finish", status=status, answer=answer, usage=usage, verify=verification,
                     findings=ctx.findings.stats(), seconds=round(time.time() - started, 1),
                     memory_chars=memory.size(), kv=kv_stats)
            self._write_result(ctx, status, answer, usage)
            await ctx.close()
            await self.skills.close()
            journal.close()

        return RunResult(answer=to_trad(answer), status=status, steps=ctx.step, usage=usage,
                         run_dir=str(journal.run_dir) if journal.run_dir else None,
                         findings=to_trad_many([f.text for f in ctx.findings.items]), verification=verification)

    # ---------------- 反覆查證 ----------------
    def _evidence(self, ctx: RunContext) -> str:
        parts = []
        board = ctx.findings.evidence_text(6000)
        if board:
            parts.append(board)
        observations = ctx.scratch.get("observations", [])[-6:]
        for i, (name, text) in enumerate(observations, 1):
            parts.append(f"E{i}（{name} 的輸出）{clip(text, 1500)}")
        snap = ctx.browser.last_snapshot if ctx.browser_started else None
        if snap and snap.digest:
            parts.append(f"E{len(observations) + 1}（目前頁面 {snap.url} 的正文）{clip(compact_text(snap.digest), 3000)}")
        return "\n".join(parts)

    async def _verify_gate(self, ctx: RunContext, outcome: Outcome, budget: ProgressBudget, last: bool) -> Outcome:
        vs = self.settings.verify
        if not vs.enabled or ctx.profile.name not in vs.profiles:
            return outcome
        rounds = ctx.scratch.get("verify_rounds", 0) + 1
        ctx.scratch["verify_rounds"] = rounds
        answer = outcome.data.get("answer", outcome.detail)
        report = await fact_check(self.llm, ctx.briefed_task, answer, self._evidence(ctx), rounds)
        retry = not report.passed and not last and rounds <= vs.max_rounds and budget.remaining >= 2
        ctx.emit("verify", action="reject" if retry else "accept", **report.as_dict())
        if retry:
            return Outcome.fail(f"答覆未通過第 {rounds} 輪查證（{len(report.problems)} 條陳述無據或矛盾），已退回補查",
                                report.feedback(), data={"verify": report.as_dict()})
        final_answer = answer + report.appendix(rounds)
        return replace(outcome, detail=final_answer,
                       data={**outcome.data, "answer": final_answer, "verify": {**report.as_dict(), "rounds": rounds}})

    async def _act(self, ctx: RunContext, calls: list[ToolCall], budget: ProgressBudget,
                   allowed_last: bool) -> tuple[list[Outcome], list[Message]]:
        outcomes: list[Outcome] = []
        tool_messages: list[Message] = []
        finished = False
        for call in calls:
            if finished:
                outcome = Outcome.fail("任務已提交 finish，此呼叫被跳過")
            elif allowed_last and call.name != "finish":
                outcome = Outcome.fail("預算已用盡，只能呼叫 finish")
            else:
                skill = self.skills.get(call.name)
                if skill is None:
                    outcome = Outcome.fail(f"未知工具：{call.name}", f"可用工具：{', '.join(self.skills.names())}")
                else:
                    try:
                        outcome = await skill.invoke(ctx, call.args())
                    except Exception as exc:  # 技能內部未處理的異常，不讓它打斷整個執行
                        outcome = Outcome.fail(f"{call.name} 執行異常：{clip(str(exc), 200)}")
                    if outcome.finish:
                        outcome = await self._verify_gate(ctx, outcome, budget, allowed_last)
                    elif outcome.ok:
                        ctx.scratch.setdefault("observations", []).append((call.name, outcome.for_model(3000)))
            finished = finished or outcome.finish
            ctx.emit("outcome", name=call.name, ok=outcome.ok, verified=outcome.verified,
                     summary=outcome.summary, detail=clip(outcome.detail, 2000),
                     grounding=outcome.grounding, artifacts=outcome.artifacts, progress=outcome.progress)
            outcomes.append(outcome)
            tool_messages.append(Message.tool(call, outcome.for_model(self.settings.agent.max_observation_chars),
                                              digest=("✔ " if outcome.ok else "✘ ") + outcome.summary))
        return outcomes, tool_messages

    def _salvage(self, ctx: RunContext) -> str:
        """沒有正常 finish 時，至少把已經確認的發現交給使用者。"""
        if ctx.findings.items:
            return "任務未能正常完成，以下是已確認的發現：\n" + "\n".join(f"- {f.text}" for f in ctx.findings.items)
        return "任務未能完成，也沒有收集到可靠的資訊。請檢視執行報告瞭解過程。"

    def _usage(self) -> dict[str, int]:
        total = Usage()
        for meter in (getattr(self.llm, "usage", None), getattr(self.vision, "usage", None)):
            if meter:
                total.calls += meter.calls
                total.prompt_tokens += meter.prompt_tokens
                total.completion_tokens += meter.completion_tokens
                total.cached_tokens += getattr(meter, "cached_tokens", 0)
        return total.as_dict()

    def _write_result(self, ctx: RunContext, status: str, answer: str, usage: dict[str, int]) -> None:
        lines = [f"# {ctx.task}", "", f"- 狀態：{status}", f"- 步數：{ctx.step}",
                 f"- 模型呼叫：{usage['calls']} 次，{usage['total_tokens']} tokens", "", "## 答覆", "", answer]
        if ctx.findings.items:
            lines += ["", "## 發現板（反覆查證狀態）", ""]
            for i, f in enumerate(ctx.findings.items, 1):
                line = f"- F{i} {f.mark}：{f.text}（來源：{'；'.join(f.sources) or '未註明'}）"
                if f.conflicts:
                    line += "；矛盾：" + "；".join(f.conflicts)
                lines.append(line)
        ctx.journal.write_text("result.md", "\n".join(lines))
