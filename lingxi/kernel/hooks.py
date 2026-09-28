"""鉤子（Hook）：以組合代替繼承的擴充點。

核心會：
  0. 開始前呼叫 hook.on_start(ctx) 收集系統提示片段（整次執行不變，對 KV 快取友善，例如經驗記憶）；
  每一步：
  1. 依次呼叫 hook.brief(ctx) 收集簡報片段（頁面感知、白板、警告……）；
  2. 執行工具；
  3. 依次呼叫 hook.after_step(ctx, calls, outcomes, budget)；
  結束時呼叫 hook.on_finish(ctx, status, answer)（例如把這次經驗寫入長期記憶）。
想加新能力（例如"每步自動截圖給多模態主模型"），寫一個 Hook 掛上去即可，不需要改核心。
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from lingxi.llm.messages import ToolCall
from lingxi.skills.base import Outcome

if TYPE_CHECKING:
    from lingxi.kernel.budget import ProgressBudget
    from lingxi.kernel.context import RunContext


class Hook:
    async def on_start(self, ctx: "RunContext") -> str | None:
        return None

    async def brief(self, ctx: "RunContext") -> str | None:
        return None

    async def after_step(self, ctx: "RunContext", calls: list[ToolCall], outcomes: list[Outcome],
                         budget: "ProgressBudget") -> None:
        return None

    async def on_finish(self, ctx: "RunContext", status: str, answer: str) -> None:
        return None


class PageSense(Hook):
    """瀏覽器已開啟時，把最新頁面快照渲染進簡報；網址變化時記一筆 page 事件（記憶與復盤用）。"""

    async def brief(self, ctx: "RunContext") -> str | None:
        if not ctx.browser_started:
            ctx.scratch["page_fp"] = "no-browser"
            return None
        try:
            snap = await ctx.browser.snapshot()
        except Exception as exc:
            ctx.scratch["page_fp"] = "error"
            return f"【頁面】讀取頁面狀態失敗：{exc}"
        ctx.scratch["page_fp"] = snap.fingerprint()
        if snap.url != ctx.scratch.get("page_url"):
            ctx.scratch["page_url"] = snap.url
            ctx.emit("page", url=snap.url, title=snap.title)
        b = ctx.settings.browser
        return snap.render(max_per_kind=b.max_elements_per_kind, digest_chars=b.digest_chars)


class BoardSense(Hook):
    async def brief(self, ctx: "RunContext") -> str | None:
        parts = [ctx.plan.render(), ctx.findings.render()]
        return "\n".join(p for p in parts if p) or None


class LoopGuard(Hook):
    """原地打轉檢測：同一頁面狀態下重複同一個動作 / 同一工具連續失敗 / 連續多步沒有進展。"""

    def __init__(self, repeat_limit: int = 2, fail_limit: int = 3, idle_limit: int = 4):
        self.repeat_limit = repeat_limit
        self.fail_limit = fail_limit
        self.idle_limit = idle_limit
        self.seen: Counter[tuple[str, str]] = Counter()
        self.fail_streak: Counter[str] = Counter()
        self.idle_steps = 0
        self.pending: list[str] = []

    async def brief(self, ctx: "RunContext") -> str | None:
        if not self.pending:
            return None
        text = "【警告】" + "\n      ".join(self.pending)
        self.pending = []
        return text

    async def after_step(self, ctx, calls, outcomes, budget) -> None:
        page_fp = ctx.scratch.get("page_fp", "")
        progressed = False
        for call, outcome in zip(calls, outcomes):
            key = (call.fingerprint(), page_fp)
            self.seen[key] += 1
            if self.seen[key] > self.repeat_limit:
                delta = budget.penalize(2, "原地打轉")
                msg = (f"你在同一頁面狀態下第 {self.seen[key]} 次執行 {call.name}，結果不會不同。"
                       f"請換一種方法（換入口 / 換定位方式 / 關閉浮層 / 直接構造 URL），或基於已有資訊 finish。")
                self.pending.append(msg)
                ctx.emit("guard", message=msg, penalty=delta)
            if outcome.ok:
                self.fail_streak[call.name] = 0
            else:
                self.fail_streak[call.name] += 1
                if self.fail_streak[call.name] >= self.fail_limit:
                    msg = f"{call.name} 已連續失敗 {self.fail_streak[call.name]} 次，請停止重試同一種做法。"
                    self.pending.append(msg)
                    ctx.emit("guard", message=msg)
                    self.fail_streak[call.name] = 0
            progressed = progressed or bool(outcome.progress)
        self.idle_steps = 0 if progressed else self.idle_steps + 1
        if self.idle_steps >= self.idle_limit:
            msg = f"已經連續 {self.idle_steps} 步沒有新進展。請評估：換策略，還是用已掌握的資訊給出階段性結論（finish）。"
            self.pending.append(msg)
            ctx.emit("guard", message=msg)
            self.idle_steps = 0


def default_hooks() -> list[Hook]:
    return [BoardSense(), LoopGuard(), PageSense()]
