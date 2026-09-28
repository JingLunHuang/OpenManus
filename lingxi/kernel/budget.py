"""進展感知的步數預算（Progress-aware Budget）。

OpenManus 固定 max_steps=20：課程裡"終於選對了單程和日期，20 步也用完了"，
只好把上限改成 25、30……改大又會讓原地打轉的任務白白燒 token。

靈犀的規則：
  - 每一步消耗 1 點預算；
  - 這一步取得了"可驗證的進展"（新頁面、已驗證的互動、新事實、計劃完成、產出檔案），返還 1 點；
  - 被判定為原地打轉，額外扣 2 點；
  - 總發放量不超過 基礎預算 × hard_cap_factor；
  - 最後 1 點預算只能用來 finish：強制基於已有資訊收尾，絕不"什麼都沒交代就結束"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

STRONG_SIGNALS = {"facts", "plan_done", "artifact", "results"}


@dataclass
class ProgressBudget:
    base: int
    cap: int
    remaining: int = 0
    granted: int = 0
    used: int = 0
    history: list[tuple[int, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining = self.granted = self.base

    @classmethod
    def for_profile(cls, base: int, factor: float) -> "ProgressBudget":
        return cls(base=base, cap=max(base, int(base * factor)))

    def spend(self) -> None:
        self.used += 1
        self.remaining -= 1

    @property
    def is_last(self) -> bool:
        """當前這一步是不是最後一步（spend 之後呼叫）。"""
        return self.remaining <= 0

    def reward(self, signals: set[str]) -> tuple[int, str]:
        if not signals:
            return 0, ""
        strong = signals & STRONG_SIGNALS
        solid = {"new_url", "verified_action"} & signals
        if not (strong or solid):
            return 0, ""
        if self.granted >= self.cap:
            return 0, "已達預算上限"
        self.granted += 1
        self.remaining += 1
        reason = "取得進展：" + "、".join(sorted(strong | solid))
        self.history.append((1, reason))
        return 1, reason

    def penalize(self, amount: int, reason: str) -> int:
        if self.remaining <= 1:
            # 已經在收尾階段（最後一步跑完時 remaining = 0）：懲罰不能反過來把預算「補回」1 步，
            # 否則模型在最後一步仍不 finish 時會無限迴圈
            return 0
        before = self.remaining
        self.remaining = max(1, self.remaining - amount)  # 至少留 1 步用於收尾
        delta = self.remaining - before
        if delta:
            self.history.append((delta, reason))
        return delta
