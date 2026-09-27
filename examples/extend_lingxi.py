"""示例：如何擴充靈犀 —— 自定義技能 + 自定義鉤子，不改動框架內部任何一行程式碼。

    python examples/extend_lingxi.py "把 1 到 100 的質數加起來，再告訴我今天是星期幾"

1. 技能（Skill）：宣告一個 pydantic 參數模型 + 一個 run()，Schema 與參數校驗自動完成；
2. 鉤子（Hook）：brief() 往每一步的簡報裡加內容，after_step() 在每一步結束後觀察結果。
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lingxi import LingXi  # noqa: E402
from lingxi.kernel.hooks import Hook, default_hooks  # noqa: E402
from lingxi.skills.base import Outcome, Skill  # noqa: E402


class PrimeSumParams(BaseModel):
    upper: int = Field(ge=2, le=1_000_000, description="上限（含）")


class PrimeSum(Skill):
    name = "prime_sum"
    description = "計算 2..upper 之間所有質數之和。"
    Params = PrimeSumParams

    async def run(self, ctx, p: PrimeSumParams) -> Outcome:
        sieve = bytearray([1]) * (p.upper + 1)
        sieve[:2] = b"\x00\x00"
        for i in range(2, int(p.upper ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
        total = sum(i for i, flag in enumerate(sieve) if flag)
        ctx.findings.add([f"2..{p.upper} 的質數和 = {total}"], source="prime_sum", step=ctx.step)
        return Outcome(ok=True, summary=f"質數和 = {total}", verified=True, progress=["facts"])


class ClockHook(Hook):
    """每一步都把精確到秒的當前時間放進簡報。"""

    async def brief(self, ctx) -> str:
        return f"【時鐘】{datetime.now():%Y-%m-%d %H:%M:%S}"


async def main() -> None:
    task = " ".join(sys.argv[1:]) or "把 1 到 100 的質數加起來，再告訴我今天是星期幾"
    agent = LingXi(extra_skills=[PrimeSum()], hooks=[ClockHook(), *default_hooks()])
    result = await agent.run(task)
    print("\n最終答覆：", result.answer)


if __name__ == "__main__":
    asyncio.run(main())
