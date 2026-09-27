"""示例：Daytona 雲沙箱（對應課程 CASE-daytona）。

    pip install daytona
    set DAYTONA_API_KEY=dtn_xxx        (PowerShell: $env:DAYTONA_API_KEY="dtn_xxx")
    python examples/daytona_hello.py

配置了 DAYTONA_API_KEY 後，靈犀會自動啟用 sandbox_run 技能：
模型想執行不可信程式碼時，可以把程式碼丟進雲端容器執行，只把結果帶回來。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lingxi.journal.recorder import Journal  # noqa: E402
from lingxi.kernel.context import RunContext  # noqa: E402
from lingxi.kernel.profiles import PROFILES  # noqa: E402
from lingxi.settings import get_settings  # noqa: E402
from lingxi.skills.remote import DaytonaRun  # noqa: E402


async def main() -> None:
    settings = get_settings()
    skill = DaytonaRun(settings.daytona)
    ctx = RunContext(task="demo", settings=settings, journal=Journal(), llm=None, profile=PROFILES["general"])
    try:
        outcome = await skill.invoke(ctx, {"code": 'print("Hello World from LingXi sandbox!")'})
        print(("✔ " if outcome.ok else "✘ ") + outcome.summary)
        if outcome.detail:
            print(outcome.detail)
    finally:
        await skill.close()  # 用完即刪，避免沙箱持續計費


if __name__ == "__main__":
    asyncio.run(main())
