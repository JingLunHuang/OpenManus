"""受保護端到端評測的離線乾跑器。

此處只證明 Kernel 任務路徑、預算對等及 RSI 閘門接線；ScriptedLLM 的合成
token 不能當成真實模型成本或真實站點成功率。題目只由 protected.py 傳入。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from lingxi.evolve.state import apply_harness
from lingxi.evaluation.fixtures import build_site_dir
from lingxi.journal.recorder import Journal
from lingxi.kernel.loop import Kernel
from lingxi.knowledge.playbook import Playbook, PlaybookLibrary
from lingxi.llm.messages import Reply, ToolCall, Usage
from lingxi.skills import SkillSet, builtin_skills


class _ScriptedModel:
    def __init__(self, script: list[dict]):
        self.script = iter(script)
        self.usage = Usage()

    async def chat(self, messages, tools=None, tool_choice="auto", **kwargs) -> Reply:
        try:
            item = next(self.script)
        except StopIteration as exc:
            raise RuntimeError("受保護任務的假模型劇本已用盡") from exc
        self.usage.add(50, 10)  # 固定合成成本；基準與候選完全相同
        if "tool" in item:
            call = ToolCall(f"e2e-{self.usage.calls}", item["tool"],
                            json.dumps(item.get("args", {}), ensure_ascii=False))
            return Reply(tool_calls=[call], prompt_tokens=50, completion_tokens=10)
        return Reply(content=item.get("content", ""), prompt_tokens=50, completion_tokens=10)


def _fixture(root: Path) -> str:
    return build_site_dir(root).as_uri()


def run_suite(data: dict, settings, harness: dict, playbook_files: list[Path]) -> dict:
    """同一份題目、基礎設定與合成成本，逐題使用全新隔離工作區。"""
    with tempfile.TemporaryDirectory(prefix="protected-e2e-", dir=settings.evolve_dir) as tmp:
        root = Path(tmp)
        fixture = _fixture(root)
        rows = []
        for index, case in enumerate(data["cases"]):
            s = apply_harness(settings, harness)
            s.llm.api_key = ""
            s.llm.api_key_env = "LINGXI_E2E_NO_SUCH_KEY"
            s.llm.vision.enabled = False
            s.search.providers = []
            s.memory.enabled = False
            s.verify.enabled = False  # 固定劇本不含額外的查證員呼叫
            s.browser.headless = True
            s.agent.workspace = str(root / f"case-{index}" / "workspace")
            s.agent.runs_dir = str(root / f"case-{index}" / "runs")
            library = PlaybookLibrary.load([s.path(d) for d in s.playbook_dirs])
            library.playbooks += [Playbook.from_toml(path) for path in playbook_files]
            script = json.loads(json.dumps(case["script"], ensure_ascii=False).replace("{fixture}", fixture))
            model = _ScriptedModel(script)
            result = asyncio.run(Kernel(s, model, SkillSet(builtin_skills()), library=library).run(case["task"]))
            outcomes = [event.data for event in Journal.load(Path(result.run_dir)) if event.kind == "outcome"]
            required = [item["tool"] for item in script if "tool" in item]
            actions_ok = ([row["name"] for row in outcomes] == required and
                          all(row["ok"] and row["verified"] is not False for row in outcomes))
            rows.append({"success": result.status == "success" and actions_ok and
                         all(word in result.answer for word in case["expect"]),
                         "steps": result.steps, "tokens": result.usage["total_tokens"]})
        successes = sum(row["success"] for row in rows)
        tokens = sum(row["tokens"] for row in rows)
        return {"mode": "scripted", "cases": len(rows), "successes": successes,
                "tokens": tokens, "steps": sum(row["steps"] for row in rows),
                "score": round(successes / len(rows) - 0.1 * tokens / data["cost_budget"], 4)}
