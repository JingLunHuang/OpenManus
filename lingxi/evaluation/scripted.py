"""離線實驗用的固定劇本模型，不讀取 API 金鑰。"""

from __future__ import annotations

import importlib.util
import itertools
import json

from lingxi.llm.messages import Reply, ToolCall, Usage


def browser_available() -> bool:
    """確認 Playwright 與本機 Chromium 都可用。"""
    if importlib.util.find_spec("playwright") is None:
        return False
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            playwright.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


_ids = itertools.count(1)


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=json.dumps(args, ensure_ascii=False))


def reply(*calls: ToolCall, content: str = "") -> Reply:
    return Reply(content=content, tool_calls=list(calls))


class ScriptedLLM:
    """按固定劇本回覆；劇本項可為 Reply 或 (messages, tools) -> Reply。"""

    def __init__(self, script, fallback=None):
        self.script = list(script)
        self.fallback = fallback
        self.usage = Usage()
        self.requests: list[tuple[list, list | None, object]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", **kw) -> Reply:
        self.requests.append((messages, tools, tool_choice))
        self.usage.add(100, 20)
        if isinstance(tool_choice, dict) and tools:
            forced = tool_choice.get("function", {}).get("name")
            tools = [tool for tool in tools if tool["function"]["name"] == forced]
        if self.script:
            item = self.script.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            item = reply(call("finish", answer="劇本已結束", status="partial"))
        return item(messages, tools) if callable(item) else item
