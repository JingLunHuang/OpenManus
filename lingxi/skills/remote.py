"""外部能力接入：MCP 伺服器的工具、Daytona 雲沙箱。都是可選依賴，按需載入。"""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack
from typing import Any

from pydantic import BaseModel, Field

from lingxi.settings import DaytonaSettings, McpServer
from lingxi.skills.base import Outcome, Skill
from lingxi.utils import clip


# ======================= MCP =======================
class McpToolSkill(Skill):
    """把一個 MCP 工具包裝成靈犀技能：Schema 直接透傳，參數校驗交給 MCP 伺服器。"""

    Params = BaseModel  # 不使用 pydantic 校驗

    def __init__(self, server_id: str, tool: Any, session: Any):
        self.server_id = server_id
        self.tool = tool
        self.session = session
        self.name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp_{server_id}_{tool.name}")[:64]
        self.description = f"[MCP·{server_id}] {tool.description or tool.name}"

    def tool_schema(self) -> dict[str, Any]:
        schema = getattr(self.tool, "inputSchema", None) or {"type": "object", "properties": {}}
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": schema}}

    async def invoke(self, ctx, raw_args: dict[str, Any]) -> Outcome:
        try:
            result = await self.session.call_tool(self.tool.name, raw_args)
        except Exception as exc:
            return Outcome.fail(f"MCP 呼叫失敗：{clip(str(exc), 200)}")
        texts = [getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None)]
        body = "\n".join(texts) or "（無文字輸出）"
        if getattr(result, "isError", False):
            return Outcome.fail(f"{self.tool.name} 返回錯誤", clip(body, 4000))
        return Outcome(ok=True, summary=clip(body.splitlines()[0], 120), detail=clip(body, 6000))

    async def run(self, ctx, params):  # pragma: no cover —— invoke 已覆蓋
        raise NotImplementedError


class McpHub:
    def __init__(self, servers: dict[str, McpServer]):
        self.servers = servers
        self._stack = AsyncExitStack()

    async def connect(self) -> tuple[list[Skill], list[str]]:
        skills: list[Skill] = []
        errors: list[str] = []
        if not self.servers:
            return skills, errors
        try:
            from mcp import ClientSession
        except ImportError:
            return skills, ["未安裝 mcp：pip install 'lingxi[mcp]'"]
        for server_id, cfg in self.servers.items():
            try:
                if cfg.transport == "stdio":
                    from mcp import StdioServerParameters
                    from mcp.client.stdio import stdio_client

                    params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
                    streams = await self._stack.enter_async_context(stdio_client(params))
                elif cfg.transport == "sse":
                    from mcp.client.sse import sse_client

                    streams = await self._stack.enter_async_context(sse_client(cfg.url))
                else:
                    from mcp.client.streamable_http import streamable_http_client

                    streams = await self._stack.enter_async_context(streamable_http_client(cfg.url))
                session = await self._stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await asyncio.wait_for(session.initialize(), timeout=30)
                listing = await session.list_tools()
                skills.extend(McpToolSkill(server_id, t, session) for t in listing.tools)
            except Exception as exc:
                errors.append(f"{server_id}: {clip(str(exc), 160)}")
        return skills, errors

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            pass


# ======================= Daytona 雲沙箱 =======================
class SandboxParams(BaseModel):
    code: str = Field(description="要在雲端隔離沙箱中執行的 Python 程式碼")


class DaytonaRun(Skill):
    name = "sandbox_run"
    description = "在 Daytona 雲端隔離沙箱裡執行 Python 程式碼（不影響本機）。適合執行不可信或有副作用的程式碼。"
    Params = SandboxParams

    def __init__(self, settings: DaytonaSettings):
        self.settings = settings
        self._client = None
        self._sandbox = None

    async def _ensure(self):
        if self._sandbox is None:
            from daytona import Daytona, DaytonaConfig  # 可選依賴

            key = self.settings.resolve_api_key()
            if not key:
                raise RuntimeError(f"缺少 Daytona API Key（環境變數 {self.settings.api_key_env}）")
            self._client = Daytona(DaytonaConfig(api_key=key, target=self.settings.target))
            self._sandbox = await asyncio.to_thread(self._client.create)
        return self._sandbox

    async def run(self, ctx, p: SandboxParams) -> Outcome:
        try:
            sandbox = await self._ensure()
            response = await asyncio.to_thread(sandbox.process.code_run, p.code)
        except Exception as exc:
            return Outcome.fail(f"沙箱執行失敗：{clip(str(exc), 200)}")
        text = str(getattr(response, "result", ""))
        if getattr(response, "exit_code", 0) != 0:
            return Outcome.fail(f"沙箱退出碼 {response.exit_code}", clip(text, 4000))
        return Outcome(ok=True, summary=clip(text.strip().splitlines()[-1] if text.strip() else "執行完成", 120),
                       detail=clip(text, 6000), verified=True)

    async def close(self) -> None:
        if self._client and self._sandbox:
            remove = getattr(self._client, "delete", None) or getattr(self._client, "remove", None)
            if remove:
                try:
                    await asyncio.to_thread(remove, self._sandbox)
                except Exception:
                    pass
            self._sandbox = None
