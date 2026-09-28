"""與模型互動的最小訊息模型。

刻意不復用 OpenAI SDK 的型別：核心只依賴這裡的 dataclass，
換成別的廠商 SDK 時只需要改 to_openai() 這一處。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"

    def args(self) -> dict[str, Any]:
        """解析參數；模型偶爾會給出空串或非法 JSON，這裡寬容處理。"""
        if not self.arguments or not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {"__invalid_json__": self.arguments}
        return value if isinstance(value, dict) else {"value": value}

    def fingerprint(self) -> str:
        """動作指紋：工具名 + 規範化參數，用於迴圈檢測。"""
        args = self.args()
        return f"{self.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    images: list[str] = field(default_factory=list)  # base64 PNG/JPEG
    # 被摺疊後保留的一行摘要（RollingMemory 使用）
    digest: str | None = None
    # 顯式快取斷點：輸出成 cache_control（DashScope 顯式快取 / Anthropic 相容端點），見 llm/kvcache.py
    cache: bool = False

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str, images: list[str] | None = None) -> "Message":
        return cls("user", content, images=list(images or []))

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls("assistant", content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, call: ToolCall, content: str, digest: str | None = None) -> "Message":
        return cls("tool", content, tool_call_id=call.id, name=call.name, digest=digest)

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.images and self.role == "user":
            parts: list[dict[str, Any]] = [{"type": "text", "text": self.content}]
            for img in self.images:
                mime = "image/jpeg" if img.startswith("/9j/") else "image/png"
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img}"}})
            msg["content"] = parts
        elif self.cache and self.content:
            msg["content"] = [{"type": "text", "text": self.content, "cache_control": {"type": "ephemeral"}}]
        else:
            msg["content"] = self.content or ""
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments or "{}"},
                }
                for c in self.tool_calls
            ]
        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name
        return msg


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # 輸入裡命中推理端 KV 快取的 token（廠商回報）

    def add(self, prompt: int, completion: int, cached: int = 0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    cached_tokens: int = 0
