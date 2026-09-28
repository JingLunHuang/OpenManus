"""OpenAI 相容的模型客戶端：帶指數退避重試與 token 計量。

所有呼叫都走 chat()，核心拿到的永遠是 lingxi.llm.messages.Reply，
測試裡用 ScriptedLLM 替換本類即可離線跑通整條鏈路。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Protocol

from lingxi.llm.kvcache import read_cache_usage
from lingxi.llm.messages import Message, Reply, ToolCall, Usage
from lingxi.settings import ModelSettings

log = logging.getLogger("lingxi.llm")

_RETRYABLE = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
)


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    """核心依賴的最小介面，真實客戶端與測試替身都實現它。"""

    usage: Usage

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Reply: ...


class LLMClient:
    def __init__(self, settings: ModelSettings, max_retries: int = 4):
        self.settings = settings
        self.max_retries = max_retries
        self.usage = Usage()
        self._client = None

    @property
    def model(self) -> str:
        return self.settings.model

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            key = self.settings.resolve_api_key()
            if not key:
                raise LLMError(
                    f"未找到 API Key：請設定環境變數 {self.settings.api_key_env} 或 LINGXI_API_KEY，"
                    "或在 config/lingxi.toml 的 [llm] 中填寫 api_key"
                )
            self._client = AsyncOpenAI(
                api_key=key, base_url=self.settings.base_url, timeout=self.settings.timeout
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Reply:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        body = {**self.settings.extra_body, **(extra_body or {})}
        if body:
            kwargs["extra_body"] = body

        response = await self._with_retry(lambda: client.chat.completions.create(**kwargs))
        if not response.choices:
            raise LLMError("模型返回了空的 choices")
        choice = response.choices[0]
        msg = choice.message
        calls = [
            ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=tc.function.arguments or "{}")
            for i, tc in enumerate(msg.tool_calls or [])
            if getattr(tc, "function", None)
        ]
        pt = getattr(response.usage, "prompt_tokens", 0) or 0
        ct = getattr(response.usage, "completion_tokens", 0) or 0
        cached, _ = read_cache_usage(response.usage)
        self.usage.add(pt, ct, cached)
        return Reply(
            content=msg.content or "",
            tool_calls=calls,
            prompt_tokens=pt,
            completion_tokens=ct,
            finish_reason=choice.finish_reason,
            cached_tokens=cached,
        )

    async def ask(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        msgs = ([Message.system(system)] if system else []) + [Message.user(prompt)]
        return (await self.chat(msgs, **kw)).content

    async def _with_retry(self, make_call):
        delay = 1.5
        for attempt in range(1, self.max_retries + 1):
            try:
                return await make_call()
            except Exception as exc:  # noqa: BLE001 —— 按類名判斷，避免強依賴 openai 版本
                retryable = type(exc).__name__ in _RETRYABLE
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                wait = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                log.warning("模型呼叫失敗（%s），%.1fs 後第 %d 次重試", type(exc).__name__, wait, attempt)
                await asyncio.sleep(wait)
        raise LLMError("unreachable")
