"""KV 快取加速：讓推理引擎能重用上一次請求的前綴。

Transformer 推理時，每個 token 的 Key/Value 張量都會被快取（KV cache）。現今主流引擎都支援「前綴快取」：
  DashScope 隱式快取（命中價 20%）/ 顯式快取 cache_control（命中價 10%，5 分鐘有效）、
  DeepSeek 硬碟快取、OpenAI 自動快取、vLLM Automatic Prefix Caching、SGLang RadixAttention、llama.cpp cache_prompt。
它們有同一個前提：這次請求的開頭必須和之前的請求「逐字相同」，只要中間某一段變了，後面全部重算。

代理框架決定了前綴穩不穩。靈犀的每一步請求長這樣：

    [工具定義] [系統提示] [歷史輪次 T1 … Tn] [本步簡報]
     └────────── 可以沿用的前綴 ──────────┘  └ 每步都新 ┘

破壞前綴的三個常見原因，以及靈犀的對策（見 kernel/loop.py、kernel/memory.py）：
  1. 逐步摺疊舊觀察：每一步都把第 n-6 輪改寫成摘要 → 從那一輪起全部重算。
     對策：分段摺疊（fold_block），累積滿 N 輪才一次摺疊，前綴在 N 步內只增不改。
  2. 最後一步縮減工具清單：工具定義在最前面，一變就全部失效。
     對策：工具清單不變，改用 tool_choice 指定 finish。
  3. 動態內容寫進系統提示（時間、頁面、白板）。
     對策：系統提示整次執行不變；會變的內容一律放在最後的本步簡報。

本模組提供：
  - PrefixMeter：離線量測「這次請求有多少比例沿用了上一次的前綴」，不依賴廠商回報，假模型測試也能量；
  - read_cache_usage()：從各家 usage 讀出實際命中的 token 數。
顯式快取標記由 Message.cache（messages.py）輸出成 DashScope / Anthropic 相容的 cache_control。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _segments(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[str]:
    segs = [json.dumps(tools or [], ensure_ascii=False, sort_keys=True)]
    segs += [json.dumps(m, ensure_ascii=False, sort_keys=True) for m in messages]
    return segs


@dataclass
class PrefixMeter:
    """以「訊息」為單位比對前綴（和 DeepSeek 快取單元、DashScope 區塊回溯的粒度一致，屬於保守估計）。"""

    history: list[float] = field(default_factory=list)
    reused_chars: int = 0
    total_chars: int = 0
    _prev: list[str] = field(default_factory=list)

    def observe(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> float:
        segs = _segments(messages, tools)
        reused = 0
        for a, b in zip(segs, self._prev):
            if a != b:
                break
            reused += len(a)
        total = sum(len(s) for s in segs)
        ratio = reused / total if total else 0.0
        if self._prev:  # 第一個請求沒有可比對的前綴，不計入
            self.history.append(round(ratio, 4))
            self.reused_chars += reused
            self.total_chars += total
        self._prev = segs
        return ratio

    @property
    def reuse(self) -> float:
        return round(self.reused_chars / self.total_chars, 4) if self.total_chars else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"prefix_reuse": self.reuse, "requests": len(self.history) + (1 if self._prev else 0),
                "per_step": self.history}


def replay(tool_outputs: list[int], fold_after: int, fold_block: int, max_obs: int, *, tools_chars: int = 6000,
           system_chars: int = 3500, brief_chars: int = 2600, cache_price: float = 0.2) -> dict[str, float]:
    """用一條「每步工具輸出字數」的軌跡重放 RollingMemory，回傳前綴重用率、平均請求體積、平均有效計費量。
    有效計費量 = 未命中部分 ＋ cache_price × 命中部分（DashScope 隱式快取命中價為 20%）。
    受保護評測（固定種子軌跡）、改進器的開發評測（真實執行軌跡）與 lingxi bench 共用這個演算法。"""
    from lingxi.kernel.memory import RollingMemory
    from lingxi.llm.messages import Message, ToolCall

    tools = [{"type": "function", "function": {"name": "t", "description": "x" * tools_chars}}]
    system = Message.system("S" * system_chars)
    memory = RollingMemory(fold_after, max_obs, fold_block=fold_block)
    meter = PrefixMeter()
    sizes, billed = [], []
    for step, length in enumerate(tool_outputs, 1):
        payload = [m.to_openai() for m in [system, *memory.messages(),
                                            Message.user(f"【第 {step} 步】" + "B" * brief_chars)]]
        size = sum(len(json.dumps(m, ensure_ascii=False)) for m in payload) + len(json.dumps(tools))
        reuse = meter.observe(payload, tools)
        sizes.append(size)
        billed.append(size * (1 - reuse) + cache_price * size * reuse)
        call = ToolCall(id=f"c{step}", name="web_read", arguments="{}")
        memory.add_turn(Message.assistant(f"第 {step} 步的想法", [call]),
                        [Message.tool(call, f"觀察{step}：" + "o" * length, digest=f"✔ 摘要 {step}")])
    n = max(len(sizes), 1)
    return {"reuse": meter.reuse, "avg_chars": sum(sizes) / n, "billed_chars": sum(billed) / n}


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def read_cache_usage(usage: Any) -> tuple[int, int]:
    """回傳 (命中快取的 token 數, 新建快取的 token 數)。各家欄位：
    OpenAI / DashScope：usage.prompt_tokens_details.cached_tokens、…cache_creation_input_tokens
    DeepSeek：usage.prompt_cache_hit_tokens
    Anthropic 相容：usage.cache_read_input_tokens、usage.cache_creation_input_tokens
    """
    details = _get(usage, "prompt_tokens_details")
    cached = (_get(details, "cached_tokens") or _get(usage, "prompt_cache_hit_tokens")
              or _get(usage, "cache_read_input_tokens") or 0)
    created = _get(details, "cache_creation_input_tokens") or _get(usage, "cache_creation_input_tokens") or 0
    return int(cached or 0), int(created or 0)
