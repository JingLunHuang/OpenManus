"""零散但被多處複用的小工具。"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any | None:
    """從模型輸出裡寬容地取出第一個 JSON 物件/陣列：去掉 ``` 圍欄、補齊缺失的右括號。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    body = text[min(starts):]
    opener = body[0]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc, end = 0, False, False, None
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    candidate = body[:end] if end else body + closer * max(depth, 1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
