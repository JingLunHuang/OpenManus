"""零散但被多處複用的小工具。"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse


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


def site_key(url: str) -> str:
    """把 flights.ctrip.com / www.ctrip.com 歸到同一站點 ctrip.com。"""
    host = (urlparse(url).hostname or "").lower()
    if not host or re.fullmatch(r"[\d.]+|\[?[0-9a-f:]+\]?", host) or "." not in host:
        return host  # IP 地址、localhost 保持原樣
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("com", "net", "org", "gov", "edu") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])  # xxx.com.cn
    return ".".join(parts[-2:]) if len(parts) >= 2 else host
