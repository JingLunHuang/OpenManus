"""視覺神諭（Vision Oracle）：給截圖和一句指令，返回螢幕座標；或看圖回答問題。

它在靈犀裡是"最後一道證據"，只在 DOM 定位失敗時才呼叫，並且結果要經過
grounding.vision_consensus 的 DOM 交叉驗證。

GUI 類模型（gui-plus / Qwen-VL）的輸出格式並不穩定，課程裡就遇到過
{"x": 139, 675}、{"x": [139, 675]}、缺右括號、包在 ```json 裡等情況。
parse_vision_reply() 把這些都歸一成 VisionPoint。
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

from lingxi.llm.client import LLMClient
from lingxi.llm.messages import Message
from lingxi.settings import VisionSettings
from lingxi.utils import extract_json

_LOCATE_SYSTEM = """你是介面定位助手。給你一張網頁截圖和一個目標描述，找出目標元素的中心點座標。
只輸出一個 JSON 物件，不要輸出其他文字：
{"thought": "一句話說明你看到了什麼", "action": "CLICK", "parameters": {"x": 整數, "y": 整數}}
座標以截圖左上角為原點、單位為畫素。只能基於截圖裡真實可見的內容作答。
如果截圖中確實沒有該目標，輸出：{"thought": "...", "action": "FAIL", "parameters": {"reason": "原因"}}"""


@dataclass
class VisionPoint:
    x: float
    y: float
    thought: str = ""
    raw: str = ""


def _extract_json(text: str) -> dict | None:
    # 修復 {"x": 139, 675} 與 {"x": [139, 675]} 兩種常見錯誤
    text = re.sub(r'"x"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*(?=[,}])', r'"x": \1, "y": \2', text)
    text = re.sub(r'"x"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', r'"x": \1, "y": \2', text)
    value = extract_json(text[text.find("{"):] if "{" in text else text)
    return value if isinstance(value, dict) else None


def _pair(value) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        if len(nums) == 2:
            return nums[0], nums[1]
        if len(nums) == 4:  # bbox → 中心
            return (nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2
    return None


def parse_vision_reply(text: str, size: tuple[int, int], coord_space: str = "pixel") -> VisionPoint | None:
    width, height = size
    thought, xy = "", None
    data = _extract_json(text)
    if data:
        thought = str(data.get("thought", ""))
        if str(data.get("action", "")).strip().upper() in ("FAIL", "FAILE"):
            return None
        params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
        for source in (params, data):
            if "x" in source and "y" in source:
                try:
                    xy = float(source["x"]), float(source["y"])
                except (TypeError, ValueError):
                    xy = None
                break
            for key in ("point", "coordinate", "coordinates", "position", "bbox", "box"):
                if key in source and (xy := _pair(source[key])):
                    break
            if xy:
                break
    if xy is None:
        for pattern in (r'"x"\s*:\s*(\d+(?:\.\d+)?)[\s\S]*?"y"\s*:\s*(\d+(?:\.\d+)?)',
                        r"<point>\s*(\d+(?:\.\d+)?)[\s,]+(\d+(?:\.\d+)?)\s*</point>",
                        r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
                        r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"):
            m = re.search(pattern, text)
            if m:
                xy = float(m.group(1)), float(m.group(2))
                break
    if xy is None:
        return None
    x, y = xy
    if coord_space == "norm1000":
        x, y = x / 1000 * width, y / 1000 * height
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return VisionPoint(x, y, thought, text)


class VisionOracle:
    def __init__(self, settings: VisionSettings, client: LLMClient | None = None):
        self.settings = settings
        self.client = client or LLMClient(settings)

    @property
    def usage(self):
        return self.client.usage

    async def locate(self, png: bytes, instruction: str, size: tuple[int, int]) -> VisionPoint | None:
        image = base64.b64encode(png).decode()
        reply = await self.client.chat(
            [Message.system(_LOCATE_SYSTEM), Message.user(instruction, images=[image])],
            model=self.settings.model,
        )
        return parse_vision_reply(reply.content, size, self.settings.coord_space)

    async def ask(self, png: bytes, question: str) -> str:
        image = base64.b64encode(png).decode()
        reply = await self.client.chat(
            [Message.system("你是細緻的網頁截圖觀察員，只根據截圖中可見的內容用中文簡潔回答。"),
             Message.user(question, images=[image])],
            model=self.settings.qa_model,
        )
        return reply.content.strip()
