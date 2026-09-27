"""示例：單獨呼叫視覺神諭，在一張靜態截圖上定位"1月30日"（對應課程 CASE-gui-plus）。

    python examples/vision_grounding.py
    python examples/vision_grounding.py path/to/screenshot.png "幫我點選搜尋按鈕"

需要 DASHSCOPE_API_KEY（或 config/lingxi.toml 中的 [llm.vision]）。
與課程腳本的區別：輸出統一解析成座標（相容多種不規範 JSON），並可在截圖上畫出落點方便核對。
"""

import asyncio
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingxi.senses.vision import VisionOracle  # noqa: E402
from lingxi.settings import get_settings  # noqa: E402


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


async def main() -> None:
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "assets" / "ctrip_calendar.png"
    target = sys.argv[2] if len(sys.argv) > 2 else "日曆中的 1月30日"
    png = image.read_bytes()
    size = png_size(png)

    oracle = VisionOracle(get_settings().llm.vision)
    point = await oracle.locate(png, f"請找到需要點選的介面元素：{target}", size)
    if point is None:
        print("視覺模型沒有找到目標。")
        return
    print(f"截圖尺寸 {size[0]}×{size[1]}，目標「{target}」→ 座標 ({point.x:.0f}, {point.y:.0f})")
    print(f"模型思考：{point.thought}")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("（安裝 pillow 後可以把落點畫在截圖上）")
        return
    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    x, y = point.x, point.y
    draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline="red", width=3)
    out = image.with_name(image.stem + "_marked.png")
    img.save(out)
    print(f"已標註：{out}")


if __name__ == "__main__":
    asyncio.run(main())
