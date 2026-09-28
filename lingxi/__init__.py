"""靈犀 LingXi —— 證據驅動的通用瀏覽器智慧代理框架。"""

__version__ = "0.2.0"

from lingxi.agent import LingXi  # noqa: E402
from lingxi.kernel.loop import RunResult  # noqa: E402
from lingxi.settings import Settings, load_settings  # noqa: E402

__all__ = ["LingXi", "RunResult", "Settings", "load_settings", "__version__"]
