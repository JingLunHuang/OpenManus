"""配置層：TOML 檔案 + 環境變數，統一成帶型別的 Settings 物件。

查詢順序：
1. 環境變數 LINGXI_CONFIG 指向的檔案
2. <專案根>/config/lingxi.toml
3. 全部使用預設值（API Key 從環境變數讀取）

金鑰永遠優先從環境變數讀取，配置檔案裡的 api_key 只是兜底，
這樣倉庫裡的 lingxi.example.toml 可以放心提交。
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelSettings(BaseModel):
    """一個 OpenAI 相容端點上的模型。DashScope / DeepSeek / Ollama / OpenAI 都走這一套。"""

    model: str = "qwen-max"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0
    # 主模型是否能直接看圖（如 qwen3.x-plus）。為 True 時，最新截圖會隨簡報一起發給主模型
    supports_vision: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        return os.getenv("LINGXI_API_KEY") or os.getenv(self.api_key_env) or self.api_key


class VisionSettings(ModelSettings):
    """視覺神諭：只在 DOM 定位失敗時才被呼叫。"""

    model: str = "gui-plus"  # 負責"指出座標"
    qa_model: str = "qwen-vl-plus"  # 負責"看圖回答問題"（web_look）
    max_tokens: int = 1024
    # 模型返回座標的座標系：pixel=截影象素；norm1000=0~1000 歸一化（部分 Qwen-VL 版本）
    coord_space: str = "pixel"
    enabled: bool = True


class LLMSettings(ModelSettings):
    vision: VisionSettings = Field(default_factory=VisionSettings)


class BrowserSettings(BaseModel):
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    locale: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    user_agent: str = ""
    # 接管已登入的真實 Chrome（chrome --remote-debugging-port=9222）
    cdp_url: str = ""
    executable_path: str = ""
    # 降低自動化特徵：隱藏 navigator.webdriver 等
    stealth: bool = True
    navigation_timeout_ms: int = 30000
    action_timeout_ms: int = 8000
    # 簡報裡正文摘要的最大字數
    digest_chars: int = 1800
    # 每類元素最多列出多少個
    max_elements_per_kind: int = 40


class AgentSettings(BaseModel):
    language: str = "zh"
    workspace: str = "workspace"
    runs_dir: str = "runs"
    # 每步都儲存 HTML + 截圖到黑匣子（排障時開啟）
    debug_snapshots: bool = False
    # 預算上限 = 模式基礎步數 × 該係數
    hard_cap_factor: float = 2.0
    # 超過多少輪的舊觀察會被摺疊成一行摘要
    # 代理寫進工作區的文字檔（.md / .txt / .html / .csv / .json）一律存成繁體
    traditional_output: bool = True
    fold_after_turns: int = 6
    max_observation_chars: int = 6000


class SearchSettings(BaseModel):
    # 依次嘗試：browser:bing / browser:baidu / ddgs / llm
    providers: list[str] = Field(default_factory=lambda: ["browser:bing", "browser:baidu"])
    max_results: int = 6


class IntentSettings(BaseModel):
    # 規則打分拿不準時，是否再問一次 LLM
    llm_fallback: bool = False


class VerifySettings(BaseModel):
    """反覆查證：finish 之前先核對答覆中的事實陳述是否有證據。"""

    enabled: bool = True
    max_rounds: int = 2  # 最多退回幾輪
    profiles: list[str] = Field(default_factory=lambda: ["research", "web_query"])


class McpServer(BaseModel):
    transport: str = "stdio"  # stdio | sse
    command: str = ""
    args: list[str] = Field(default_factory=list)
    url: str = ""
    env: dict[str, str] = Field(default_factory=dict)


class DaytonaSettings(BaseModel):
    api_key: str = ""
    api_key_env: str = "DAYTONA_API_KEY"
    target: str = "us"

    def resolve_api_key(self) -> str:
        return os.getenv(self.api_key_env) or self.api_key


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    intent: IntentSettings = Field(default_factory=IntentSettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    playbook_dirs: list[str] = Field(default_factory=lambda: ["playbooks"])
    mcp_servers: dict[str, McpServer] = Field(default_factory=dict)
    daytona: DaytonaSettings = Field(default_factory=DaytonaSettings)
    source: str = "defaults"

    # ---- 路徑解析：相對路徑一律相對專案根 ----
    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def workspace_dir(self) -> Path:
        d = self.path(self.agent.workspace)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def runs_dir(self) -> Path:
        d = self.path(self.agent.runs_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d


def _flatten_toml(raw: dict[str, Any]) -> dict[str, Any]:
    """把 TOML 的表結構對映到 Settings 欄位（[playbooks] dirs / [mcp.servers.x]）。"""
    data = dict(raw)
    if "playbooks" in data:
        data["playbook_dirs"] = data.pop("playbooks").get("dirs", ["playbooks"])
    if "mcp" in data:
        data["mcp_servers"] = data.pop("mcp").get("servers", {})
    return data


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """極簡 .env 讀取：KEY=VALUE，已存在的環境變數不覆蓋。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def load_settings(path: str | Path | None = None) -> Settings:
    load_dotenv()
    candidates = [path, os.getenv("LINGXI_CONFIG"), PROJECT_ROOT / "config" / "lingxi.toml"]
    for c in candidates:
        if c and Path(c).is_file():
            with open(c, "rb") as f:
                raw = tomllib.load(f)
            settings = Settings.model_validate(_flatten_toml(raw))
            settings.source = str(c)
            return settings
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
