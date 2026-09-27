"""測試公共設施：本地 HTTP 夾具伺服器、隔離的配置、可編排的假模型 ScriptedLLM。"""

from __future__ import annotations

import functools
import http.server
import importlib.util
import itertools
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingxi.llm.messages import Reply, ToolCall, Usage  # noqa: E402
from lingxi.settings import Settings  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _browser_available() -> bool:
    if importlib.util.find_spec("playwright") is None:
        return False
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            pw.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


BROWSER_OK = _browser_available()
needs_browser = pytest.mark.skipif(not BROWSER_OK, reason="本機沒有可用的 Chromium（python -m playwright install chromium）")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def build_site_dir(target: Path) -> Path:
    """準備夾具網站：繁體原檔 flight.html ＋ 以 unify() 即時產生的簡體版 flight-cn.html（模擬大陸網站）。"""
    from lingxi.hanzi import unify

    target.mkdir(parents=True, exist_ok=True)
    html = (FIXTURES / "flight.html").read_text(encoding="utf-8")
    (target / "flight.html").write_text(html, encoding="utf-8")
    cn = unify(html).replace('lang="zh-Hant"', 'lang="zh-CN"')
    (target / "flight-cn.html").write_text(cn, encoding="utf-8")
    return target


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    root = build_site_dir(tmp_path_factory.mktemp("site"))
    handler = functools.partial(_Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    # 測試絕不能用到真實金鑰：遮蔽所有可能的 API Key 來源，避免產生任何付費呼叫
    monkeypatch.delenv("LINGXI_API_KEY", raising=False)
    s = Settings()
    s.llm.api_key_env = "LINGXI_TEST_NO_SUCH_KEY"
    s.llm.vision.api_key_env = "LINGXI_TEST_NO_SUCH_KEY"
    s.daytona.api_key_env = "LINGXI_TEST_NO_SUCH_KEY"
    s.browser.headless = True
    s.verify.enabled = False  # 反覆查證會多一次模型呼叫；專門的測試再開啟
    s.agent.workspace = str(tmp_path / "workspace")
    s.agent.runs_dir = str(tmp_path / "runs")
    s.playbook_dirs = [str(ROOT / "playbooks")]
    s.search.providers = []
    s.llm.vision.enabled = False
    return s


_ids = itertools.count(1)


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=json.dumps(args, ensure_ascii=False))


def reply(*calls: ToolCall, content: str = "") -> Reply:
    return Reply(content=content, tool_calls=list(calls))


class ScriptedLLM:
    """按劇本回復的假模型。劇本項可以是 Reply，也可以是 (messages, tools) -> Reply 的函式。"""

    def __init__(self, script, fallback=None):
        self.script = list(script)
        self.fallback = fallback
        self.usage = Usage()
        self.requests: list[tuple[list, list | None]] = []

    async def chat(self, messages, tools=None, tool_choice="auto", **kw) -> Reply:
        self.requests.append((messages, tools))
        self.usage.add(100, 20)
        if self.script:
            item = self.script.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            item = reply(call("finish", answer="劇本已結束", status="partial"))
        return item(messages, tools) if callable(item) else item
