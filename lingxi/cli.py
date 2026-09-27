"""命令列入口。

  lingxi run "查詢 6月26日 從上海到北京的機票"   執行任務
  lingxi serve                                  啟動 Web 介面
  lingxi playbooks "任務"                        試跑知識層：看時間錨定 / 手冊匹配 / 任務模式（不呼叫模型）
  lingxi replay latest                          把最近一次執行渲染成 report.html
  lingxi runs                                   列出最近的執行
  lingxi doctor                                 檢查配置、金鑰、瀏覽器
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import webbrowser
from pathlib import Path

from lingxi.settings import load_settings


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _settings(args):
    s = load_settings(args.config)
    if getattr(args, "headless", False):
        s.browser.headless = True
    if getattr(args, "debug", False):
        s.agent.debug_snapshots = True
    return s


# ---------------- run ----------------
def cmd_run(args) -> int:
    from lingxi.agent import LingXi
    from lingxi.kernel.context import ConsoleHuman

    task = " ".join(args.task).strip() or input("請輸入任務：").strip()
    if not task:
        print("任務為空。")
        return 1
    result = asyncio.run(LingXi(_settings(args), human=ConsoleHuman()).run(task))
    if result.run_dir:
        print(f"\n黑匣子：{result.run_dir}\n復盤報告：lingxi replay \"{result.run_dir}\"")
    return 0 if result.status != "failed" else 2


# ---------------- playbooks（試跑）----------------
def cmd_playbooks(args) -> int:
    from lingxi.kernel.preflight import preflight
    from lingxi.knowledge.playbook import PlaybookLibrary

    s = _settings(args)
    library = PlaybookLibrary.load([s.path(d) for d in s.playbook_dirs])
    task = " ".join(args.task).strip()
    if not task:
        print(f"已載入 {len(library.playbooks)} 本手冊：")
        for info in library.describe():
            print(f"  · {info['id']:<20} {info['title']}  觸發詞：{'、'.join(info['triggers'][:6])}")
        return 0
    pf = asyncio.run(preflight(task, library))
    p = pf.profile
    print(f"任務    ：{task}")
    print(f"錨定後  ：{pf.briefed}")
    print(f"任務模式：{p.name}（{p.title}）· {pf.reason} · 基礎預算 {p.base_budget} 步 · 日期偏好 {pf.preference}")
    if not pf.playbooks:
        print("手冊    ：未命中")
    for pb in pf.playbooks:
        print(f"\n{pb.render()}")
    return 0


# ---------------- replay / runs ----------------
def _resolve_run(s, ref: str) -> Path | None:
    if ref == "latest":
        runs = sorted((p for p in s.runs_dir.iterdir() if (p / "events.jsonl").exists()), key=lambda p: p.name)
        return runs[-1] if runs else None
    p = Path(ref)
    if not p.is_absolute() and not p.exists():
        p = s.runs_dir / ref
    return p if (p / "events.jsonl").exists() else None


def cmd_replay(args) -> int:
    from lingxi.journal.report import write_report

    s = _settings(args)
    run_dir = _resolve_run(s, args.run)
    if not run_dir:
        print(f"找不到執行記錄：{args.run}")
        return 1
    out = write_report(run_dir)
    print(f"報告已生成：{out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


def cmd_runs(args) -> int:
    from lingxi.journal.recorder import Journal

    s = _settings(args)
    runs = sorted((p for p in s.runs_dir.iterdir() if (p / "events.jsonl").exists()), key=lambda p: p.name)[-args.limit:]
    if not runs:
        print("還沒有執行記錄。")
    for run_dir in runs:
        events = Journal.load(run_dir)
        finish = next((e for e in reversed(events) if e.kind == "run.finish"), None)
        status = finish.data.get("status") if finish else "未結束"
        print(f"  {run_dir.name:<48} {status:<8} {finish.step if finish else '-'} 步")
    return 0


# ---------------- serve ----------------
def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("需要安裝 Web 依賴：pip install -e \".[web]\"")
        return 1
    from lingxi.web.server import create_app

    app = create_app(_settings(args))
    print(f"靈犀 Web 介面：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------- doctor ----------------
def cmd_doctor(args) -> int:
    s = _settings(args)
    ok = True

    def line(flag: bool, text: str) -> None:
        nonlocal ok
        ok = ok and flag
        print(f"  {'✔' if flag else '✘'} {text}")

    print("靈犀環境檢查")
    print(f"  · 配置來源：{s.source}")
    key = s.llm.resolve_api_key()
    line(bool(key), f"主模型 {s.llm.model} @ {s.llm.base_url} · API Key {'已設定（' + key[:4] + '…）' if key else '未設定'}")
    vkey = s.llm.vision.resolve_api_key()
    print(f"  {'✔' if vkey else '·'} 視覺模型 {s.llm.vision.model}：{'可用' if vkey and s.llm.vision.enabled else '未啟用（DOM 定位失敗時將無法視覺兜底）'}")
    for mod, label in (("playwright", "Playwright"), ("fastapi", "Web 介面"), ("mcp", "MCP"), ("daytona", "Daytona"), ("ddgs", "DDGS 搜尋")):
        found = importlib.util.find_spec(mod) is not None
        if mod == "playwright":
            line(found, f"{label}{'' if found else '（pip install playwright）'}")
        else:
            print(f"  {'✔' if found else '·'} {label}{'' if found else '（可選，未安裝）'}")

    async def _browser_check() -> str:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            b = await pw.chromium.launch(headless=True)
            v = b.version
            await b.close()
            return v

    if importlib.util.find_spec("playwright"):
        try:
            line(True, f"Chromium {asyncio.run(_browser_check())} 可以啟動")
        except Exception as exc:
            line(False, f"Chromium 無法啟動：{exc}（執行 python -m playwright install chromium）")
    print(f"  · 工作區：{s.workspace_dir}\n  · 黑匣子：{s.runs_dir}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="lingxi", description="靈犀 LingXi —— 證據驅動的通用瀏覽器智慧代理")
    parser.add_argument("--config", help="配置檔案路徑（預設 config/lingxi.toml）")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("run", help="執行一個任務")
    p.add_argument("task", nargs="*")
    p.add_argument("--headless", action="store_true", help="無頭模式執行瀏覽器")
    p.add_argument("--debug", action="store_true", help="每一步都儲存截圖與 HTML")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="啟動 Web 介面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("playbooks", help="列出手冊，或對一個任務試跑知識層")
    p.add_argument("task", nargs="*")
    p.set_defaults(func=cmd_playbooks)

    p = sub.add_parser("replay", help="把一次執行渲染成 HTML 報告")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--open", action="store_true", help="生成後用瀏覽器開啟")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("runs", help="列出最近的執行")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("doctor", help="檢查環境")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
