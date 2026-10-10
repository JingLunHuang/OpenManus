"""W1「有記憶／無記憶」對照實驗執行器（M1 實驗基建）。

一個命令跑完一組任務 × {memory-off, memory-on} 兩組對照，預設完全離線：
模型由 ScriptedLLM（lingxi/evaluation/scripted.py 的假模型）扮演，網頁由 lingxi/evaluation/fixtures
的夾具頁經本地 HTTP 伺服器提供，不需要任何 API key、不產生任何付費呼叫。

    # 離線乾跑（預設）
    python scripts/experiments/run_experiment.py --suite main

    # 冒煙（2 題、不需瀏覽器，測試用）
    python scripts/experiments/run_experiment.py --suite smoke

    # 接真實模型（M2）：沿用 config/lingxi.toml 與環境變數金鑰
    python scripts/experiments/run_experiment.py --suite main --provider dashscope --repeats 3

量測：成功率、步數、模型呼叫數、token（含快取命中）、查證輪數、牆鐘時間。
雙重護欄：每次執行有「呼叫數 / token / 牆鐘秒數」上限（GuardedModel 強制執行，
超限即中止該次執行並記錄）；整個實驗另有「總呼叫數 / 總 token」上限，超限中止
後續所有執行並記錄在 meta.json / summary.json。

產出（experiments/runs/<時間戳>/）：
    meta.json        實驗配置、護欄上限、中止原因
    records.jsonl    每次執行一筆紀錄（含黑匣子 run_dir，可用 lingxi replay 復盤）
    summary.json     兩組彙總與逐題明細
    report.md        對照報告（report.py 產生）
    site/            夾具網站（離線任務的頁面）
    memory-off/      無記憶組的 runs、workspace、evolve 狀態（各自隔離）
    memory-on/       有記憶組的 runs、workspace、memory（記憶圖）、evolve 狀態
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
for _p in (str(ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):  # Windows 主控台預設 cp950：統一用 UTF-8 輸出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from lingxi.evaluation.fixtures import build_site_dir  # noqa: E402
from lingxi.evaluation.scripted import ScriptedLLM, browser_available  # noqa: E402
from lingxi import LingXi, Settings, load_settings  # noqa: E402
from lingxi.journal.recorder import Journal  # noqa: E402
from lingxi.llm.client import LLMClient, LLMError  # noqa: E402
from report import generate_report  # noqa: E402
from tasks import SUITES, Task  # noqa: E402

ARMS = ("off", "on")  # memory-off（不掛 MemoryHook）／ memory-on（掛 MemoryHook + Recall）
BROWSER_OK = browser_available()


# ---------------------------------------------------------------- 預算護欄


class _NoKey:
    """離線模式的設定替身：讓 LingXi 的「有沒有真實模型」判斷得到否定答案。"""

    def resolve_api_key(self) -> str:
        return ""


class GuardedModel(LLMClient):
    """單次執行的預算護欄：包住真實客戶端或 ScriptedLLM，超限即拋 LLMError 中止。

    刻意繼承 LLMClient：LingXi 內部以 isinstance(llm, LLMClient) 判斷「是否接得上
    真實模型」（影響睡眠整理用模型還是規則），包裝後這個判斷仍然正確。
    """

    def __init__(self, inner, *, max_calls: int = 0, max_tokens: int = 0, max_seconds: float = 0.0):
        self.inner = inner
        self.settings = getattr(inner, "settings", None) or _NoKey()
        self.usage = inner.usage  # 與內層共用計量：核心的 _usage() 讀的就是它
        self.max_retries = 0
        self._client = None
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.deadline = time.time() + max_seconds if max_seconds else 0.0
        self.exceeded = ""  # 觸發護欄的原因（未觸發為空串）

    def _check(self) -> None:
        if self.exceeded:
            return
        if self.max_calls and self.usage.calls >= self.max_calls:
            self.exceeded = f"模型呼叫數達到單次執行上限（{self.max_calls} 次）"
        elif self.max_tokens and self.usage.total >= self.max_tokens:
            self.exceeded = f"token 用量達到單次執行上限（{self.max_tokens}）"
        elif self.deadline and time.time() > self.deadline:
            self.exceeded = "超出單次執行牆鐘時間上限"

    async def chat(self, messages, tools=None, tool_choice="auto", **kw):
        self._check()
        if self.exceeded:
            raise LLMError(self.exceeded)
        return await self.inner.chat(messages, tools=tools, tool_choice=tool_choice, **kw)

    async def ask(self, prompt: str, system: str | None = None, **kw) -> str:
        from lingxi.llm.messages import Message

        msgs = ([Message.system(system)] if system else []) + [Message.user(prompt)]
        return (await self.chat(msgs, **kw)).content


@dataclass
class Totals:
    """整個實驗的累計成本（雙重護欄的第二層）。"""

    calls: int = 0
    tokens: int = 0
    runs: int = 0

    def add(self, calls: int, tokens: int) -> None:
        self.calls += calls
        self.tokens += tokens
        self.runs += 1

    def over(self, max_calls: int, max_tokens: int) -> str:
        if max_calls and self.calls >= max_calls:
            return f"實驗總模型呼叫數達到上限（{self.calls}/{max_calls}）"
        if max_tokens and self.tokens >= max_tokens:
            return f"實驗總 token 達到上限（{self.tokens}/{max_tokens}）"
        return ""


# ---------------------------------------------------------------- 夾具網站


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def start_site(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    """把離線夾具頁建成網站並用隨機埠提供。"""
    build_site_dir(root)
    handler = functools.partial(_Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------- 設定與執行


def build_settings(args, arm: str, arm_dir: Path, provider: str) -> Settings:
    """一組（arm）一份隔離設定：runs / workspace / memory / evolve 各自獨立，絕不寫進專案目錄。"""
    if provider == "scripted":
        s = Settings()
        # 離線硬保證：遮蔽所有金鑰來源，任何 accidental 付費呼叫都會直接失敗
        s.llm.api_key = ""
        s.llm.api_key_env = "LINGXI_EXPERIMENT_NO_SUCH_KEY"
        s.llm.vision.enabled = False
        s.llm.vision.api_key_env = "LINGXI_EXPERIMENT_NO_SUCH_KEY"
        s.daytona.api_key_env = "LINGXI_EXPERIMENT_NO_SUCH_KEY"
        s.search.providers = []
        s.verify.enabled = False  # 查證會多一次模型呼叫；離線劇本沒有為它編排回復
        s.retrieval.embedding = "hash"  # 離線向量化（零費用）
        s.playbook_dirs = [str(ROOT / "playbooks")]
    else:
        s = load_settings()  # 真實模型：沿用 config/lingxi.toml 與環境變數
        if args.model:
            s.llm.model = args.model
        if args.base_url:
            s.llm.base_url = args.base_url
        if args.api_key_env:
            s.llm.api_key_env = args.api_key_env
        s.playbook_dirs = [str(ROOT / Path(d)) if not Path(d).is_absolute() else d for d in s.playbook_dirs]
    s.browser.headless = not args.headed
    s.agent.workspace = str(arm_dir / "workspace")
    s.agent.runs_dir = str(arm_dir / "runs")
    s.memory.dir = str(arm_dir / "memory")
    s.memory.enabled = arm == "on"
    s.evolve.dir = args.evolve_dir or str(arm_dir / "evolve")
    return s


def make_model(args, task: Task, settings: Settings, provider: str, site_url: str):
    if provider == "scripted":
        if task.script is None:
            return None
        return ScriptedLLM(task.script(site_url))
    return LLMClient(settings.llm)


def run_one(args, task: Task, arm: str, attempt: int, settings: Settings, provider: str,
            site_url: str) -> dict[str, Any]:
    """跑一次（任務 × 組別 × 第幾次），返回一筆紀錄。"""
    record: dict[str, Any] = {
        "task_id": task.id, "mode": task.mode, "site": task.site, "arm": arm, "attempt": attempt,
        "provider": provider, "status": "skipped", "success": False, "steps": 0,
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "total_tokens": 0,
        "verify_rounds": 0, "seconds": 0.0,
        "run_dir": None, "budget_exceeded": "", "error": "", "answer": "",
    }
    if task.needs_browser and not BROWSER_OK:
        record["error"] = "本機沒有可用的 Chromium（python -m playwright install chromium），跳過瀏覽器任務"
        return record

    prompt = task.prompt.replace("{site}", site_url or "")
    inner = make_model(args, task, settings, provider, site_url)
    if inner is None:
        record["error"] = "此題沒有離線劇本，僅支援真實模型"
        return record
    guard = GuardedModel(inner, max_calls=args.max_calls_per_run, max_tokens=args.max_tokens_per_run,
                         max_seconds=args.max_seconds_per_run)
    agent = LingXi(settings=settings, llm=guard, console=False)

    started = time.time()
    result, error = None, ""
    try:
        result = asyncio.run(agent.run(prompt))
    except Exception as exc:  # 實驗基建本身出錯也要留紀錄，不能讓整個實驗崩掉
        error = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.time() - started, 2)
    record["budget_exceeded"] = guard.exceeded
    record["error"] = guard.exceeded or error

    if result is None:
        record["status"] = "error"
        return record

    usage = result.usage or {}
    record.update(status=result.status, steps=result.steps, calls=usage.get("calls", 0),
                  prompt_tokens=usage.get("prompt_tokens", 0),
                  completion_tokens=usage.get("completion_tokens", 0),
                  cached_tokens=usage.get("cached_tokens", 0),
                  total_tokens=usage.get("total_tokens", 0),
                  run_dir=result.run_dir, answer=(result.answer or "")[:300])
    missing = [x for x in task.expect if x not in (result.answer or "")]
    record["success"] = result.status == "success" and not missing and not guard.exceeded
    if missing:
        record["error"] = record["error"] or f"答覆缺少預期內容：{'、'.join(missing)}"

    if result.run_dir:
        events = Journal.load(Path(result.run_dir))
        record["verify_rounds"] = sum(1 for e in events if e.kind == "verify")
    return record


def build_summary(records: list[dict[str, Any]], suite: list[Task], args, aborted: str,
                  site_url: str) -> dict[str, Any]:
    """彙總：兩組對照 ＋ 逐題明細（repeats > 1 時成功率取多數）。"""

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        effective = [r for r in rows if r["status"] != "skipped"]
        successes = [r for r in effective if r["success"]]
        n = len(effective)
        return {
            "runs": len(rows), "effective": n, "skipped": len(rows) - n,
            "successes": len(successes),
            "success_rate": round(len(successes) / n, 4) if n else 0.0,
            "avg_steps": round(sum(r["steps"] for r in effective) / n, 2) if n else 0.0,
            "calls": sum(r["calls"] for r in effective),
            "total_tokens": sum(r["total_tokens"] for r in effective),
            "cached_tokens": sum(r["cached_tokens"] for r in effective),
            "avg_tokens": round(sum(r["total_tokens"] for r in effective) / n, 1) if n else 0.0,
            "verify_rounds": sum(r["verify_rounds"] for r in effective),
            "avg_seconds": round(sum(r["seconds"] for r in effective) / n, 2) if n else 0.0,
            "budget_exceeded": sum(1 for r in rows if r["budget_exceeded"]),
        }

    arms: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    for arm in args.arms:
        arm_rows = [r for r in records if r["arm"] == arm]
        arms[f"memory-{arm}"] = aggregate(arm_rows)
        for task in suite:
            rows = [r for r in arm_rows if r["task_id"] == task.id]
            if not rows:
                continue
            effective = [r for r in rows if r["status"] != "skipped"]
            wins = sum(1 for r in effective if r["success"])
            per_task.setdefault(task.id, {"mode": task.mode, "site": task.site,
                                          "prompt": task.prompt.replace("{site}", site_url or "")})[f"memory-{arm}"] = {
                "attempts": len(rows), "effective": len(effective), "successes": wins,
                "majority": wins * 2 > len(effective) if effective else False,
                "avg_steps": round(sum(r["steps"] for r in effective) / len(effective), 2) if effective else 0.0,
                "tokens": sum(r["total_tokens"] for r in effective),
                "verify_rounds": sum(r["verify_rounds"] for r in effective),
                "avg_seconds": round(sum(r["seconds"] for r in effective) / len(effective), 2) if effective else 0.0,
                "statuses": [r["status"] for r in rows],
            }
    return {"arms": arms, "per_task": per_task, "aborted": aborted}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1 有記憶／無記憶對照實驗（預設離線乾跑）")
    p.add_argument("--suite", default="main", choices=sorted(SUITES), help="任務集（main=12 題正式集；smoke=2 題冒煙）")
    p.add_argument("--provider", default=os.getenv("LINGXI_EXPERIMENT_PROVIDER", "scripted"),
                   help="scripted=離線假模型（預設）；其他值=真實模型端點名稱，沿用 config/lingxi.toml")
    p.add_argument("--arms", default="off,on", help="要跑的對照組，逗號分隔：off,on")
    p.add_argument("--repeats", type=int, default=1, help="每題重複執行次數（W1 正式實驗建議 3，成功率取多數）")
    p.add_argument("--out-dir", default=str(ROOT / "experiments" / "runs"), help="實驗產出的根目錄")
    p.add_argument("--evolve-dir", default="", help="RSI 狀態目錄；預設每組隔離在實驗目錄內（從 v0 開始）")
    # 單次執行護欄
    p.add_argument("--max-calls-per-run", type=int, default=80, help="單次執行模型呼叫數上限（0=不限）")
    p.add_argument("--max-tokens-per-run", type=int, default=300_000, help="單次執行 token 上限（0=不限）")
    p.add_argument("--max-seconds-per-run", type=float, default=600.0, help="單次執行牆鐘秒數上限（0=不限）")
    # 整個實驗的總成本護欄
    p.add_argument("--max-total-calls", type=int, default=800, help="整個實驗模型呼叫總數上限（0=不限）")
    p.add_argument("--max-total-tokens", type=int, default=3_000_000, help="整個實驗 token 總量上限（0=不限）")
    # 真實模型端點覆寫
    p.add_argument("--model", default="", help="覆寫 [llm] model（僅真實模型模式）")
    p.add_argument("--base-url", default="", help="覆寫 [llm] base_url（僅真實模型模式）")
    p.add_argument("--api-key-env", default="", help="覆寫 [llm] api_key_env（僅真實模型模式）")
    p.add_argument("--headed", action="store_true", help="瀏覽器不以無頭模式執行（除錯用）")
    p.add_argument("--no-latest-report", action="store_true",
                   help="只把報告寫進實驗目錄，不更新 experiments/report.md（測試用）")
    args = p.parse_args(argv)
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    invalid = [a for a in args.arms if a not in ARMS]
    if invalid:
        p.error(f"--arms 只支援 {ARMS}，收到：{invalid}")
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    provider = args.provider
    suite: list[Task] = SUITES[args.suite]()

    if provider != "scripted":
        probe = load_settings()
        if args.api_key_env:
            probe.llm.api_key_env = args.api_key_env
        if not probe.llm.resolve_api_key():
            raise SystemExit(
                f"provider={provider} 需要真實模型，但找不到 API Key："
                f"請設定環境變數 {probe.llm.api_key_env} 或 LINGXI_API_KEY，"
                "或用 --api-key-env 指定；離線乾跑請改用預設的 --provider scripted")

    run_root = Path(args.out_dir) / f"{datetime.now():%Y%m%d-%H%M%S}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"🧪 W1 對照實驗：suite={args.suite}（{len(suite)} 題）· provider={provider} · "
          f"arms={args.arms} · repeats={args.repeats}\n   產出目錄：{run_root}", flush=True)

    server = None
    site_url = ""
    if any(t.needs_site for t in suite):
        server, site_url = start_site(run_root / "site")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite, "provider": provider, "arms": args.arms, "repeats": args.repeats,
        "tasks": [{"id": t.id, "mode": t.mode, "site": t.site, "prompt": t.prompt,
                   "needs_browser": t.needs_browser, "expect": list(t.expect)} for t in suite],
        "caps": {"max_calls_per_run": args.max_calls_per_run, "max_tokens_per_run": args.max_tokens_per_run,
                 "max_seconds_per_run": args.max_seconds_per_run,
                 "max_total_calls": args.max_total_calls, "max_total_tokens": args.max_total_tokens},
        "browser_available": BROWSER_OK, "site_url": site_url, "aborted": "",
    }

    records_path = run_root / "records.jsonl"
    records: list[dict[str, Any]] = []
    totals = Totals()
    aborted = ""
    try:
        for arm in args.arms:
            arm_dir = run_root / f"memory-{arm}"
            settings = build_settings(args, arm, arm_dir, provider)
            for task in suite:
                for attempt in range(1, args.repeats + 1):
                    if not aborted:
                        aborted = totals.over(args.max_total_calls, args.max_total_tokens)
                    if aborted:
                        break
                    print(f"  ▶ [{task.id} · memory-{arm} · 第 {attempt} 次] {task.prompt[:40]}…", flush=True)
                    rec = run_one(args, task, arm, attempt, settings, provider, site_url)
                    records.append(rec)
                    with open(records_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    totals.add(rec["calls"], rec["total_tokens"])
                    flag = "✔" if rec["success"] else ("⊘" if rec["status"] == "skipped" else "✘")
                    note = f"（{rec['error']}）" if rec["error"] and not rec["success"] else ""
                    print(f"    {flag} {rec['status']} · {rec['steps']} 步 · {rec['total_tokens']} tokens"
                          f" · {rec['seconds']}s {note}", flush=True)
                if aborted:
                    break
            if aborted:
                break
    finally:
        if server is not None:
            server.shutdown()

    if aborted:
        print(f"⛔ 實驗因成本護欄中止：{aborted}", flush=True)
    meta.update(aborted=aborted, finished=datetime.now().isoformat(timespec="seconds"),
                totals={"runs": totals.runs, "calls": totals.calls, "tokens": totals.tokens})
    (run_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {"meta": {k: meta[k] for k in ("created", "suite", "provider", "arms", "repeats", "aborted", "totals")},
               **build_summary(records, suite, args, aborted, site_url)}
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    generate_report(run_root, update_latest=not args.no_latest_report)

    for name, agg in summary["arms"].items():
        print(f"📊 {name}：成功率 {agg['success_rate']:.0%}（{agg['successes']}/{agg['effective']}）· "
              f"平均 {agg['avg_steps']} 步 · {agg['total_tokens']} tokens · 平均 {agg['avg_seconds']}s", flush=True)
    print(f"✅ 紀錄已寫入 {run_root}；報告：{run_root / 'report.md'}", flush=True)
    return run_root


if __name__ == "__main__":
    main()
