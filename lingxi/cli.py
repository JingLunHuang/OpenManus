"""命令列入口。

  lingxi run "查詢 6月26日 從上海到北京的機票"   執行任務
  lingxi serve                                  啟動 Web 介面
  lingxi playbooks "任務"                        試跑知識層：看時間錨定 / 手冊匹配 / 任務模式（不呼叫模型）
  lingxi replay latest                          把最近一次執行渲染成 report.html
  lingxi runs                                   列出最近的執行
  lingxi doctor                                 檢查配置、金鑰、瀏覽器

  lingxi memory [stats|recall 查詢|ingest|sleep] 自我學習記憶（LightMem ＋ FluxMem）
  lingxi evolve [status|round|rollback 版本]     遞迴自我改進（RSI）閘門
  lingxi bench                                  加速基準：KV 快取、LlamaIndex 檢索、精讀聚焦、搜尋快取
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


# ---------------- memory ----------------
def _memory(s):
    from lingxi.llm.client import LLMClient
    from lingxi.memory import MemorySystem

    return MemorySystem(s, llm=LLMClient(s.llm) if s.llm.resolve_api_key() else None)


def cmd_memory(args) -> int:
    s = _settings(args)
    mem = _memory(s)
    action = args.action or "stats"
    if action == "stats":
        st = mem.stats()
        print(f"記憶圖（{s.memory_dir / 'graph.json'}）· 檢索後端 {st['backend']} · 向量 {st['embedder']}")
        print(f"  語義 {st['semantic']} · 情節 {st['episodic']} · 程序 {st['procedural']}"
              f"（已被取代 {st['superseded']}、已退役 {st['retired']}）")
        print(f"  邊：ground {st['edges']['ground']} · distill {st['edges']['distill']}"
              f" · 距上次睡眠 {st['runs_since_sleep']} 次執行 · 已睡眠 {st['sleeps']} 次")
        for n in mem.graph.active("procedural"):
            d = n.data
            print(f"  {n.id}《{n.title}》PEMS {' → '.join(f'{x:.3f}' for x in d.get('pems', []))}"
                  f"（{'已收斂' if d.get('converged') else '未收斂'}，支撐 {len(d.get('support', []))} 次）")
        return 0
    if action == "recall":
        query = " ".join(args.query).strip()
        sub = mem.recall(query)
        print(sub.render() or "（沒有召回任何記憶）")
        return 0
    if action == "ingest":
        rows = mem.ingest(s.runs_dir)
        for r in rows:
            print(f"  + {r['run']}：{r.get('episode', '')} 站點知識 {len(r.get('notes', []))} 條、事實 {len(r.get('facts', []))} 條")
        print(f"補寫 {len(rows)} 次執行。")
        return 0
    if action == "sleep":
        report = asyncio.run(mem.sleep(args.mode))
        u, c = report["update"], report["consolidation"]
        print(f"睡眠整理（{report['mode']}）")
        print(f"  LightMem 離線更新：{u['queues']} 個更新佇列 → 合併 {u['merged']}、以新換舊 {u['updated']}")
        print(f"  FluxMem 鞏固：{c['clusters']} 個情節群 → {len(c['skills'])} 個程序技能；重塑 {c['reshaped']}、退役 {c['retired']}")
        for sk in c["skills"]:
            print(f"    {sk['id']}《{sk['title']}》PEMS {' → '.join(f'{x:.3f}' for x in sk['pems'])}"
                  f"（{'收斂' if sk['converged'] else '未收斂'}，{sk['support']}/{sk['cluster']} 次成功）")
        return 0
    print(f"未知動作：{action}")
    return 1


# ---------------- evolve ----------------
def cmd_evolve(args) -> int:
    from lingxi.evolve import RSILoop
    from lingxi.memory import MemorySystem

    s = _settings(args)
    loop = RSILoop(s, MemorySystem(s).graph if s.memory.enabled else None)
    action = args.action or "status"
    if action == "round":
        r = loop.round()
        target = f"v{r.new_version}" if r.new_version is not None else "（沒有候選通過，維持原版本）"
        print(f"RSI 第 {r.round} 輪 · 評測週期 {r.epoch} · v{r.base_version} → {target}")
        print(f"  經驗：{r.experience_runs} 次新執行 · 評測預算 {r.budget['used']}/{r.budget['limit']}")
        for c in r.candidates:
            mark = "✔" if c["accepted"] else "✘"
            what = c.get("playbook") or "，".join(f"{k} {v[0]}→{v[1]}" for k, v in c["change"].items())
            gain = f"{c['gain']:+.4f}" if "gain" in c else "—"
            print(f"  {mark} [{c['target']}] {what} · 受保護 {c['suite']} {gain} · {c['reason']}")
        print("  分數：" + " · ".join(f"{k} {r.baseline.get(k, 0):.3f}→{r.final.get(k, 0):.3f}"
                                    f"（HCI {r.hci.get(k) if r.hci.get(k) is not None else '—'}）" for k in r.baseline))
        for d in r.drift:
            print(f"  ⚠ 目標漂移：{d}")
        if r.curriculum:
            print("  建議重練：" + "；".join(r.curriculum))
        print(f"  自主等級：{r.autonomy['level']}；L5 {r.autonomy['l5']}")
        if r.note:
            print(f"  備註：{r.note}")
        return 0
    if action == "rollback":
        loop.rollback(int(args.version))
        print(f"已回滾：目前生效版本為 v{args.version}")
        return 0
    current = loop.store.current()
    print(f"目前生效版本：v{current.version} · 受保護評測週期 {loop.evaluator.epoch}")
    for st in loop.store.history():
        mark = "▶" if st.version == current.version else " "
        scores = " · ".join(f"{k} {v:.3f}" for k, v in st.scores.items() if k != "epoch")
        print(f" {mark} v{st.version}（父 {'v' + str(st.parent) if st.parent is not None else '—'}）{scores}  {st.note[:60]}")
    rounds = loop.rounds()
    print(f"已執行 {len(rounds)} 輪 RSI；最近一輪：" + (rounds[-1].get("autonomy", {}).get("level", "回滾") if rounds else "—"))
    return 0


def cmd_bench(args) -> int:
    from lingxi.bench import render, run_all

    print(render(run_all(_settings(args))))
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
    for mod, label in (("playwright", "Playwright"), ("llama_index.core", "LlamaIndex 檢索"), ("fastapi", "Web 介面"),
                       ("mcp", "MCP"), ("daytona", "Daytona"), ("ddgs", "DDGS 搜尋")):
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

    p = sub.add_parser("memory", help="自我學習記憶：stats / recall 查詢 / ingest / sleep")
    p.add_argument("action", nargs="?", choices=["stats", "recall", "ingest", "sleep"])
    p.add_argument("query", nargs="*")
    p.add_argument("--mode", choices=["rules", "llm"], help="睡眠整理用規則或模型（預設依 memory.consolidate_with）")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("evolve", help="遞迴自我改進：status / round / rollback 版本")
    p.add_argument("action", nargs="?", choices=["status", "round", "rollback"])
    p.add_argument("version", nargs="?")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("bench", help="加速基準（離線）")
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
