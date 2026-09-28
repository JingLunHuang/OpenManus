"""Web 介面：FastAPI + SSE。

介面只是黑匣子的另一個訂閱者：執行中即時推送事件，執行結束後同樣能從磁碟迴放。
同一時間只允許一個執行（瀏覽器會話是獨佔資源）。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from lingxi import __version__
from lingxi.agent import LingXi
from lingxi.journal.recorder import Journal
from lingxi.journal.report import render_report
from lingxi.kernel.preflight import preflight
from lingxi.kernel.profiles import PROFILES
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.settings import Settings, get_settings
from lingxi.skills import SkillSet, builtin_skills
from lingxi.skills.remote import DaytonaRun

STATIC = Path(__file__).parent / "static"


class WebHuman:
    """ask_human 的網頁版：提問時掛起，等待前端 POST 回答。"""

    def __init__(self) -> None:
        self._future: asyncio.Future[str] | None = None

    async def ask(self, question: str) -> str:
        self._future = asyncio.get_running_loop().create_future()
        try:
            return await asyncio.wait_for(self._future, timeout=900)
        except asyncio.TimeoutError:
            return "（使用者 15 分鐘內沒有回答，請自行決定或給出階段性結論）"
        finally:
            self._future = None

    def answer(self, text: str) -> bool:
        if self._future and not self._future.done():
            self._future.set_result(text)
            return True
        return False


class RunHandle:
    def __init__(self, journal: Journal, human: WebHuman):
        self.journal = journal
        self.human = human
        self.task: asyncio.Task | None = None


class NewRun(BaseModel):
    task: str


class Answer(BaseModel):
    answer: str


class Rollback(BaseModel):
    version: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="靈犀 LingXi")
    runs: dict[str, RunHandle] = {}
    busy = asyncio.Lock()

    def run_dir(run_id: str) -> Path:
        path = (settings.runs_dir / run_id).resolve()
        if settings.runs_dir.resolve() not in path.parents or not (path / "events.jsonl").exists():
            raise HTTPException(404, "執行記錄不存在")
        return path

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/runs")
    async def start(body: NewRun):
        task = body.task.strip()
        if not task:
            raise HTTPException(400, "任務不能為空")
        if busy.locked():
            raise HTTPException(409, "已有任務在執行，請等待它結束")
        journal = Journal.create(settings.runs_dir, task)
        handle = RunHandle(journal, WebHuman())
        runs[journal.run_id] = handle

        async def worker():
            async with busy:
                try:
                    await LingXi(settings, human=handle.human, console=False).run(task, journal=journal)
                except Exception as exc:  # 兜底：確保前端能收到結束訊號
                    journal.emit("error", message=str(exc))
                    journal.close()

        handle.task = asyncio.create_task(worker())
        return {"run_id": journal.run_id}

    @app.get("/api/runs")
    async def list_runs():
        items = []
        for path in sorted(settings.runs_dir.iterdir(), key=lambda p: p.name, reverse=True)[:40]:
            if not (path / "events.jsonl").exists():
                continue
            live = path.name in runs and not runs[path.name].journal.closed
            item = {"run_id": path.name, "task": path.name, "status": "running" if live else "unknown",
                    "steps": None, "tokens": None, "profile": None, "verify": None, "seconds": None}
            try:
                events = Journal.load(path)
                start = next((e for e in events if e.kind == "run.start"), None)
                finish = next((e for e in reversed(events) if e.kind == "run.finish"), None)
                if start:
                    item.update(task=start.data.get("task", path.name), profile=start.data.get("profile"))
                if finish:
                    verify = finish.data.get("verify") or {}
                    item.update(status=finish.data.get("status", item["status"]), steps=finish.step,
                                tokens=finish.data.get("usage", {}).get("total_tokens"),
                                seconds=finish.data.get("seconds"),
                                verify=None if not verify else ("passed" if verify.get("passed") else "flagged"))
            except Exception:
                pass
            items.append(item)
        return items

    @app.get("/api/framework")
    async def framework():
        """框架自我描述：給「總覽 / 技能 / 手冊」頁面使用，不呼叫任何模型。"""
        skills = SkillSet(builtin_skills())
        if settings.daytona.resolve_api_key():
            skills.add(DaytonaRun(settings.daytona))
        if settings.memory.enabled:
            from lingxi.memory import Recall

            skills.add(Recall(None))  # 只用來描述介面，不會被呼叫
        skill_list = []
        for skill in skills:
            schema = skill.tool_schema()["function"]
            props = schema["parameters"].get("properties", {})
            skill_list.append({
                "name": skill.name, "description": skill.description,
                "group": ("瀏覽器" if skill.name.startswith("web_") and skill.name != "web_search" else
                          "搜尋" if skill.name == "web_search" else
                          "記憶" if skill.name == "recall" else
                          "雲端" if skill.name == "sandbox_run" else "本地"),
                "params": [{"name": k, "description": v.get("description", ""),
                            "required": k in schema["parameters"].get("required", [])} for k, v in props.items()],
            })
        library = PlaybookLibrary.load([settings.path(d) for d in settings.playbook_dirs])
        llm, vision = settings.llm, settings.llm.vision
        return {
            "name": "靈犀 LingXi", "version": __version__, "lineage": "OpenManus 開發框架",
            "config": {
                "model": llm.model, "endpoint": urlparse(llm.base_url).hostname, "key_ready": bool(llm.resolve_api_key()),
                "vision_model": vision.model, "vision_ready": bool(vision.enabled and vision.resolve_api_key()),
                "headless": settings.browser.headless, "cdp": bool(settings.browser.cdp_url),
                "search": settings.search.providers, "verify": settings.verify.model_dump(),
                "mcp_servers": list(settings.mcp_servers), "source": settings.source,
            },
            "skills": skill_list,
            "playbooks": [{**info, "url": pb.url, "warmup": pb.warmup, "tips": pb.tips}
                          for info, pb in zip(library.describe(), library.playbooks)],
            "profiles": [{"name": p.name, "title": p.title, "budget": p.base_budget, "date": p.date_preference,
                          "guidance": p.guidance} for p in PROFILES.values()],
        }

    @app.post("/api/preflight")
    async def dry_run(body: NewRun):
        """試跑知識層：意圖模式、時間錨定、手冊編譯（確定性，不花 token）。"""
        library = PlaybookLibrary.load([settings.path(d) for d in settings.playbook_dirs])
        pf = await preflight(body.task, library)
        return {
            "briefed": pf.briefed, "profile": pf.profile.name, "title": pf.profile.title, "reason": pf.reason,
            "budget": pf.profile.base_budget, "preference": pf.preference,
            "anchors": [{"text": a.text, "date": a.label()} for a in pf.anchors],
            "playbooks": [{"id": pb.id, "title": pb.title, "url": pb.url, "missing": pb.missing,
                           "slots": pb.shown, "render": pb.render()} for pb in pf.playbooks],
        }

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str):
        handle = runs.get(run_id)

        async def stream():
            if handle:
                q = handle.journal.subscribe(replay=True)
                try:
                    while True:
                        event = await q.get()
                        if event is None:
                            break
                        yield f"data: {json.dumps(asdict(event), ensure_ascii=False, default=str)}\n\n"
                finally:
                    handle.journal.unsubscribe(q)
            else:
                for event in Journal.load(run_dir(run_id)):
                    yield f"data: {json.dumps(asdict(event), ensure_ascii=False, default=str)}\n\n"
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/answer")
    async def answer(run_id: str, body: Answer):
        handle = runs.get(run_id)
        if not handle or not handle.human.answer(body.answer):
            raise HTTPException(409, "當前沒有等待回答的問題")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/artifacts/{name:path}")
    async def artifact(run_id: str, name: str):
        base = run_dir(run_id)
        path = (base / name).resolve()
        if base not in path.parents or not path.is_file():
            raise HTTPException(404, "附件不存在")
        return FileResponse(path)

    # ---------------- 學習：記憶 / RSI / 加速 ----------------
    def memory_system():
        from lingxi.memory import MemorySystem

        return MemorySystem(settings)

    @app.get("/api/learning")
    async def learning():
        """自我學習面板：記憶圖統計、程序技能、RSI 版本與帳本、KV 快取統計（不呼叫模型）。"""
        from lingxi.evolve import TUNABLE, RSILoop, effective_harness

        out: dict = {"memory": None, "evolve": None, "kv": None, "settings": {
            "memory": settings.memory.model_dump(), "kv_cache": settings.kv_cache.model_dump(),
            "retrieval": settings.retrieval.model_dump(), "evolve": settings.evolve.model_dump()}}
        graph = None
        if settings.memory.enabled:
            mem = memory_system()
            graph = mem.graph
            out["memory"] = {"stats": mem.stats(), "procedural": [
                {"id": n.id, "title": n.title, "site": n.site, "steps": n.data.get("steps", []),
                 "pitfalls": n.data.get("pitfalls", []), "pems": n.data.get("pems", []),
                 "converged": n.data.get("converged"), "support": len(n.data.get("support", [])),
                 "success_rate": n.data.get("success_rate"), "mode": n.data.get("mode"), "uses": n.uses}
                for n in graph.active("procedural")],
                "recent_semantic": [{"id": n.id, "site": n.site, "text": n.text, "kind": n.data.get("kind"),
                                     "helpful": n.helpful, "harmful": n.harmful, "age": n.age()}
                                    for n in sorted(graph.active("semantic"), key=lambda n: -n.created)[:12]]}
        if settings.evolve.enabled:
            loop = RSILoop(settings, graph)
            current = loop.store.current()
            out["evolve"] = {"current": asdict(current), "epoch": loop.evaluator.epoch,
                             "harness": effective_harness(settings, current),
                             "tunable": {k: v[1:] for k, v in TUNABLE.items()},
                             "versions": [asdict(s) for s in loop.store.history()], "rounds": loop.rounds()[-10:]}
        kv = []
        for path in sorted(settings.runs_dir.iterdir(), key=lambda p: p.name, reverse=True)[:20]:
            try:
                finish = next((e for e in reversed(Journal.load(path)) if e.kind == "run.finish"), None)
            except Exception:
                continue
            if finish and finish.data.get("kv"):
                kv.append({"run_id": path.name, **{k: v for k, v in finish.data["kv"].items() if k != "per_step"},
                           "prompt_tokens": finish.data.get("usage", {}).get("prompt_tokens", 0)})
        out["kv"] = kv
        return out

    @app.post("/api/memory/recall")
    async def memory_recall(body: NewRun):
        if not settings.memory.enabled:
            raise HTTPException(400, "記憶未啟用")
        sub = await asyncio.to_thread(memory_system().recall, body.task)
        return {"render": sub.render() or "", **sub.summary()}

    @app.post("/api/memory/sleep")
    async def memory_sleep():
        if not settings.memory.enabled:
            raise HTTPException(400, "記憶未啟用")
        return await memory_system().sleep("rules")

    @app.post("/api/evolve/round")
    async def evolve_round():
        from lingxi.evolve import RSILoop

        if not settings.evolve.enabled:
            raise HTTPException(400, "RSI 未啟用")
        graph = memory_system().graph if settings.memory.enabled else None
        return asdict(await asyncio.to_thread(RSILoop(settings, graph).round))

    @app.post("/api/evolve/rollback")
    async def evolve_rollback(body: Rollback):
        from lingxi.evolve import RSILoop

        try:
            RSILoop(settings).rollback(body.version)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True, "current": body.version}

    @app.get("/api/bench")
    async def bench():
        from lingxi.bench import render, run_all

        result = await asyncio.to_thread(run_all, settings)
        return {"result": result, "render": render(result)}

    @app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
    async def report(run_id: str):
        base = run_dir(run_id)
        html = render_report(Journal.load(base))
        return html.replace("src='artifacts/", f"src='/api/runs/{run_id}/artifacts/artifacts/")

    return app
