"""核心端到端：用 ScriptedLLM 代替真實模型，在夾具頁上跑完整的一次任務。

不花一分錢 token，卻覆蓋了：預處理（錨定/手冊/模式）→ 簡報 → 工具執行與驗證 →
進展預算續航 → 黑匣子事件 → result.md → 復盤報告 → Web 介面。
"""

import asyncio
from pathlib import Path

import pytest

from conftest import ScriptedLLM, call, needs_browser, reply
from lingxi.journal.recorder import Journal
from lingxi.journal.report import write_report
from lingxi.kernel.loop import Kernel
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.skills import SkillSet, builtin_skills


def kernel(settings, llm):
    library = PlaybookLibrary.load([settings.path(d) for d in settings.playbook_dirs])
    return Kernel(settings, llm, SkillSet(builtin_skills()), library=library)


def last_brief(messages) -> str:
    return messages[-1].content


@needs_browser
def test_full_flight_query_on_fixture(settings, site):
    seen = {}

    def conclude(messages, tools):
        brief = last_brief(messages)
        seen["brief"] = brief
        assert "【頁面】" in brief and "MU5101" in brief
        return reply(call("finish", answer="找到 3 個航班：MU5101 ¥680、CA1858 ¥720、HU7604 ¥880", status="success"))

    llm = ScriptedLLM([
        reply(call("web_open", url=f"{site}/flight.html"), content="先開啟訂票頁"),
        reply(call("web_type", target="出發城市", text="上海")),
        reply(call("web_click", target="上海(SHA)")),
        reply(call("web_type", target="到達城市", text="北京")),
        reply(call("web_click", target="北京(BJS)")),
        reply(call("web_click", target="出發日期")),
        reply(call("web_click", target="6月26日")),
        reply(call("web_click", target="搜尋")),
        conclude,
    ])
    result = asyncio.run(kernel(settings, llm).run("在測試頁查詢 6月26日 從上海到北京的航班"))

    assert result.status == "success", result.answer
    assert "MU5101" in result.answer and result.steps == 9

    run_dir = Path(result.run_dir)
    events = Journal.load(run_dir)
    kinds = [e.kind for e in events]
    assert kinds[0] == "run.start" and kinds[-1] == "run.finish"
    intent = next(e for e in events if e.kind == "intent")
    assert intent.data["profile"] == "web_query"
    assert next(e for e in events if e.kind == "temporal").data["anchors"][0]["date"].startswith("2027-06-26")
    assert next(e for e in events if e.kind == "playbook").data["id"] == "ctrip-flight"

    outcomes = [e.data for e in events if e.kind == "outcome"]
    assert all(o["ok"] for o in outcomes), [o["summary"] for o in outcomes if not o["ok"]]
    interactions = [o for o in outcomes if o["name"] in ("web_type", "web_click")]
    assert all(o["verified"] for o in interactions), [o["summary"] for o in interactions]
    assert {o["grounding"]["method"] for o in interactions} >= {"text", "date"}

    # 每一步都有可驗證的進展 → 預算續航
    assert any(e.data.get("delta", 0) > 0 for e in events if e.kind == "budget")

    assert (run_dir / "result.md").read_text(encoding="utf-8").count("MU5101") >= 1
    html = write_report(run_dir).read_text(encoding="utf-8")
    assert "最終答覆" in html and "定位：date" in html

    # 系統提示裡帶著已編譯好的手冊與當前日期
    system = llm.requests[0][0][0].content
    assert "經驗手冊" in system and "oneway-sha-bjs" in system


def test_model_that_never_calls_tools_still_gets_an_answer(settings):
    llm = ScriptedLLM([reply(content="我覺得是 42。"), reply(content="答案是 42。")])
    result = asyncio.run(kernel(settings, llm).run("你好，隨便回答一個數字"))
    assert result.status == "success" and result.answer == "答案是 42。"
    # 第一次沒調工具時，核心會提示模型"請呼叫工具或 finish"
    second_request = llm.requests[1][0]
    assert any("請透過呼叫工具推進任務" in m.content for m in second_request if m.role == "user")


def test_loop_guard_penalizes_and_last_step_forces_finish(settings):
    def stubborn(messages, tools):
        names = [t["function"]["name"] for t in tools]
        if names == ["finish"]:
            return reply(call("finish", answer="只能給出階段性結論", status="partial"))
        return reply(call("files", action="list", path="."))

    llm = ScriptedLLM([], fallback=stubborn)
    result = asyncio.run(kernel(settings, llm).run("你好"))
    assert result.status == "partial"
    events = Journal.load(next(settings.runs_dir.iterdir()))
    assert any(e.kind == "guard" and "同一頁面狀態" in e.data["message"] for e in events)
    assert result.steps < 24  # 通用模式基礎預算 24 步，原地打轉被扣減後提前收尾
    # 最後一步用 tool_choice 強制 finish，工具清單與上一步完全相同（KV 快取前綴不失效）
    assert llm.requests[-1][2] == {"type": "function", "function": {"name": "finish"}}
    assert llm.requests[-1][1] == llm.requests[-2][1]


def test_unknown_tool_and_bad_arguments_are_reported_to_model(settings):
    llm = ScriptedLLM([
        reply(call("fly_to_moon"), call("plan", action="done")),
        reply(call("finish", answer="ok")),
    ])
    result = asyncio.run(kernel(settings, llm).run("你好"))
    assert result.status == "success"
    tool_msgs = [m for m in llm.requests[1][0] if m.role == "tool"]
    assert "未知工具" in tool_msgs[0].content
    assert "index" in tool_msgs[1].content  # plan done 缺少 index，給出了明確提示


def test_web_api_serves_recorded_runs(settings):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lingxi.web.server import create_app

    asyncio.run(kernel(settings, ScriptedLLM([reply(call("finish", answer="完成"))])).run("你好"))
    client = TestClient(create_app(settings))
    runs = client.get("/api/runs").json()
    assert runs and runs[0]["status"] == "success"
    run_id = runs[0]["run_id"]
    body = client.get(f"/api/runs/{run_id}/events").text
    assert '"kind": "run.finish"' in body and "event: end" in body
    assert "最終答覆" in client.get(f"/api/runs/{run_id}/report").text
    assert client.get("/api/runs/../../etc/events").status_code == 404
    assert "靈犀" in client.get("/").text
