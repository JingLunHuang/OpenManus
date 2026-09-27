"""核心部件：意圖路由、進展預算、滾動記憶、視覺解析、元素定位打分。"""

import asyncio

import pytest

from lingxi.kernel.budget import ProgressBudget
from lingxi.kernel.intent import route, route_by_rules
from lingxi.kernel.memory import RollingMemory
from lingxi.llm.messages import Message, Reply, ToolCall
from lingxi.senses.grounding import Ambiguous, match_snapshot, parse_id
from lingxi.senses.page import Element, PageSnapshot
from lingxi.senses.vision import parse_vision_reply
from lingxi.skills.web import fix_stale_dates, normalize_keys
from lingxi.utils import extract_json


# ---------------- 意圖路由 ----------------
@pytest.mark.parametrize("task, profile", [
    ("查一下明天杭州的天氣", "web_query"),
    ("寫一份新能源汽車競品分析報告", "research"),
    ("用 python 把 data.csv 按月彙總", "coding"),
])
def test_rule_routing(task, profile):
    assert route_by_rules(task).profile.name == profile


def test_unclear_task_falls_back_to_general():
    assert route_by_rules("你好") is None
    assert asyncio.run(route("你好")).profile.name == "general"


def test_llm_fallback_is_used_only_when_rules_are_unsure():
    class OneWord:
        async def chat(self, messages, **kw):
            return Reply(content="coding")

    assert asyncio.run(route("你好", llm=OneWord(), use_llm=True)).profile.name == "coding"


# ---------------- 進展預算 ----------------
def test_budget_extends_on_progress_and_is_capped():
    b = ProgressBudget.for_profile(3, 2.0)
    assert (b.remaining, b.cap) == (3, 6)
    while b.remaining > 0:  # 與核心主迴圈一致
        b.spend()
        b.reward({"new_url", "verified_action"})
    # 每一步都有進展：預算一路續航到上限 6，之後自然耗盡
    assert (b.granted, b.used, b.remaining) == (6, 6, 0)


def test_budget_ignores_weak_signals_and_penalty_keeps_last_step():
    b = ProgressBudget.for_profile(5, 2.0)
    assert b.reward(set()) == (0, "")
    assert b.reward({"unknown"}) == (0, "")
    b.penalize(10, "loop")
    assert b.remaining == 1


# ---------------- 滾動記憶 ----------------
def test_memory_folds_old_observations_but_keeps_tool_pairing():
    mem = RollingMemory(fold_after=2, max_observation_chars=50)
    for i in range(5):
        c = ToolCall(id=f"c{i}", name="web_open", arguments="{}")
        mem.add_turn(Message.assistant("思考" * 200, [c]), [Message.tool(c, "頁面內容" * 100, digest=f"摘要{i}")])
    msgs = mem.messages()
    tools = [m for m in msgs if m.role == "tool"]
    assert [m.tool_call_id for m in tools] == [f"c{i}" for i in range(5)]
    assert tools[0].content == "[已摺疊] 摘要0"
    assert tools[-1].content.startswith("頁面內容") and len(tools[-1].content) <= 50
    assert len(msgs[0].content) <= 200  # 舊的思考被截短


# ---------------- 視覺模型輸出解析 ----------------
@pytest.mark.parametrize("raw", [
    '{"thought": "看到了日曆", "action": "CLICK", "parameters": {"x": 652, "y": 537}}',
    '```json\n{"action": "CLICK", "parameters": {"x": 652, "y": 537}}\n```',
    '{"action": "CLICK", "parameters": {"x": 652, 537}}',
    '{"action": "CLICK", "parameters": {"x": [652, 537]}}',
    '{"action": "CLICK", "parameters": {"x": 652, "y": 537}',
    '點選位置 (652, 537)',
    '{"bbox": [642, 527, 662, 547]}',
])
def test_parse_vision_reply_variants(raw):
    p = parse_vision_reply(raw, (1280, 800))
    assert p is not None and (round(p.x), round(p.y)) == (652, 537)


def test_parse_vision_fail_and_normalized_coords():
    assert parse_vision_reply('{"action": "FAIL", "parameters": {"reason": "沒有"}}', (100, 100)) is None
    p = parse_vision_reply('{"x": 500, "y": 250}', (1280, 800), coord_space="norm1000")
    assert (p.x, p.y) == (640, 200)


def test_extract_json_repairs_fences_and_braces():
    assert extract_json('結果如下：```json\n{"facts": ["a", "b"], "found": true\n```') == {"facts": ["a", "b"], "found": True}
    assert extract_json("沒有 JSON") is None


# ---------------- 元素定位打分 ----------------
def _el(i, kind, label, **kw):
    return Element(id=i, tag="div", kind=kind, label=label, **kw)


SNAP = PageSnapshot(url="http://x", title="t", elements=[
    _el(1, "input", "出發城市", value="新加坡"),
    _el(2, "input", "到達城市", value=""),
    _el(3, "button", "搜尋"),
    _el(4, "date", "26 ¥560", date={"day": 26, "month": 6, "year": 2027}),
    _el(5, "date", "26 ¥700", date={"day": 26, "month": 7, "year": 2027}),
    _el(6, "date", "27 ¥600", date={"day": 27, "month": 6, "year": 2027}),
    _el(7, "link", "搜尋", inView=False),
])


def test_grounding_by_label_intent_and_date():
    assert match_snapshot(SNAP, "出發城市", "type").element.id == 1
    assert match_snapshot(SNAP, "搜尋按鈕", "click").element.id == 3  # "按鈕"被剝離；視口內的優先
    r = match_snapshot(SNAP, "6月26日", "click")
    assert r.element.id == 4 and r.method == "date"
    assert match_snapshot(SNAP, "7月26號", "click").element.id == 5


def test_grounding_refuses_to_guess_when_ambiguous():
    with pytest.raises(Ambiguous) as exc:
        match_snapshot(SNAP, "26", "click")  # 兩個月都有 26 日
    assert {e.id for e in exc.value.candidates} == {4, 5}


def test_parse_id_explicit_vs_bare():
    assert parse_id("#45") == 45 and parse_id("編號 45") == 45
    assert parse_id("26") is None and parse_id("26", allow_bare=True) == 26


# ---------------- 其他小工具 ----------------
def test_fix_stale_dates_only_touches_declared_params():
    from datetime import date

    url, changes = fix_stale_dates("https://a.com/x?depdate=2023-06-26&id=2023-06-26", {"depdate"}, date(2026, 9, 27))
    assert "depdate=2027-06-26" in url and "id=2023-06-26" in url and len(changes) == 1


def test_normalize_keys():
    assert normalize_keys("esc") == "Escape"
    assert normalize_keys("ctrl+a") == "Control+A"
    assert normalize_keys("arrowdown") == "ArrowDown"
