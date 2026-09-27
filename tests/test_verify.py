"""反覆查證：發現板的多源印證，以及 finish 之前的證據閘門。"""

import asyncio
import json

from conftest import ScriptedLLM, call, reply
from lingxi.hanzi import fold
from lingxi.journal.recorder import Journal
from lingxi.kernel.boards import FindingsBoard
from lingxi.kernel.loop import Kernel
from lingxi.kernel.verify import FactCheck, Claim, fact_check
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.llm.messages import Reply
from lingxi.skills import SkillSet, builtin_skills


# ---------------- 發現板：多源印證 ----------------
def test_board_marks_corroboration_and_conflicts():
    board = FindingsBoard()
    assert board.add(["比亞迪 2025 年歐洲銷量 18 萬輛"], source="a.com") == 1
    # 同一事實（以 fold() 產生的簡體寫法）出現在另一個來源 → 印證，而不是新增
    assert board.add([fold("比亞迪 2025 年歐洲銷量 18 萬輛")], source="b.com") == 0
    assert board.items[0].status == "corroborated" and board.items[0].sources == ["a.com", "b.com"]
    board.add(["小鵬 G6 歐洲售價 4.2 萬歐元"], source="c.com")
    assert board.contradict("F2", "d.com", "d.com 報 4.0 萬歐元")
    assert board.stats() == {"single": 0, "corroborated": 1, "conflict": 1}
    text = board.render()
    assert "F1 ✔2方印證" in text and "F2 ⚠有矛盾" in text and "d.com 報 4.0 萬歐元" in text
    assert board.get("F9") is None and not board.corroborate("F9", "x")


# ---------------- 查證員輸出解析 ----------------
def test_fact_check_parses_claims_and_builds_feedback():
    class Checker:
        async def chat(self, messages, **kw):
            return Reply(content=json.dumps({"claims": [
                {"text": "票價 ¥680", "verdict": "supported", "evidence": "E1"},
                {"text": "另有 ¥999 優惠", "verdict": "unsupported", "evidence": ""},
                {"text": "航程 2 小時", "verdict": "CONFLICT", "evidence": "E1 顯示 2 小時 15 分"},
            ]}, ensure_ascii=False))

    report = asyncio.run(fact_check(Checker(), "查票價", "票價 ¥680，另有 ¥999 優惠", "E1 ¥680", 1))
    assert not report.passed and len(report.supported) == 1 and len(report.problems) == 2
    assert "[無據] 另有 ¥999 優惠" in report.feedback() and "[矛盾] 航程 2 小時" in report.feedback()
    assert "⚠ 未能查證：另有 ¥999 優惠" in report.appendix(2)


def test_fact_check_without_evidence_or_with_garbage_does_not_block():
    report = asyncio.run(fact_check(None, "t", "a", "   ", 1))
    assert report.passed and "沒有收集到任何證據" in report.skipped

    class Garbage:
        async def chat(self, messages, **kw):
            return Reply(content="我覺得都對")

    report = asyncio.run(fact_check(Garbage(), "t", "a", "E1 x", 1))
    assert report.passed and "無法解析" in report.appendix(1)


def test_all_supported_appendix_lists_evidence():
    report = FactCheck(1, [Claim("A", "supported", "F1"), Claim("B", "supported", "E2")])
    assert report.passed and "✔ 2 條陳述有證據支持（E2、F1）" in report.appendix(1)


# ---------------- 內核：退回 → 補查 → 放行 ----------------
def _kernel(settings, llm):
    library = PlaybookLibrary.load([settings.path(d) for d in settings.playbook_dirs])
    return Kernel(settings, llm, SkillSet(builtin_skills()), library=library)


def _checker(claims):
    return reply(content=json.dumps({"claims": claims}, ensure_ascii=False))


def test_finish_is_rejected_until_claims_are_backed_by_evidence(settings):
    settings.verify.enabled = True
    llm = ScriptedLLM([
        reply(call("python_run", code="print('測試商品的價格是 ¥680')")),
        reply(call("finish", answer="價格是 ¥680，另外會員價 ¥599。")),
        _checker([{"text": "價格是 ¥680", "verdict": "supported", "evidence": "E1"},
                  {"text": "會員價 ¥599", "verdict": "unsupported"}]),
        reply(call("finish", answer="價格是 ¥680（會員價未查到公開資料）。")),
        _checker([{"text": "價格是 ¥680", "verdict": "supported", "evidence": "E1"}]),
    ])
    result = asyncio.run(_kernel(settings, llm).run("查一下測試商品現在的價格"))

    assert result.status == "success"
    assert result.answer.startswith("價格是 ¥680（會員價未查到公開資料）。")
    assert "查證摘要（反覆查證 2 輪）：✔ 1 條陳述有證據支持（E1）" in result.answer
    assert result.verification["passed"] and result.verification["rounds"] == 2

    events = Journal.load(next(settings.runs_dir.iterdir()))
    verdicts = [(e.data["round"], e.data["action"]) for e in events if e.kind == "verify"]
    assert verdicts == [(1, "reject"), (2, "accept")]
    # 被退回時，模型在下一輪看到了具體的問題清單
    rejected = [m for m in llm.requests[3][0] if m.role == "tool"][-1].content
    assert "未通過第 1 輪查證" in rejected and "會員價 ¥599" in rejected
    # 查證員拿到的證據裡確實有 python_run 的輸出
    assert "¥680" in llm.requests[2][0][-1].content


def test_rounds_are_capped_and_remaining_problems_are_disclosed(settings):
    settings.verify.enabled = True
    settings.verify.max_rounds = 0  # 不退回，只標註
    llm = ScriptedLLM([
        reply(call("python_run", code="print('今天氣溫 25 度')")),
        reply(call("finish", answer="今天氣溫 25 度，明天會下雨。")),
        _checker([{"text": "氣溫 25 度", "verdict": "supported", "evidence": "E1"},
                  {"text": "明天會下雨", "verdict": "unsupported"}]),
    ])
    result = asyncio.run(_kernel(settings, llm).run("查一下今天的天氣"))
    assert "⚠ 未能查證：明天會下雨" in result.answer
    assert result.verification["passed"] is False and result.verification["problems"] == 1


def test_verification_is_skipped_for_profiles_not_listed(settings):
    settings.verify.enabled = True
    llm = ScriptedLLM([reply(call("finish", answer="Hello"))])
    result = asyncio.run(_kernel(settings, llm).run("把這段話翻譯成英文"))  # general 模式不查證
    assert result.answer == "Hello" and result.verification is None
    assert len(llm.requests) == 1
