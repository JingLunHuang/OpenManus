"""遞迴自我改進（RSI）：受保護評測的隔離與預算、接受 / 拒絕 / 繼承 / 回滾、手冊匯出、門面套用系統版本。"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from conftest import ScriptedLLM, call, reply
from test_memory import flight_run

import lingxi.evolve.improver as improver_module
from lingxi.agent import LingXi
from lingxi.evolve import BudgetExceeded, ProtectedEvaluator, RSILoop, StateStore, apply_harness
from lingxi.evolve.improver import Strategy, export_playbook
from lingxi.journal.recorder import Journal
from lingxi.knowledge.playbook import Playbook, PlaybookLibrary
from lingxi.memory import MemorySystem
from lingxi.retrieval import HashEmbedder


def test_improver_cannot_see_the_protected_suites():
    source = inspect.getsource(improver_module)
    assert "protected" not in source.replace("不匯入 protected.py", "")  # 改進器不匯入評測模組
    ev = ProtectedEvaluator([], query_budget=1)
    assert not any(not k.startswith("_") and "suite" in k for k in vars(ev))  # 評測資料只存在私有屬性裡
    assert len(ev.epoch) == 12


def test_protected_budget_is_enforced(settings):
    ev = ProtectedEvaluator([settings.path(d) for d in settings.playbook_dirs], query_budget=2)
    ev.score("routing", {})
    ev.score("context", {"agent.fold_after_turns": 6, "kv_cache.fold_block": 4})
    with pytest.raises(BudgetExceeded):
        ev.score("retrieval", {})


def test_context_suite_prefers_block_folding_and_enforces_retention_floor(settings):
    ev = ProtectedEvaluator([], query_budget=10)
    base = {"agent.fold_after_turns": 6, "agent.max_observation_chars": 6000}
    assert ev.score("context", {**base, "kv_cache.fold_block": 4}) > ev.score("context", {**base, "kv_cache.fold_block": 1}) + 0.05
    assert ev.score("context", {**base, "agent.fold_after_turns": 3, "kv_cache.fold_block": 4}) == 0.0


def test_apply_harness_clamps_and_ignores_unknown_keys(settings):
    s = apply_harness(settings, {"kv_cache.fold_block": 99, "memory.min_score": 0.001, "llm.model": "evil"})
    assert s.kv_cache.fold_block == 8 and s.memory.min_score == 0.05 and s.llm.model == settings.llm.model
    assert settings.kv_cache.fold_block == 4  # 原設定不變


def test_strategy_flips_direction_and_halves_step_after_rejection():
    st = Strategy()
    st.record("context", "kv_cache.fold_block", accepted=False)
    assert st.dirs["kv_cache.fold_block"] == -1 and st.steps["kv_cache.fold_block"] == 1
    st.record("retrieval", "memory.min_score", accepted=True)
    assert st.steps["memory.min_score"] == pytest.approx(0.075)
    assert st.priority("retrieval", 1.0) > st.priority("context", 1.0)


def test_rsi_round_inherits_improvements_and_can_roll_back(settings):
    mem = MemorySystem(settings, embedder=HashEmbedder(256))
    for src, run_id in (("上海", "r1"), ("廣州", "r2")):
        mem.write_run(flight_run(src, "北京", "6月26日").events, run_id=run_id)
    asyncio.run(mem.sleep("rules"))
    loop = RSILoop(settings, mem.graph)
    report = loop.round()

    assert report.new_version == 1 and report.baseline and report.final
    accepted = [c for c in report.candidates if c["accepted"]]
    assert {c["target"] for c in accepted} >= {"playbook", "retrieval"}
    assert report.final["retrieval"] > report.baseline["retrieval"]  # 檢索參數的改進通過受保護評測
    assert report.final["routing"] == report.baseline["routing"]  # 學到的手冊沒有搶走任何不該命中的任務
    assert report.hci["retrieval"] > 0 and report.budget["used"] <= settings.evolve.query_budget
    assert "外部" in report.autonomy["decisions"]["verifier"] and report.autonomy["level"].startswith("L")

    store = StateStore(settings.evolve_dir)
    v1 = store.current()
    assert v1.version == 1 and v1.parent == 0 and v1.playbooks
    book = Playbook.from_toml(next(store.playbook_dir(1).glob("*.toml")))
    assert book.id.startswith("learned-") and PlaybookLibrary([book]).match("在測試頁查詢 8月3日 從成都到北京的航班", [])
    ledger = [json.loads(line) for line in (settings.evolve_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["new_version"] == 1

    loop.rollback(0)
    assert store.current().version == 0
    assert loop.rounds()[-1]["rollback"] == 0


def test_rejected_candidates_leave_the_system_unchanged(settings):
    settings.evolve.min_gain = 5.0  # 不可能達成的門檻：所有參數候選都應被拒絕
    report = RSILoop(settings).round()
    assert report.new_version is None and report.candidates
    assert all(not c["accepted"] for c in report.candidates)
    assert StateStore(settings.evolve_dir).current().version == 0
    strategy = json.loads((settings.evolve_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["targets"]["context"]["tried"] >= 1  # 被拒絕的經驗仍然改寫了策略


def test_export_playbook_needs_shared_phrases(settings):
    mem = MemorySystem(settings, embedder=HashEmbedder(256))
    for src, run_id in (("上海", "r1"), ("廣州", "r2")):
        mem.write_run(flight_run(src, "北京", "6月26日").events, run_id=run_id)
    asyncio.run(mem.sleep("rules"))
    proc = mem.graph.active("procedural")[0]
    toml, tasks = export_playbook(proc, 3)
    assert "由 RSI 自動產生（靈犀 v3）" in toml and len(tasks) == 2
    proc.data["tasks"] = ["寫一首詩", "查天氣"]
    assert export_playbook(proc, 3) is None


def test_lingxi_applies_the_active_version_and_sleeps_on_schedule(settings):
    store = StateStore(settings.evolve_dir)
    store.commit(store.current(), {"kv_cache.fold_block": 6}, {}, {}, {}, "測試用版本")
    settings.memory.sleep_every = 1
    llm = ScriptedLLM([reply(call("plan", action="show"), content="頁面要先登入才能看到資料。"),
                       reply(call("finish", answer="完成"))])
    journal = Journal(None)
    agent = LingXi(settings, llm=llm, console=False)
    asyncio.run(agent.run("整理資料", journal=journal))
    evolve = next(e for e in journal.events if e.kind == "evolve")
    finish = next(e for e in journal.events if e.kind == "run.finish")
    assert evolve.data["version"] == 1 and finish.data["kv"]["fold_block"] == 6
    assert any(e.kind == "memory" and e.data.get("action") == "write" for e in journal.events)
    assert agent.last_sleep is not None and agent.last_sleep["mode"] == "rules"  # 注入的假模型不會被拿去做睡眠整理
    assert len(llm.requests) == 2
