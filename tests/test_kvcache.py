"""KV 快取：分段摺疊讓前綴只增不改、最後一步不換工具清單、顯式快取標記、廠商用量欄位解析。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from conftest import ScriptedLLM, call, reply

from lingxi.journal.recorder import Journal
from lingxi.kernel.loop import Kernel
from lingxi.kernel.memory import RollingMemory
from lingxi.llm.kvcache import PrefixMeter, read_cache_usage, replay
from lingxi.llm.messages import Message, ToolCall
from lingxi.skills import SkillSet, builtin_skills


def _turn(memory: RollingMemory, i: int) -> None:
    c = ToolCall(id=f"c{i}", name="files", arguments="{}")
    memory.add_turn(Message.assistant(f"想法{i}", [c]), [Message.tool(c, "觀察" * 50, digest=f"摘要{i}")])


def test_block_folding_keeps_history_append_only_between_folds():
    memory = RollingMemory(fold_after=3, max_observation_chars=1000, fold_block=4)
    snapshots = []
    for i in range(12):
        _turn(memory, i)
        snapshots.append([m.to_openai() for m in memory.messages()])
    assert [memory.folded_count() for _ in [0]] == [8]
    rewrites = [i for i in range(1, len(snapshots)) if snapshots[i][: len(snapshots[i - 1])] != snapshots[i - 1]]
    assert rewrites == [6, 10]  # 只在累積滿 4 輪時改寫一次，其餘步驟只追加
    every_step = RollingMemory(fold_after=3, max_observation_chars=1000, fold_block=1)
    for i in range(6):
        _turn(every_step, i)
    assert every_step.folded_count() == 3  # fold_block=1 即舊行為


def test_prefix_meter_and_replay_show_block_folding_saves_billing():
    meter = PrefixMeter()
    tools = [{"type": "function", "function": {"name": "t"}}]
    a = [{"role": "system", "content": "S"}, {"role": "user", "content": "1"}]
    meter.observe(a, tools)
    assert meter.observe(a[:1] + [{"role": "user", "content": "2"}], tools) > 0.5  # 工具定義＋系統提示沿用
    assert meter.observe(a, [{"type": "function", "function": {"name": "other"}}]) == 0.0  # 工具清單一變就全部失效
    trace = [3000] * 20
    old, new = replay(trace, 6, 1, 6000), replay(trace, 6, 4, 6000)
    assert new["reuse"] > old["reuse"] + 0.15 and new["billed_chars"] < old["billed_chars"]


def test_read_cache_usage_understands_each_vendor():
    dashscope = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=900, cache_creation_input_tokens=0))
    assert read_cache_usage(dashscope) == (900, 0)
    assert read_cache_usage({"prompt_cache_hit_tokens": 640, "prompt_cache_miss_tokens": 10}) == (640, 0)
    assert read_cache_usage({"cache_read_input_tokens": 5, "cache_creation_input_tokens": 7}) == (5, 7)
    assert read_cache_usage(None) == (0, 0)


def test_explicit_mode_marks_system_and_last_history_message(settings):
    settings.kv_cache.mode = "explicit"
    llm = ScriptedLLM([reply(call("plan", action="show")), reply(call("finish", answer="好"))])
    asyncio.run(Kernel(settings, llm, SkillSet(builtin_skills()), journal=Journal(None)).run("你好"))
    second = [m.to_openai() for m in llm.requests[1][0]]
    marked = [i for i, m in enumerate(second) if isinstance(m["content"], list)
              and m["content"][0].get("cache_control") == {"type": "ephemeral"}]
    assert marked[0] == 0 and marked[-1] == len(second) - 2  # 系統提示 ＋ 歷史最後一則；本步簡報不標記


def test_kernel_reports_prefix_reuse_and_cached_tokens(settings):
    def long_run(messages, tools):
        return reply(call("files", action="list", path="."), content="繼續查看")

    llm = ScriptedLLM([], fallback=long_run)
    journal = Journal(None)
    result = asyncio.run(Kernel(settings, llm, SkillSet(builtin_skills()), journal=journal).run("整理工作區"))
    # 模型連最後一步（tool_choice 已指定 finish）都不 finish：必須照樣結束，不能因懲罰把預算補回來而無限迴圈
    assert result.status == "partial" and result.steps <= 24
    finish = next(e for e in journal.events if e.kind == "run.finish")
    kv = finish.data["kv"]
    assert kv["fold_block"] == settings.kv_cache.fold_block and kv["prefix_reuse"] > 0.5
    thinks = [e for e in journal.events if e.kind == "think"]
    assert all("reuse" in e.data and "cached" in e.data for e in thinks)
