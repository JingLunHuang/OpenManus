"""預處理（Preflight）：進入主迴圈之前、完全確定性（預設不呼叫模型）的三件事。

  意圖路由（初判）→ 時間錨定 → 手冊匹配與編譯 →（手冊可改寫模式與日期偏好）→ 定稿

核心和 `lingxi playbooks "任務"` 試跑命令共用這一個函式，保證"試跑看到的就是實際執行的"。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from lingxi.kernel.intent import route
from lingxi.kernel.profiles import Profile
from lingxi.knowledge.playbook import CompiledPlaybook, PlaybookLibrary
from lingxi.knowledge.temporal import DateAnchor, TemporalAnchor
from lingxi.llm.client import ChatModel


@dataclass
class Preflight:
    profile: Profile
    reason: str
    scores: dict[str, int]
    preference: str
    anchors: list[DateAnchor]
    briefed: str
    playbooks: list[CompiledPlaybook]


async def preflight(task: str, library: PlaybookLibrary, llm: ChatModel | None = None,
                    use_llm: bool = False, now: datetime | None = None) -> Preflight:
    first = await route(task, None, llm, use_llm=False)
    pref = first.profile.date_preference
    anchors = TemporalAnchor(now, prefer=pref).find(task)
    playbooks = library.match(task, anchors)

    decided = await route(task, playbooks, llm, use_llm=use_llm)
    top = playbooks[0].playbook if playbooks else None
    new_pref = (top.date_preference if top and top.date_preference else decided.profile.date_preference)
    if new_pref != pref:  # 偏好變了：重新錨定並重新編譯手冊
        pref = new_pref
        anchors = TemporalAnchor(now, prefer=pref).find(task)
        playbooks = library.match(task, anchors)

    briefed, anchors = TemporalAnchor(now, prefer=pref).annotate(task)
    profile = decided.profile
    if profile.date_preference != pref:
        profile = replace(profile, date_preference=pref)
    return Preflight(profile, decided.reason, decided.scores, pref, anchors, briefed, playbooks)
