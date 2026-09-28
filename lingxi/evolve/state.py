"""系統狀態（System State）：RSI 每一輪真正「被繼承」的東西，按版本存放、可回滾。

    evolve/
      current            目前生效的版本號
      versions/v0003.json  {version, parent, harness, playbooks, strategy, scores, note}
      playbooks/v0003/     這個版本生效的「學到的手冊」（整份複製，版本之間互不影響）
      ledger.jsonl         每一輪的完整紀錄（見 rsi.py）

harness 只允許調整 TUNABLE 裡列出的參數，範圍也寫死在這裡——
「能改什麼、能改多少」屬於外部規定，改進器不能自己放寬（論文所說的 authority boundary）。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TUNABLE: dict[str, tuple[type, float, float]] = {
    "agent.fold_after_turns": (int, 4, 10),
    "kv_cache.fold_block": (int, 1, 8),
    "agent.max_observation_chars": (int, 2000, 8000),
    "retrieval.dense_weight": (float, 0.2, 2.0),
    "retrieval.bm25_weight": (float, 0.0, 2.0),
    "memory.min_score": (float, 0.05, 0.6),
}


@dataclass
class SystemState:
    version: int = 0
    parent: int | None = None
    created: float = field(default_factory=time.time)
    harness: dict[str, Any] = field(default_factory=dict)
    playbooks: list[str] = field(default_factory=list)  # 來源程序技能的節點編號（P3…）
    strategy: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    note: str = ""


def read_setting(settings, key: str) -> Any:
    section, name = key.split(".", 1)
    return getattr(getattr(settings, section), name)


def clamp(key: str, value: Any) -> Any:
    kind, lo, hi = TUNABLE[key]
    value = min(max(value, lo), hi)
    return int(round(value)) if kind is int else round(float(value), 3)


def apply_harness(settings, harness: dict[str, Any]):
    """回傳套用了 harness 參數的設定副本（原設定不變）。"""
    s = settings.model_copy(deep=True)
    for key, value in harness.items():
        if key not in TUNABLE:
            continue  # 不在白名單裡的參數一律忽略
        section, name = key.split(".", 1)
        setattr(getattr(s, section), name, clamp(key, value))
    return s


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        (root / "versions").mkdir(parents=True, exist_ok=True)
        (root / "playbooks").mkdir(parents=True, exist_ok=True)

    def _path(self, version: int) -> Path:
        return self.root / "versions" / f"v{version:04d}.json"

    def playbook_dir(self, version: int) -> Path:
        d = self.root / "playbooks" / f"v{version:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, version: int) -> SystemState | None:
        p = self._path(version)
        return SystemState(**json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else None

    def current(self) -> SystemState:
        marker = self.root / "current"
        if marker.is_file():
            state = self.get(int(marker.read_text().strip() or 0))
            if state:
                return state
        state = self.get(0) or SystemState(version=0, note="初始版本（v0）")
        if not self._path(0).is_file():
            self.save(state)
        return state

    def history(self) -> list[SystemState]:
        return [SystemState(**json.loads(p.read_text(encoding="utf-8")))
                for p in sorted((self.root / "versions").glob("v*.json"))]

    def save(self, state: SystemState) -> None:
        self._path(state.version).write_text(json.dumps(asdict(state), ensure_ascii=False, indent=1),
                                             encoding="utf-8")

    def commit(self, parent: SystemState, harness: dict[str, Any], new_playbooks: dict[str, str],
               strategy: dict[str, Any], scores: dict[str, float], note: str) -> SystemState:
        """產生繼任版本：沿用父版本的手冊，再加入新接受的手冊；設為目前版本。"""
        version = max((s.version for s in self.history()), default=0) + 1
        dst = self.playbook_dir(version)
        for f in self.playbook_dir(parent.version).glob("*.toml"):
            shutil.copy2(f, dst / f.name)
        for name, text in new_playbooks.items():
            (dst / name).write_text(text, encoding="utf-8")
        state = SystemState(version=version, parent=parent.version, harness=harness,
                            playbooks=parent.playbooks + [n.split(".")[0] for n in new_playbooks],
                            strategy=strategy, scores=scores, note=note)
        self.save(state)
        self.set_current(version)
        return state

    def set_current(self, version: int) -> None:
        if not self._path(version).is_file():
            raise ValueError(f"沒有版本 v{version}")
        (self.root / "current").write_text(str(version))
