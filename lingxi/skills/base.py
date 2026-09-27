"""技能（Skill）契約。

與 OpenManus 手寫 JSON Schema 的 BaseTool 不同：
- 參數用 pydantic 模型宣告，Schema 自動生成、參數自動校驗，
  校驗失敗時把"哪個欄位錯了"原樣告訴模型，讓它自己改；
- 返回統一的 Outcome：除了文字，還攜帶 verified（是否經過事後驗證）、
  grounding（元素是怎麼定位到的）、progress（本步取得了哪些進展，用於動態預算）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from lingxi.kernel.context import RunContext


@dataclass
class Outcome:
    ok: bool
    summary: str
    detail: str = ""
    verified: bool | None = None
    grounding: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    image: str | None = None
    progress: list[str] = field(default_factory=list)
    finish: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fail(cls, summary: str, detail: str = "", **kw: Any) -> "Outcome":
        return cls(ok=False, summary=summary, detail=detail, **kw)

    def for_model(self, limit: int = 6000) -> str:
        """給模型看的觀察文字。未驗證的動作會被顯式標註，避免模型"以為成功了"。"""
        head = ("成功" if self.ok else "失敗") + "：" + self.summary
        if self.verified is False:
            head += "\n[注意] 動作已執行，但沒有觀察到預期變化，請核實後再繼續。"
        body = f"{head}\n{self.detail}".strip() if self.detail else head
        if len(body) > limit:
            body = body[:limit] + f"\n…（已截斷，原文 {len(body)} 字）"
        return body


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


class Skill(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def tool_schema(self) -> dict[str, Any]:
        params = _strip_titles(self.Params.model_json_schema())
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }

    async def invoke(self, ctx: "RunContext", raw_args: dict[str, Any]) -> Outcome:
        if "__invalid_json__" in raw_args:
            return Outcome.fail("參數不是合法 JSON", f"收到：{raw_args['__invalid_json__'][:300]}")
        try:
            params = self.Params.model_validate(raw_args)
        except ValidationError as exc:
            problems = "；".join(
                f"{'.'.join(str(p) for p in err['loc']) or '參數'}：{err['msg']}" for err in exc.errors()
            )
            return Outcome.fail("參數校驗失敗", problems)
        return await self.run(ctx, params)

    @abstractmethod
    async def run(self, ctx: "RunContext", params: Any) -> Outcome: ...

    async def close(self) -> None:
        """執行結束時釋放資源（瀏覽器、沙箱、MCP 連線等）。"""


class SkillSet:
    def __init__(self, skills: list[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for s in skills or []:
            self.add(s)

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        return [s.tool_schema() for n, s in self._skills.items() if only is None or n in only]

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())

    async def close(self) -> None:
        for s in self._skills.values():
            try:
                await s.close()
            except Exception:
                pass
