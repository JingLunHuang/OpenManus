"""本地技能：Python 執行、工作區檔案、計劃板、向人提問、結束任務。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from lingxi.hanzi import to_trad
from lingxi.skills.base import Outcome, Skill
from lingxi.utils import clip


# ---------------- Python 執行 ----------------
class PythonParams(BaseModel):
    code: str = Field(description="要執行的 Python 程式碼；用 print() 輸出結果")
    timeout: int = Field(60, ge=1, le=600, description="超時秒數")


class PythonRun(Skill):
    name = "python_run"
    description = ("在獨立子行程中執行 Python 程式碼（工作目錄為工作區），適合計算、資料處理、生成檔案。"
                   "只有 print() 的內容會被返回。")
    Params = PythonParams

    async def run(self, ctx, p: PythonParams) -> Outcome:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-X", "utf8", "-c", p.code,
            cwd=str(ctx.settings.workspace_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=p.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return Outcome.fail(f"執行超時（{p.timeout}s），行程已終止")
        stdout = out.decode("utf-8", "replace")
        stderr = err.decode("utf-8", "replace")
        if proc.returncode != 0:
            return Outcome.fail(f"退出碼 {proc.returncode}", clip(stderr or stdout, 4000))
        body = stdout or "（沒有輸出，記得用 print() 列印結果）"
        if stderr.strip():
            body += f"\n[stderr]\n{clip(stderr, 1000)}"
        return Outcome(ok=True, summary=clip(stdout.strip().splitlines()[-1], 120) if stdout.strip() else "執行完成",
                       detail=clip(body, 6000), verified=True)


# ---------------- 工作區檔案 ----------------
# 這些文字檔寫入時一律轉成繁體（程式碼檔不轉，避免改動字串常數的語意）
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".json"}

class FileParams(BaseModel):
    action: Literal["view", "write", "append", "replace", "list"] = Field(description="檔案操作")
    path: str = Field(".", description="相對工作區的路徑")
    content: str | None = Field(None, description="write/append 的內容")
    old: str | None = Field(None, description="replace：要被替換的原文（必須在檔案中唯一出現）")
    new: str | None = Field(None, description="replace：替換後的新文字")


class Files(Skill):
    name = "files"
    description = "讀寫工作區中的檔案（報告、資料、程式碼）。路徑被限制在工作區之內。"
    Params = FileParams

    @staticmethod
    def _safe(root: Path, rel: str) -> Path:
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise PermissionError(f"路徑越界：{rel} 不在工作區內")
        return target

    async def run(self, ctx, p: FileParams) -> Outcome:
        root = ctx.settings.workspace_dir.resolve()
        try:
            path = self._safe(root, p.path)
        except PermissionError as exc:
            return Outcome.fail(str(exc))
        rel = path.relative_to(root).as_posix() or "."
        if p.action == "list":
            if not path.is_dir():
                return Outcome.fail(f"{rel} 不是目錄")
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            listing = "\n".join(f"{'📄' if i.is_file() else '📁'} {i.name}" for i in items[:200])
            return Outcome(ok=True, summary=f"{rel} 下有 {len(items)} 項", detail=listing)
        if p.action == "view":
            if not path.is_file():
                return Outcome.fail(f"檔案不存在：{rel}")
            text = path.read_text(encoding="utf-8", errors="replace")
            numbered = "\n".join(f"{n:>4}│{line}" for n, line in enumerate(text.splitlines(), 1))
            return Outcome(ok=True, summary=f"{rel}（{len(text)} 字）", detail=clip(numbered, 8000))
        if p.action in ("write", "append"):
            if p.content is None:
                return Outcome.fail(f"{p.action} 需要 content")
            if ctx.settings.agent.traditional_output and path.suffix.lower() in _TEXT_SUFFIXES:
                p.content = to_trad(p.content)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a" if p.action == "append" else "w", encoding="utf-8") as f:
                f.write(p.content)
            verified = path.is_file() and p.content in path.read_text(encoding="utf-8")
            return Outcome(ok=True, summary=f"已{'追加' if p.action == 'append' else '寫入'} {rel}（{len(p.content)} 字）",
                           verified=verified, progress=["artifact"], data={"path": str(path)})
        # replace
        if not path.is_file():
            return Outcome.fail(f"檔案不存在：{rel}")
        if not p.old or p.new is None:
            return Outcome.fail("replace 需要 old 與 new")
        text = path.read_text(encoding="utf-8")
        if ctx.settings.agent.traditional_output and path.suffix.lower() in _TEXT_SUFFIXES:
            p.new = to_trad(p.new)
        count = text.count(p.old)
        if count != 1:
            return Outcome.fail(f"原文在檔案中出現了 {count} 次，必須恰好 1 次才能替換")
        path.write_text(text.replace(p.old, p.new), encoding="utf-8")
        return Outcome(ok=True, summary=f"已修改 {rel}", verified=True, progress=["artifact"])


# ---------------- 計劃板 ----------------
class PlanParams(BaseModel):
    action: Literal["set", "done", "note"] = Field(description="set=制定計劃；done=完成第 index 項；note=給第 index 項加備註")
    items: list[str] | None = Field(None, description="set 時的計劃條目（3~7 條為宜）")
    index: int | None = Field(None, description="條目編號（從 1 開始）")
    note: str | None = Field(None, description="備註或該項的結論")


class Plan(Skill):
    name = "plan"
    description = "維護任務計劃板。複雜任務先 set 一份計劃，每完成一項就 done，計劃會一直顯示在簡報裡。"
    Params = PlanParams

    async def run(self, ctx, p: PlanParams) -> Outcome:
        if p.action == "set":
            if not p.items:
                return Outcome.fail("set 需要 items")
            ctx.plan.set(p.items)
            return Outcome(ok=True, summary=f"已制定 {len(ctx.plan.items)} 項計劃")
        if p.index is None or not 1 <= p.index <= len(ctx.plan.items):
            return Outcome.fail(f"index 需要在 1~{len(ctx.plan.items)} 之間")
        item = ctx.plan.mark(p.index, done=(p.action == "done") or ctx.plan.items[p.index - 1].done, note=p.note or "")
        done, total = ctx.plan.progress
        return Outcome(ok=True, summary=f"計劃 {p.index}「{clip(item.text, 30)}」{'完成' if item.done else '已備註'}（{done}/{total}）",
                       progress=["plan_done"] if p.action == "done" else [])


# ---------------- 向人提問 ----------------
class AskParams(BaseModel):
    question: str = Field(description="要問使用者的問題（只在確實缺少關鍵資訊、或需要人工完成驗證/登入時使用）")


class AskHuman(Skill):
    name = "ask_human"
    description = "向使用者提問並等待回答。僅在缺少無法推斷的關鍵資訊，或需要使用者手動登入/完成驗證時使用。"
    Params = AskParams

    async def run(self, ctx, p: AskParams) -> Outcome:
        ctx.emit("human.ask", question=p.question)
        answer = await ctx.human.ask(p.question)
        ctx.emit("human.answer", answer=answer)
        return Outcome(ok=True, summary=f"使用者回答：{clip(answer, 120)}", detail=answer)


# ---------------- 結束 ----------------
class FinishParams(BaseModel):
    answer: str = Field(description="給使用者的最終答覆（完整、可直接閱讀；若生成了檔案請註明路徑）")
    status: Literal["success", "partial", "failed"] = Field("success", description="任務完成度")


class Finish(Skill):
    name = "finish"
    description = "任務完成（或無法繼續）時呼叫，提交最終答覆並結束執行。"
    Params = FinishParams

    async def run(self, ctx, p: FinishParams) -> Outcome:
        return Outcome(ok=True, summary=f"提交最終答覆（{p.status}）", detail=p.answer, finish=True,
                       data={"answer": p.answer, "status": p.status})


def local_skills() -> list[Skill]:
    return [PythonRun(), Files(), Plan(), AskHuman(), Finish()]
