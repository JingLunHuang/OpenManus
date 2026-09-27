"""反覆查證（Iterative Fact-Check）：答覆在交給使用者之前，必須先過一道證據閘門。

流程：
  模型呼叫 finish
    → 查證員把答覆拆成可查證的事實陳述（數字、價格、日期、名稱、排名、引用）
    → 逐條對照「證據」：發現板（F 編號，含多源印證狀態）＋ 最近的頁面與工具輸出（E 編號）
    → 全部有據：放行，並在答覆末尾附上查證摘要
    → 有「無據 / 矛盾」且還有預算與輪次：退回 finish，把問題清單交給模型補查或修正，然後再查一輪
    → 預算用盡的最後一步：不退回，但把未能查證的陳述明確標註在答覆裡

查證員只根據證據判斷、不使用自身知識，因此它檢查的是「答覆有沒有根據」，而不是「答覆對不對」
—— 前者可以被程式驗證，後者需要證據本身可靠，這正是發現板多源印證要解決的事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lingxi.llm.messages import Message
from lingxi.utils import clip, extract_json

_SYSTEM = "你是嚴格的事實查證員。只根據使用者提供的證據判斷，絕不使用你自己的知識補充或推測。"

_PROMPT = """任務：{task}

【待查證的答覆】
{answer}

【證據】（F = 發現板上的事實，附印證狀態與來源；E = 最近的頁面內容與工具輸出）
{evidence}

請找出答覆中所有「可查證的事實性陳述」（數字、價格、日期、時間、名稱、排名、引用的說法），最多 12 條，逐條判定：
- supported：證據中有明確支持，evidence 填證據編號（如 F2、E1）
- unsupported：證據中找不到依據（可能來自模型自身知識或推測）
- conflict：與證據矛盾，evidence 說明矛盾之處
意見、建議、段落標題等非事實內容不要列出。只輸出 JSON：
{{"claims": [{{"text": "陳述原文", "verdict": "supported|unsupported|conflict", "evidence": "編號或說明"}}]}}"""


@dataclass
class Claim:
    text: str
    verdict: str
    evidence: str = ""


@dataclass
class FactCheck:
    round: int
    claims: list[Claim] = field(default_factory=list)
    skipped: str = ""  # 查證本身沒能完成的原因

    @property
    def supported(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict == "supported"]

    @property
    def problems(self) -> list[Claim]:
        return [c for c in self.claims if c.verdict in ("unsupported", "conflict")]

    @property
    def passed(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "passed": self.passed,
            "supported": len(self.supported),
            "problems": len(self.problems),
            "skipped": self.skipped,
            "claims": [c.__dict__ for c in self.claims],
        }

    def feedback(self) -> str:
        lines = [f"第 {self.round} 輪查證：{len(self.supported)} 條有據，{len(self.problems)} 條有問題。"]
        for c in self.problems:
            tag = "無據" if c.verdict == "unsupported" else "矛盾"
            lines.append(f"  ✘ [{tag}] {c.text}" + (f" —— {c.evidence}" if c.evidence else ""))
        lines.append("請擇一處理後再呼叫 finish：① 用 web_search / web_open / web_read 找到來源補證；"
                     "② 修正與證據矛盾的內容；③ 刪除無法查證的陳述，或明確標註為「推測 / 未經查證」。")
        return "\n".join(lines)

    def appendix(self, rounds: int) -> str:
        if self.skipped:
            return f"\n\n---\n查證摘要：{self.skipped}"
        if not self.claims:
            return ""
        head = f"\n\n---\n查證摘要（反覆查證 {rounds} 輪）：✔ {len(self.supported)} 條陳述有證據支持"
        if self.passed:
            refs = "、".join(sorted({c.evidence for c in self.supported if c.evidence})[:8])
            return head + (f"（{refs}）" if refs else "") + "。"
        lines = [head + f" · ⚠ {len(self.problems)} 條未能查證："]
        for c in self.problems:
            tag = "未能查證" if c.verdict == "unsupported" else "與證據矛盾"
            lines.append(f"- ⚠ {tag}：{c.text}" + (f"（{c.evidence}）" if c.evidence and c.verdict == "conflict" else ""))
        return "\n".join(lines)


async def fact_check(llm, task: str, answer: str, evidence: str, round_no: int) -> FactCheck:
    if not evidence.strip():
        return FactCheck(round_no, skipped="本次運行沒有收集到任何證據，答覆中的事實陳述均未經查證。")
    try:
        reply = await llm.chat([Message.system(_SYSTEM),
                                Message.user(_PROMPT.format(task=task, answer=clip(answer, 6000), evidence=evidence))])
    except Exception as exc:
        return FactCheck(round_no, skipped=f"查證員呼叫失敗（{clip(str(exc), 80)}），答覆未經查證。")
    data = extract_json(reply.content)
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return FactCheck(round_no, skipped="查證員的輸出無法解析，本輪略過查證。")
    claims = []
    for item in data["claims"][:12]:
        if isinstance(item, dict) and item.get("text"):
            verdict = str(item.get("verdict", "unsupported")).strip().lower()
            if verdict not in ("supported", "unsupported", "conflict"):
                verdict = "unsupported"
            claims.append(Claim(str(item["text"]), verdict, str(item.get("evidence", ""))))
    return FactCheck(round_no, claims)
