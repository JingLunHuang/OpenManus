"""滾動記憶（Rolling Memory）：控制上下文體積。

OpenManus 把每一步的完整頁面元素列表、完整工具輸出都追加進訊息歷史，
步數越多，每次請求越長（課程裡"2 天用了 500 塊 token"）。

靈犀的做法：
  - 當前頁面狀態只出現在"本步簡報"裡，不寫進歷史（每步重新生成）；
  - 最近 fold_after 輪保留完整工具輸出（仍有截斷上限）；
  - 更早的輪次把工具輸出摺疊成一行摘要（Outcome.summary），思考內容截短；
  - tool_call 與 tool 訊息一一配對的結構始終保持合法。
關鍵事實不會因此丟失：它們在發現板 / 計劃板裡，每步都會重新渲染進簡報。

分段摺疊（fold_block，為 KV 快取而設計）：
  逐輪摺疊時，每一步都會改寫第 n-fold_after 輪，推理端的前綴快取從那一輪起全部失效。
  改成累積滿 fold_block 輪才一次摺疊 fold_block 輪：完整保留的輪數在
  [fold_after, fold_after + fold_block) 之間擺動，歷史在 fold_block 步之內只追加、不改寫。
  fold_block = 1 即舊行為。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from lingxi.llm.messages import Message
from lingxi.utils import clip


@dataclass
class Turn:
    messages: list[Message]


@dataclass
class RollingMemory:
    fold_after: int = 6
    max_observation_chars: int = 6000
    fold_block: int = 1
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, assistant: Message, tool_messages: list[Message], note: str | None = None) -> None:
        msgs = [assistant, *tool_messages]
        if note:
            msgs.append(Message.user(note))
        self.turns.append(Turn(msgs))

    def folded_count(self) -> int:
        block = max(1, self.fold_block)
        return max(0, len(self.turns) - self.fold_after) // block * block

    def messages(self) -> list[Message]:
        out: list[Message] = []
        boundary = self.folded_count()
        for i, turn in enumerate(self.turns):
            folded = i < boundary
            for m in turn.messages:
                if m.role == "tool":
                    content = (f"[已摺疊] {m.digest}" if folded and m.digest
                               else clip(m.content, self.max_observation_chars))
                    out.append(replace(m, content=content))
                elif m.role == "assistant" and folded:
                    out.append(replace(m, content=clip(m.content, 200)))
                else:
                    out.append(m)
        return out

    def size(self) -> int:
        return sum(len(m.content) for m in self.messages())
