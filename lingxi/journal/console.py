"""把黑匣子事件即時渲染到終端（控制台只是 Journal 的一個訂閱者）。"""

from __future__ import annotations

import json
import os
import sys

from lingxi.journal.recorder import Event

_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
if _COLOR and os.name == "nt":
    os.system("")  # 開啟 Windows 終端的 ANSI 支援


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(event: Event) -> str | None:
    d = event.data
    k = event.kind
    if k == "run.start":
        return _c("1;36", f"◆ 靈犀啟動  任務：{d.get('task')}")
    if k == "intent":
        return _c("36", f"  ├ 任務模式：{d.get('profile')}（{d.get('reason', '')}）")
    if k == "temporal":
        anchors = "；".join(f"{a['text']}→{a['date']}" for a in d.get("anchors", []))
        return _c("36", f"  ├ 時間錨定：{anchors}") if anchors else None
    if k == "playbook":
        return _c("36", f"  ├ 命中手冊：{d.get('title')}  {d.get('route') or ''}".rstrip())
    if k == "step":
        return _c("1;34", f"\n── 第 {event.step} 步 · 剩餘預算 {d.get('remaining')} ──")
    if k == "think":
        lines = []
        if d.get("content"):
            lines.append(_c("37", f"  💭 {_short(d['content'], 300)}"))
        for call in d.get("calls", []):
            lines.append(_c("33", f"  → {call['name']} {_short(call.get('args', {}), 140)}"))
        return "\n".join(lines) or None
    if k == "outcome":
        mark = "✔" if d.get("ok") else "✘"
        color = "32" if d.get("ok") else "31"
        verified = {True: " [已驗證]", False: " [未驗證]"}.get(d.get("verified"), "")
        via = f" ⟨{d['grounding']['method']}⟩" if d.get("grounding") else ""
        return _c(color, f"  {mark} {d.get('name')}{via}{verified}：{_short(d.get('summary', ''), 220)}")
    if k == "budget" and d.get("delta"):
        sign = "+" if d["delta"] > 0 else ""
        return _c("35", f"  ⏱ 預算 {sign}{d['delta']}（{d.get('reason')}）→ 剩餘 {d.get('remaining')}")
    if k == "verify":
        verdict = "退回補查" if d.get("action") == "reject" else "放行"
        text = f"  🔍 反覆查證 第 {d.get('round')} 輪：✔ {d.get('supported', 0)} 條有據 · ✘ {d.get('problems', 0)} 條待補 → {verdict}"
        if d.get("skipped"):
            text += f"（{d['skipped']}）"
        return _c("1;36" if d.get("action") == "accept" else "1;33", text)
    if k == "findings":
        s = d.get("stats", {})
        return _c("36", f"  📌 發現板：✔ 多方印證 {s.get('corroborated', 0)} · ○ 單一來源 {s.get('single', 0)} · ⚠ 矛盾 {s.get('conflict', 0)}")
    if k == "guard":
        return _c("1;35", f"  ⚠ {d.get('message')}")
    if k == "human.ask":
        return _c("1;33", f"  ❓ 需要你的輸入：{d.get('question')}")
    if k == "error":
        return _c("1;31", f"  ✘ 錯誤：{d.get('message')}")
    if k == "run.finish":
        usage = d.get("usage", {})
        head = _c("1;32" if d.get("status") == "success" else "1;33",
                  f"\n◆ 結束（{d.get('status')}）· {event.step} 步 · "
                  f"{usage.get('calls', 0)} 次模型呼叫 · {usage.get('total_tokens', 0)} tokens")
        return f"{head}\n{d.get('answer', '')}"
    return None


def console_listener(event: Event) -> None:
    text = render(event)
    if text:
        print(text, flush=True)
