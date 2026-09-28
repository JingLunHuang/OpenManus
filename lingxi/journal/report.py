"""把黑匣子渲染成一份可離線開啟的時間線報告（report.html）。

用途：復盤失敗原因、在作品集裡展示一次完整執行，不需要重新呼叫任何模型。
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from lingxi.journal.console import render as console_line
from lingxi.journal.recorder import Event, Journal

_CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1da;
--ok:#1f7a4d;--bad:#b3261e;--warn:#9a6700;--acc:#2f5bb7}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d21;--ink:#ececf0;
--muted:#9b9ba3;--line:#2c2c33;--ok:#5cc28d;--bad:#ff8a80;--warn:#e3b341;--acc:#8fb0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}.meta{color:var(--muted);font-size:13px;margin-bottom:24px}
.step{border-left:2px solid var(--line);margin-left:8px;padding:0 0 18px 18px;position:relative}
.step:before{content:"";position:absolute;left:-7px;top:4px;width:12px;height:12px;
border-radius:50%;background:var(--acc)}
.step h2{font-size:14px;margin:0 0 6px;color:var(--muted);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:6px 0}
.think{color:var(--muted);white-space:pre-wrap}.call{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.tag{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:999px;padding:0 8px;margin-left:6px}
img{max-width:100%;border-radius:8px;border:1px solid var(--line);margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;margin:0}
.answer{border-left:3px solid var(--ok)}
"""


def _e(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(value)


def render_report(events: list[Event], title: str = "靈犀執行報告") -> str:
    head = next((e for e in events if e.kind == "run.start"), None)
    finish = next((e for e in reversed(events) if e.kind == "run.finish"), None)
    started = datetime.fromtimestamp(events[0].ts).strftime("%Y-%m-%d %H:%M:%S") if events else ""
    duration = f"{events[-1].ts - events[0].ts:.1f}s" if len(events) > 1 else "-"

    parts = [f"<h1>{_e(head.data.get('task', title)) if head else title}</h1>"]
    meta = [f"開始 {started}", f"耗時 {duration}"]
    if finish:
        u = finish.data.get("usage", {})
        meta += [f"狀態 {finish.data.get('status')}", f"{finish.step} 步",
                 f"{u.get('calls', 0)} 次模型呼叫", f"{u.get('total_tokens', 0)} tokens"]
        kv = finish.data.get("kv") or {}
        if kv:
            meta.append(f"KV 前綴重用 {kv.get('prefix_reuse', 0):.0%}")
    parts.append(f"<div class='meta'>{' · '.join(_e(m) for m in meta)}</div>")

    current_step = -1
    for ev in events:
        d = ev.data
        if ev.kind in ("intent", "playbook", "temporal") and current_step < 1:
            text = {"intent": f"任務模式：{d.get('profile')}（{d.get('reason', '')}）",
                    "playbook": f"命中手冊：{d.get('title')} {d.get('route') or ''}",
                    "temporal": "時間錨定：" + "；".join(f"{a['text']}→{a['date']}" for a in d.get("anchors", []))}[ev.kind]
            parts.append(f"<div class='card'>{_e(text)}</div>")
        elif ev.kind == "step":
            if current_step >= 1:
                parts.append("</div>")
            current_step = ev.step
            parts.append(f"<div class='step'><h2>第 {ev.step} 步 · 剩餘預算 {d.get('remaining')}</h2>")
        elif ev.kind == "think":
            body = ""
            if d.get("content"):
                body += f"<div class='think'>{_e(d['content'])}</div>"
            for call in d.get("calls", []):
                body += f"<div class='call'>→ {_e(call['name'])} {_e(call.get('args', {}))}</div>"
            if body:
                parts.append(f"<div class='card'>{body}</div>")
        elif ev.kind == "outcome":
            cls = "ok" if d.get("ok") else "bad"
            tags = ""
            if d.get("grounding"):
                tags += f"<span class='tag'>定位：{_e(d['grounding'].get('method'))}</span>"
            if d.get("verified") is True:
                tags += "<span class='tag ok'>已驗證</span>"
            elif d.get("verified") is False:
                tags += "<span class='tag warn'>未驗證</span>"
            imgs = "".join(f"<img src='{_e(a)}' alt='截圖'>" for a in d.get("artifacts", []) if a.endswith((".png", ".jpg")))
            parts.append(f"<div class='card'><span class='{cls}'>{'✔' if d.get('ok') else '✘'} "
                         f"{_e(d.get('name'))}</span>{tags}<pre>{_e(d.get('summary', ''))}</pre>{imgs}</div>")
        elif ev.kind == "verify":
            action = "退回補查" if d.get("action") == "reject" else "放行"
            rows = "".join(
                f"<div class='{'ok' if c['verdict'] == 'supported' else 'bad'}'>"
                f"{'✔' if c['verdict'] == 'supported' else '✘'} {_e(c['text'])}"
                f"{' · ' + _e(c['evidence']) if c.get('evidence') else ''}</div>" for c in d.get("claims", []))
            parts.append(f"<div class='card'><strong>🔍 反覆查證 第 {d.get('round')} 輪 → {action}</strong>"
                         f"<div class='meta'>✔ {d.get('supported', 0)} 條有據 · ✘ {d.get('problems', 0)} 條待補"
                         f"{' · ' + _e(d['skipped']) if d.get('skipped') else ''}</div>{rows}</div>")
        elif ev.kind in ("guard", "error"):
            parts.append(f"<div class='card warn'>⚠ {_e(d.get('message'))}</div>")
        elif ev.kind in ("memory", "cache", "focus", "evolve"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", console_line(ev) or "")  # 終端機著色碼不進 HTML
            if line.strip():
                parts.append(f"<div class='meta'>{_e(line.strip())}</div>")
        elif ev.kind == "budget" and d.get("delta"):
            parts.append(f"<div class='meta'>預算 {d['delta']:+d}（{_e(d.get('reason'))}）→ 剩餘 {d.get('remaining')}</div>")
    if current_step >= 1:
        parts.append("</div>")
    if finish:
        parts.append(f"<div class='card answer'><strong>最終答覆</strong><pre>{_e(finish.data.get('answer', ''))}</pre></div>")

    return (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")


def write_report(run_dir: Path) -> Path:
    events = Journal.load(run_dir)
    out = run_dir / "report.html"
    out.write_text(render_report(events), encoding="utf-8")
    return out
