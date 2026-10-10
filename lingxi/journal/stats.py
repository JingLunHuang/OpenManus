"""穩定性迴歸看板：把 runs/ 下的黑匣子彙總成「按任務」的數字。

動機（規劃書 W5）：把「穩定性」量化——同一任務跑三次都成功，應該是一個可展示的數字，
而不是一句形容詞。這裡完全離線：只讀 events.jsonl，不呼叫任何模型。

三層 API：
  summarize_run()  單次執行 → 一列摘要（run.finish 是主要來源）
  collect_runs()   掃整個 runs/ 目錄
  group_by_task()  按任務文字分組 → 成功率、步數中位數、模型呼叫數、token、KV 重用率、耗時

CLI（`lingxi stats`）與 Web（`/api/stats`）都只是這三層的不同呈現。
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from statistics import median
from typing import Any

from lingxi.journal.recorder import Journal

# 沒有可讀紀錄時的一致性訊息（CLI 與 Web 共用同一句，避免兩邊說法不同）
EMPTY_MESSAGE = "還沒有執行記錄：先用 lingxi run「你的任務」跑一次，看板就會有數字。"

# run.finish 出現過這些狀態才算「有結果」；未結束的執行不進成功率分母
DONE_STATUSES = ("success", "partial", "failed")


# ---------------- 單次執行 ----------------
def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    """讀一個 run 目錄，返回一列摘要；讀不到或沒有事件時返回 None。"""
    try:
        events = Journal.load(run_dir)
    except Exception:  # 半寫入的 jsonl、權限問題都不該讓看板報錯
        return None
    if not events:
        return None
    start = next((e for e in events if e.kind == "run.start"), None)
    finish = next((e for e in reversed(events) if e.kind == "run.finish"), None)
    usage = (finish.data.get("usage") or {}) if finish else {}
    kv = (finish.data.get("kv") or {}) if finish else {}
    verify = (finish.data.get("verify") or {}) if finish else {}
    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "task": (start.data.get("task") if start else None) or run_dir.name,
        "profile": start.data.get("profile") if start else None,
        "status": finish.data.get("status") if finish else "unfinished",
        "steps": finish.step if finish else None,
        "calls": usage.get("calls"),
        "tokens": usage.get("total_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": usage.get("cached_tokens"),
        # KV 前綴重用率：舊紀錄可能沒有 kv 欄位，缺就留 None（不當成 0）
        "kv_reuse": kv.get("prefix_reuse") if "prefix_reuse" in kv else None,
        "seconds": finish.data.get("seconds") if finish else None,
        "verify": None if not verify else ("passed" if verify.get("passed") else "flagged"),
        "started_at": round(events[0].ts, 3),
    }
    return row


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    """掃 runs/ 下所有含 events.jsonl 的目錄，依 run_id（即時間戳）由新到舊。"""
    if not runs_root or not Path(runs_root).is_dir():
        return []
    rows = []
    for path in sorted((p for p in Path(runs_root).iterdir() if (p / "events.jsonl").exists()),
                       key=lambda p: p.name, reverse=True):
        row = summarize_run(path)
        if row:
            rows.append(row)
    return rows


# ---------------- 按任務聚合 ----------------
def _median(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(float(median(nums)), 2) if nums else None


def group_by_task(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐次執行按「任務文字」分組，算出穩定性指標。

    成功率的分母只算有結果的執行（success/partial/failed），進行中或未結束的不算。
    執行次數越多、越近的任務排在前面（同次數時以最近一次執行時間排序）。
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        buckets.setdefault((row.get("task") or "").strip() or row["run_id"], []).append(row)

    groups = []
    for task, rows in buckets.items():
        done = [r for r in rows if r["status"] in DONE_STATUSES]
        ok = [r for r in done if r["status"] == "success"]
        reuses = [r["kv_reuse"] for r in rows if isinstance(r["kv_reuse"], (int, float))]
        groups.append({
            "task": task,
            "runs": len(rows),
            "success": len(ok),
            "partial": len([r for r in done if r["status"] == "partial"]),
            "failed": len([r for r in done if r["status"] == "failed"]),
            "unfinished": len(rows) - len(done),
            "success_rate": round(len(ok) / len(done), 4) if done else None,
            "steps_median": _median([r["steps"] for r in rows]),
            "calls_median": _median([r["calls"] for r in rows]),
            "calls_total": sum(r["calls"] for r in rows if isinstance(r["calls"], int)),
            "tokens_median": _median([r["tokens"] for r in rows]),
            "tokens_total": sum(r["tokens"] for r in rows if isinstance(r["tokens"], int)),
            "kv_reuse_mean": round(sum(reuses) / len(reuses), 4) if reuses else None,
            "seconds_median": _median([r["seconds"] for r in rows]),
            "verify_passed": len([r for r in rows if r["verify"] == "passed"]),
            "verify_flagged": len([r for r in rows if r["verify"] == "flagged"]),
            "last_status": rows[0]["status"],  # runs 已由新到舊
            "last_run_id": rows[0]["run_id"],
            "last_started_at": rows[0]["started_at"],
            "run_ids": [r["run_id"] for r in rows],
        })
    groups.sort(key=lambda g: (-g["runs"], -(g["last_started_at"] or 0)))
    return groups


def build(runs_root: Path) -> dict[str, Any]:
    """看板資料：CLI --json 與 Web /api/stats 回傳的就是這個結構（鍵固定，方便測試與前端）。"""
    runs = collect_runs(runs_root)
    groups = group_by_task(runs)
    done = [r for r in runs if r["status"] in DONE_STATUSES]
    return {
        "runs_root": str(runs_root),
        "runs_total": len(runs),
        "tasks_total": len(groups),
        "finished": len(done),
        "success": len([r for r in done if r["status"] == "success"]),
        "success_rate": round(len([r for r in done if r["status"] == "success"]) / len(done), 4) if done else None,
        "tokens_total": sum(r["tokens"] for r in runs if isinstance(r["tokens"], int)),
        "empty": not runs,
        "message": EMPTY_MESSAGE if not runs else "",
        "tasks": groups,
        "runs": runs,
    }


# ---------------- 文字呈現（CLI） ----------------
def _width(text: str) -> int:
    """終端機顯示寬度：中日韓全形字算 2 格，對齊才不會歪。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(1, width - _width(text))


def _clip(text: str, width: int) -> str:
    out = ""
    for ch in text:
        if _width(out + ch) > width - 1:
            return out + "…"
        out += ch
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    return (f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}") + suffix


def render(data: dict[str, Any]) -> str:
    """把 build() 的結果渲染成繁體可讀表格（與 lingxi runs / memory stats 同一風格）。"""
    if data["empty"]:
        return EMPTY_MESSAGE
    lines = [f"穩定性看板 · {data['runs_total']} 次執行、{data['tasks_total']} 個任務"
             f" · 整體成功率 {_pct(data['success_rate'])}（{data['success']}/{data['finished']}）"
             f" · 共 {data['tokens_total']:,} tokens"]
    # 欄位寬度依 _width（全形字算 2 格）計算，對齊才不會歪
    head = ("任務", 34), ("次數", 6), ("成功率", 8), ("步數", 6), ("呼叫", 6), ("Tokens", 10), ("KV重用", 8), ("耗時", 8), ("最近", 8)
    lines.append("  " + "".join(_pad(h, w) for h, w in head))
    status_zh = {"success": "成功", "partial": "部分", "failed": "失敗", "unfinished": "未結束", "running": "執行中"}
    for g in data["tasks"]:
        lines.append("  " + "".join(_pad(v, w) for v, w in (
            (_clip(g["task"], 34), 34),
            (str(g["runs"]), 6),
            (_pct(g["success_rate"]), 8),
            (_num(g["steps_median"]), 6),
            (_num(g["calls_median"]), 6),
            (_num(g["tokens_median"]), 10),
            (_pct(g["kv_reuse_mean"]), 8),
            (_num(g["seconds_median"], "s"), 8),
            (status_zh.get(g["last_status"], g["last_status"]), 8),
        )))
        detail = f"成功 {g['success']} · 部分 {g['partial']} · 失敗 {g['failed']}"
        if g["unfinished"]:
            detail += f" · 未結束 {g['unfinished']}"
        if g["verify_passed"] or g["verify_flagged"]:
            detail += f" · 查證通過 {g['verify_passed']}、待查 {g['verify_flagged']}"
        lines.append(f"      {_clip(detail + ' · tokens 合計 ' + format(g['tokens_total'], ','), 96)}")
    lines.append("  復盤單次執行：lingxi replay <run_id>")
    return "\n".join(lines)
