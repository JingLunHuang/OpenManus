"""把一次實驗的紀錄寫成人可讀的對照報告（Markdown）。

    python scripts/experiments/report.py --run-dir experiments/runs/20261010-153000

讀該次實驗產出的 meta.json / summary.json / records.jsonl，寫出：
  - <實驗目錄>/report.md      這一場實驗的完整報告（隨實驗產出一併留存）
  - experiments/report.md    最新一場實驗的報告（固定路徑，方便直接引用）

報告一律用繁體中文，並在最後附上誠實的限制說明：小樣本、不宣稱統計顯著性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ARM_LABEL = {"memory-off": "無記憶（memory off）", "memory-on": "有記憶（memory on）"}
METRIC_LABEL = {
    "success_rate": "成功率",
    "avg_steps": "平均步數",
    "calls": "模型呼叫數",
    "total_tokens": "token 總量",
    "cached_tokens": "其中快取命中",
    "avg_tokens": "平均每次 token",
    "verify_rounds": "查證輪數",
    "avg_seconds": "平均牆鐘秒數",
}


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """讀一場實驗的三個檔案；缺哪個就退回空結構（報告仍可產生）。"""
    def read_json(name: str) -> dict[str, Any]:
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def read_jsonl(name: str) -> list[dict[str, Any]]:
        p = run_dir / name
        if not p.is_file():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # 中途崩潰可能留下半行，別讓報告整個失敗
                    continue
        return out

    return read_json("meta.json"), read_json("summary.json"), read_jsonl("records.jsonl")


def _pct(value: float) -> str:
    return f"{value:.0%}" if value else "—"


def _arm_table(arms: dict[str, Any]) -> list[str]:
    """兩組對照總表：成功率 / 成本 / 效率。"""
    lines = [
        "| 指標 | 無記憶（memory off） | 有記憶（memory on） | 差異 |",
        "| --- | --- | --- | --- |",
    ]
    names = [n for n in ("memory-off", "memory-on") if n in arms]
    if not names:
        return ["（沒有任何組別的有效紀錄）"]

    def cell(agg: dict[str, Any], key: str) -> str:
        if not agg or not agg.get("effective"):
            return "—"
        value = agg.get(key, 0)
        if key == "success_rate":
            return f"{_pct(value)}（{agg['successes']}/{agg['effective']}）"
        return f"{value:,}" if isinstance(value, int) and value >= 1000 else f"{value}"

    for key, label in METRIC_LABEL.items():
        row = [cell(arms.get(n, {}), key) for n in names]
        diff = "—"
        if len(names) == 2 and all(arms.get(n, {}).get("effective") for n in names):
            a, b = (arms[n].get(key, 0) for n in names)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
                pct = (b - a) / abs(a)
                arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "→")
                diff = f"{arrow} {pct:+.0%}"
            elif key == "success_rate":
                diff = f"{(b - a):+.0%}"
        lines.append(f"| {label} | {row[0]} | {row[1] if len(row) > 1 else '—'} | {diff} |")
    return lines


def _per_task_table(per_task: dict[str, Any]) -> list[str]:
    """逐題明細：兩組的成功次數與成本並排。"""
    if not per_task:
        return ["（沒有逐題紀錄）"]
    lines = [
        "| 任務 | 模式 | 題目 | 無記憶 | 有記憶 | 平均步數（off / on） | token（off / on） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tid, info in per_task.items():
        off = info.get("memory-off", {})
        on = info.get("memory-on", {})

        def cell(stat: dict[str, Any]) -> str:
            if not stat or not stat.get("effective"):
                return "—（跳過）"
            mark = "✅" if stat["majority"] else "❌"
            return f"{mark} {stat['successes']}/{stat['effective']}"

        prompt = (info.get("prompt") or "").replace("|", "\\|")
        if len(prompt) > 46:
            prompt = prompt[:46] + "…"
        lines.append(
            f"| `{tid}` | {info.get('mode', '')} | {prompt} | {cell(off)} | {cell(on)} | "
            f"{off.get('avg_steps', '—')} / {on.get('avg_steps', '—')} | "
            f"{off.get('tokens', '—')} / {on.get('tokens', '—')} |")
    return lines


def _failures(records: list[dict[str, Any]]) -> list[str]:
    failed = [r for r in records if r.get("status") != "skipped" and not r.get("success")]
    if not failed:
        return ["沒有失敗的執行 🎉"]
    lines = []
    for r in failed:
        reason = r.get("error") or f"status={r.get('status')}"
        lines.append(f"- `{r.get('task_id')}` · {r.get('arm')} · 第 {r.get('attempt')} 次：{reason}")
    return lines


def _limitations(meta: dict[str, Any], arms: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    provider = meta.get("provider", "scripted")
    repeats = meta.get("repeats", 1)
    n = sum(a.get("effective", 0) for a in arms.values())
    lines = [
        "## 誠實的限制",
        "",
        f"- **樣本很小**：每題 {repeats} 次、兩組合計 {n} 次有效執行。"
        "這是「能不能重現、成本差多少」的量測，不是統計推論的樣本數。",
        "- **不宣稱統計顯著性**：上表的差異只是觀察值。12 題 × 少量重複，"
        "任何百分比變動都可能來自運氣或任務難度分布，不是因果證明。",
    ]
    if provider == "scripted":
        lines += [
            "- **本次是離線乾跑**：`provider=scripted`，模型由 ScriptedLLM 依固定劇本回覆，"
            "網頁是 lingxi/evaluation/fixtures 的夾具頁。**這份數字證明的是「實驗管線正確、可重現」，"
            "不是「記憶對真實模型的增益」**——真實結論要等 M2（`--provider` 接真實端點）。",
            "- 離線模式下關閉了反覆查證閘門（它會額外消耗一次模型呼叫，劇本未為它編排），"
            "因此本報告的「查證輪數」為 0，不代表功能失效。",
        ]
    else:
        lines += [
            "- **模型非確定性**：同一題每次回答都可能不同；重複次數少時，單題成敗帶有隨機性。",
            "- **任務集偏窄**：以旅遊訂票類查詢與報告為主，不代表其他領域的表現。",
        ]
    if meta.get("aborted"):
        lines.append(f"- **實驗被成本護欄中止**：`{meta['aborted']}`。後續題目的紀錄不完整，數字請只當作已跑完部分。")
    if meta.get("browser_available") is False:
        lines.append("- **本機沒有 Chromium**：所有需要瀏覽器的任務都被跳過，該模式沒有任何數字。")
    if any(r.get("budget_exceeded") for r in records):
        lines.append("- **有執行被單次預算護欄中止**：這些執行記為失敗，不應解讀為任務本身做不來。")
    return lines


def render(run_dir: Path) -> str:
    meta, summary, records = load_run(run_dir)
    arms = summary.get("arms", {})
    per_task = summary.get("per_task", {})
    if not arms and records:  # summary 遺失（例如手動中斷）時就地重算
        arms = {arm: {"runs": sum(1 for r in records if r["arm"] == arm),
                      "effective": sum(1 for r in records if r["arm"] == arm and r["status"] != "skipped"),
                      "successes": sum(1 for r in records if r["arm"] == arm and r["success"]),
                      "success_rate": 0.0}
                for arm in ("off", "on")}

    created = meta.get("created", "")
    lines = [
        "# W1 有記憶／無記憶對照實驗報告",
        "",
        f"- 執行時間：{created}",
        f"- 任務集：`{meta.get('suite', '?')}`（{len(meta.get('tasks', []))} 題）× 每題 {meta.get('repeats', 1)} 次",
        f"- 模型來源：`{meta.get('provider', 'scripted')}`"
        + ("（離線假模型，見下方限制）" if meta.get("provider", "scripted") == "scripted" else "（真實模型端點）"),
        f"- 產出目錄：`{run_dir}`（`records.jsonl` 可逐筆復盤）",
        "",
        "## 一、兩組對照",
        "",
    ]
    lines += _arm_table(arms)
    lines += ["", "## 二、逐題明細", "", "`✅ n/m` = 該題在該組的成功次數／有效次數；✅ 表示多數成功。", ""]
    lines += _per_task_table(per_task)
    lines += ["", "## 三、失敗與中止紀錄", ""]
    lines += _failures(records)
    lines += ["", *_limitations(meta, arms, records), ""]
    lines += [
        "## 復盤方式",
        "",
        "- 逐筆事件：`cat <實驗目錄>/memory-<組>/runs/*/events.jsonl`",
        "- 單次執行復盤：`lingxi replay <runs 目錄下的執行目錄>`",
        "- 重跑這一場：`python scripts/experiments/run_experiment.py --suite "
        f"{meta.get('suite', 'main')} --provider {meta.get('provider', 'scripted')}`",
        "",
    ]
    return "\n".join(lines)


def generate_report(run_dir: Path, *, update_latest: bool = True) -> Path:
    """產生報告：一份存在實驗目錄內（隨產出留存），一份更新 experiments/report.md。"""
    run_dir = Path(run_dir)
    text = render(run_dir)
    out = run_dir / "report.md"
    out.write_text(text, encoding="utf-8")
    if update_latest:
        latest = ROOT / "experiments" / "report.md"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="把實驗紀錄寫成對照報告")
    p.add_argument("--run-dir", default="", help="實驗產出目錄（experiments/runs/<時間戳>）")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if not (run_dir / "records.jsonl").is_file():
        raise SystemExit(f"{run_dir} 裡沒有 records.jsonl，請確認 --run-dir 指向一次完整的實驗產出")
    out = generate_report(run_dir)
    print(f"報告已寫入 {out}（另同步到 experiments/report.md）")
    return out


def latest_run_dir() -> Path:
    runs = ROOT / "experiments" / "runs"
    dirs = sorted((d for d in runs.glob("*") if (d / "records.jsonl").is_file()),
                  key=lambda d: d.name) if runs.is_dir() else []
    if not dirs:
        raise SystemExit("找不到任何實驗紀錄（experiments/runs/）；請先執行 run_experiment.py")
    return dirs[-1]


if __name__ == "__main__":
    main()
