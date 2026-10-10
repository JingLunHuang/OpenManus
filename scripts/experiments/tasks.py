"""W1 對照實驗的任務集：固定題目、固定劇本、可重現。

設計原則（對應 docs/05 §2 W1）：
- ≥12 題，涵蓋 web_query / research / coding 三種任務模式；
- ≥6 題是「同一站點的重複任務」——這是記憶能發揮作用的前提（站點知識、相似經歷才有機會被召回）；
- 離線（scripted）模式：每題附一段 ScriptedLLM 劇本，指向 lingxi/evaluation/fixtures 的夾具頁
  （由 build_site_dir 產生 flight.html / flight-cn.html / long.html，經本地 HTTP 伺服器提供）；
- 真實模型（M2）模式：只用 prompt（{site} 佔位符會填入站點 URL），劇本被忽略。

注意：本模組只定義「題目與劇本」，不執行任何任務；執行與量測在 run_experiment.py。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
for _p in (str(ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lingxi.evaluation.scripted import call, reply  # noqa: E402

FIXTURE_SITE = "fixture-flight"  # 同站重複任務的站點分組名（離線夾具頁，經本地伺服器提供）


@dataclass
class Task:
    id: str
    prompt: str  # 可含 {site} 佔位符，執行時填入站點 URL
    mode: str  # 預期模式：web_query / research / coding / general（僅供報表分類，實際以意圖路由為準）
    site: str = ""  # 站點分組：同站的題目共用一個值（「重複任務」的依據）
    needs_browser: bool = False
    needs_site: bool = False  # 劇本需要夾具站 URL（離線模式才用）
    expect: tuple[str, ...] = ()  # 成功判定：status == success 且答覆包含全部子串
    script: Callable[[str], list] | None = None  # site_url -> ScriptedLLM 劇本；None = 僅支援真實模型


# ---------------------------------------------------------------- 劇本元件


def _flight_steps(site: str, src: str, dst: str, date_label: str, *, page: str = "flight.html",
                  login_overlay: bool = False) -> list:
    """在夾具訂票頁完成一次查詢的動作序列。"""
    code = {"上海": "SHA", "廣州": "CAN", "北京": "BJS", "深圳": "SZX"}
    steps = [reply(call("web_open", url=f"{site}/{page}" + ("?login=1" if login_overlay else "")),
                   content="先開啟訂票頁")]
    if login_overlay:
        steps.append(reply(call("web_click", target="×"), content="頁面被登入浮層遮擋，先關閉它"))
    steps += [
        reply(call("web_type", target="出發城市", text=src)),
        reply(call("web_click", target=f"{src}({code[src]})")),
        reply(call("web_type", target="到達城市", text=dst)),
        reply(call("web_click", target=f"{dst}({code[dst]})")),
        reply(call("web_click", target="出發日期")),
        reply(call("web_click", target=date_label)),
        reply(call("web_click", target="搜尋")),
    ]
    return steps


def _flight_answer(src: str, dst: str, date_label: str) -> str:
    # 夾具頁的搜尋結果是固定的 3 個航班（見 lingxi/evaluation/fixtures/flight.html）
    return (f"{date_label}從{src}飛{dst}共 3 個航班：東方航空 MU5101 07:00 ¥680、"
            f"中國國航 CA1858 08:30 ¥720、海南航空 HU7604 10:50 ¥880（來源：測試訂票頁）")


def flight_query_task(tid: str, src: str, dst: str, date_label: str, *, page: str = "flight.html",
                      login_overlay: bool = False) -> Task:
    """一題「在夾具訂票頁查機票」的 web_query 任務（含離線劇本）。"""
    where = "簡體版測試頁" if page == "flight-cn.html" else "測試頁"
    prompt = f"在{where}查詢 {date_label} 從{src}到{dst}的機票航班"
    answer = _flight_answer(src, dst, date_label)

    def script(site_url: str) -> list:
        return _flight_steps(site_url, src, dst, date_label, page=page, login_overlay=login_overlay) + [
            reply(call("finish", answer=answer, status="success"), content="結果已出現在頁面上，整理後提交")]

    return Task(id=tid, prompt=prompt, mode="web_query", site=FIXTURE_SITE,
                needs_browser=True, needs_site=True, expect=("MU5101", "680"), script=script)


def _extraction(facts: list[str], answer: str):
    """web_read 內部會另發一次「無工具」的模型呼叫做資訊提取：劇本裡用這個 callable 攔截它。"""

    def item(messages, tools):
        if tools is None:  # 提取呼叫（web_read / 查證員）不帶工具清單
            return reply(content=json.dumps(
                {"found": True, "facts": facts, "answer": answer, "corroborates": [], "contradicts": []},
                ensure_ascii=False))
        raise AssertionError("劇本不預期在這裡收到带工具的呼叫")

    return item


def research_refund_task(tid: str, *, page: str = "long.html") -> Task:
    """調研任務：plan → 開啟夾具長文 → web_read 提取 → 寫報告檔 → finish。"""
    prompt = ("調研夾具乘機須知頁（{site}/" + page + "）的退票規則，"
              "寫一份調研報告存成 refund-report.md，並給出核心結論")

    def script(site_url: str) -> list:
        return [
            reply(call("plan", action="set", items=["開啟乘機須知頁", "精讀退票規則", "寫調研報告並給結論"]),
                  content="先制定計劃"),
            reply(call("web_open", url=f"{site_url}/{page}"), content="開啟文章頁"),
            reply(call("web_read", goal="退票規則與手續費"), content="精讀正文提取事實"),
            _extraction(["起飛前 24 小時以上免費退票", "起飛前 24 小時內退票收取票面價 20% 的手續費"],
                        "24 小時內退票收取票面價 20% 的手續費"),
            reply(call("plan", action="done", index=1)),
            reply(call("plan", action="done", index=2)),
            reply(call("files", action="write", path="refund-report.md",
                       content="# 退票規則調研報告\n\n- 起飛前 24 小時以上：免費退票\n"
                               "- 起飛前 24 小時內：收取票面價 20% 手續費\n"),
                  content="把結論寫成報告檔"),
            reply(call("plan", action="done", index=3)),
            reply(call("finish", status="success",
                       answer="調研報告已寫入 refund-report.md。核心結論：起飛前 24 小時以上免費退票；"
                              "24 小時內退票收取票面價 20% 的手續費（來源：乘機須知頁）。"),
                  content="提交結論"),
        ]

    return Task(id=tid, prompt=prompt, mode="research", site=FIXTURE_SITE,
                needs_browser=True, needs_site=True, expect=("20%", "refund-report.md"), script=script)


def flight_research_task(tid: str, *, page: str = "flight.html") -> Task:
    """調研任務（訂票頁版）：完整查詢流程後精讀價格、寫比較報告。與查詢題同站，屬重複任務。"""
    where = "簡體版測試頁" if page == "flight-cn.html" else "測試頁"
    prompt = f"調研{where}上 6月26日 上海飛北京的航班價格，寫一份比較報告 price-report.md 並給出最低價結論"

    def script(site_url: str) -> list:
        return _flight_steps(site_url, "上海", "北京", "6月26日", page=page) + [
            reply(call("web_read", goal="各航班價格，找出最低價"), content="精讀結果列表"),
            _extraction(["MU5101 票價 ¥680", "CA1858 票價 ¥720", "HU7604 票價 ¥880"],
                        "最低價為 MU5101 的 ¥680"),
            reply(call("files", action="write", path="price-report.md",
                       content="# 航班價格比較報告\n\n- MU5101 ¥680（最低）\n- CA1858 ¥720\n- HU7604 ¥880\n"),
                  content="寫入比較報告"),
            reply(call("finish", status="success",
                       answer="比較報告已寫入 price-report.md。6月26日上海飛北京最低價為 MU5101 的 ¥680"
                              "（來源：測試訂票頁）。"), content="提交結論"),
        ]

    return Task(id=tid, prompt=prompt, mode="research", site=FIXTURE_SITE,
                needs_browser=True, needs_site=True, expect=("680", "price-report.md"), script=script)


def coding_avg_task(tid: str) -> Task:
    """程式碼任務：python_run 計算 → 寫 CSV → finish（不需瀏覽器）。"""
    def script(site_url: str) -> list:
        return [
            reply(call("python_run",
                       code="prices = [680, 720, 880]\nprint('平均票價:', sum(prices) / len(prices))"),
                  content="先算出平均票價"),
            reply(call("python_run",
                       code="import csv\n"
                            "rows = [['航班', '票價'], ['MU5101', 680], ['CA1858', 720], ['HU7604', 880]]\n"
                            "with open('flights.csv', 'w', newline='', encoding='utf-8') as f:\n"
                            "    csv.writer(f).writerows(rows)\n"
                            "print('已寫入 flights.csv')"),
                  content="把資料存成 CSV"),
            reply(call("finish", status="success",
                       answer="平均票價為 760.0 元；資料已寫入 flights.csv（3 個航班）。"), content="提交"),
        ]

    return Task(id=tid, prompt="用 Python 計算機票價格 [680, 720, 880] 的平均值並做資料處理，"
                               "把航班與票價存成 flights.csv，最後回報平均票價",
                mode="coding", site="local", expect=("760", "flights.csv"), script=script)


# ---------------------------------------------------------------- 任務集


def main_suite() -> list[Task]:
    """W1 正式任務集：12 題。

    - web_query × 6（q1–q6）＋ research × 4（r1–r4）＋ coding × 2（c1–c2）；
    - 同一站點（fixture-flight，即離線夾具頁）的重複任務共 10 題 ≥ 6；
    - 題目順序刻意讓同站任務相鄰：記憶組先跑過的站點經驗，之後的同站任務才召回得到。
    """
    return [
        # —— 同站重複：夾具訂票頁的查詢（記憶的主戰場）——
        flight_query_task("q1", "上海", "北京", "6月26日"),
        flight_query_task("q2", "上海", "北京", "7月3日"),
        flight_query_task("q3", "廣州", "北京", "6月26日"),
        flight_query_task("q4", "上海", "北京", "6月20日", login_overlay=True),
        flight_query_task("q5", "上海", "北京", "7月10日", page="flight-cn.html"),
        flight_query_task("q6", "廣州", "北京", "7月15日", page="flight-cn.html"),
        # —— 同站重複：調研（同一批夾具頁）——
        flight_research_task("r1"),
        flight_research_task("r2", page="flight-cn.html"),
        research_refund_task("r3", page="long.html"),
        research_refund_task("r4", page="flight-cn.html"),
        # —— 程式碼（本地，不需瀏覽器）——
        coding_avg_task("c1"),
        coding_avg_task("c2"),
    ]


def smoke_suite() -> list[Task]:
    """冒煙用迷你任務集：2 題、不需瀏覽器與夾具站，離線秒級跑完。"""
    def s1_script(site_url: str) -> list:
        return [reply(call("finish", answer="答案是 42。", status="success"), content="直接回答")]

    def s2_script(site_url: str) -> list:
        return [
            reply(call("python_run", code="print('結果:', 6 * 7)"), content="算一下"),
            reply(call("finish", answer="6 × 7 = 42", status="success"), content="提交"),
        ]

    return [
        Task(id="s1", prompt="不用任何工具，直接回答：40 加 2 等於多少？", mode="general", site="local",
             expect=("42",), script=s1_script),
        Task(id="s2", prompt="用 Python 計算 6 乘以 7 等於多少", mode="coding", site="local",
             expect=("42",), script=s2_script),
    ]


SUITES: dict[str, Callable[[], list[Task]]] = {
    "main": main_suite,
    "smoke": smoke_suite,
}
