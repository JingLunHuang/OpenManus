"""知識層：時間錨定 + 站點手冊（純函式，不需要瀏覽器和模型）。"""

from datetime import date
from pathlib import Path

from lingxi.hanzi import fold
from lingxi.knowledge.playbook import PlaybookLibrary
from lingxi.knowledge.temporal import TemporalAnchor, cn_to_int, month_day

TODAY = date(2026, 9, 27)  # 星期日
PLAYBOOKS = Path(__file__).resolve().parent.parent / "playbooks"


def iso(text, prefer="future"):
    return [a.iso for a in TemporalAnchor(TODAY, prefer=prefer).find(text)]


def test_month_day_rolls_forward_for_future_tasks():
    # 課程故障復現：模型把"6月26日"當成 2023 年。錨定後給出確定的未來日期
    assert iso("查詢 6月26日 從上海到北京的機票") == ["2027-06-26"]
    assert iso("10月1日出發") == ["2026-10-01"]


def test_nearest_preference_for_retrospective_tasks():
    assert iso("總結6月26日的新聞", prefer="nearest") == ["2026-06-26"]


def test_relative_and_weekday_expressions():
    assert iso("明天") == ["2026-09-28"]
    assert iso("後天上午") == ["2026-09-29"]
    assert iso("下週五") == ["2026-10-02"]
    assert iso("週五") == ["2026-10-02"]
    assert iso("本週一", prefer="nearest") == ["2026-09-21"]


def test_full_dates_chinese_numerals_and_day_only():
    assert iso("2026年1月30日") == ["2026-01-30"]
    assert iso("六月二十六日") == ["2027-06-26"]
    assert iso("30號") == ["2026-09-30"]
    assert iso("26號") == ["2026-10-26"]


def test_non_dates_are_ignored():
    assert iso("三亞5日遊") == []
    assert iso("12月3000元的預算") == []
    assert iso("2026年新能源汽車出海") == []


def test_annotate_writes_absolute_date_back_into_task():
    text, anchors = TemporalAnchor(TODAY).annotate("查詢 6月26日 的機票")
    assert text == "查詢 6月26日〔=2027-06-26 星期六〕 的機票"
    assert anchors[0].weekday == "星期六"


def test_cn_to_int_and_month_day():
    assert cn_to_int("二十六") == 26 and cn_to_int("十") == 10 and cn_to_int("十二") == 12
    assert month_day("6月26日") == (6, 26)
    assert month_day("26號") == (None, 26)
    assert month_day("2027-06-26") == (6, 26)
    assert month_day("搜尋") == (None, None)


def test_ctrip_playbook_compiles_direct_url_with_city_codes():
    library = PlaybookLibrary.load([PLAYBOOKS])
    task = "查詢 6月26日 從上海到北京的機票"
    matched = library.match(task, TemporalAnchor(TODAY).find(task))
    assert matched and matched[0].id == "ctrip-flight"
    pb = matched[0]
    assert pb.shown == {"from": "上海", "to": "北京", "date": "2027-06-26"}
    assert pb.url == ("https://flights.ctrip.com/online/list/oneway-sha-bjs"
                      "?depdate=2027-06-26&cabin=y&adult=1&child=0&infant=0")
    assert pb.warmup == "https://www.ctrip.com/"
    assert "直達地址" in pb.render()


def test_playbook_slot_extraction_variants():
    library = PlaybookLibrary.load([PLAYBOOKS])
    for task in ("幫我查詢明天上海到廣州的機票", "用攜程查一下明天從上海飛廣州的航班"):
        pb = library.match(task, TemporalAnchor(TODAY).find(task))[0]
        assert (pb.values["from"], pb.values["to"]) == ("sha", "can"), task


def test_playbook_reports_missing_slots_instead_of_guessing():
    library = PlaybookLibrary.load([PLAYBOOKS])
    task = "幫我看看從上海到火星的機票"
    pb = library.match(task, TemporalAnchor(TODAY).find(task))[0]
    assert pb.url is None
    assert any("to" in m for m in pb.missing) and any("date" in m for m in pb.missing)


def test_unrelated_task_matches_no_playbook():
    library = PlaybookLibrary.load([PLAYBOOKS])
    assert library.match("把這段話翻譯成英文", []) == []


def test_simplified_input_is_equally_supported():
    """介面是繁體，但簡體任務（或從大陸網站複製來的文字）同樣要能解析。"""
    task = fold("查詢 後天 從上海到廣州的機票")  # 以 fold() 產生簡體輸入
    anchors = TemporalAnchor(TODAY).find(task)
    assert [a.iso for a in anchors] == ["2026-09-29"] and anchors[0].text == fold("後天")
    pb = PlaybookLibrary.load([PLAYBOOKS]).match(task, anchors)[0]
    assert pb.id == "ctrip-flight" and pb.values == {"from": "sha", "to": "can", "date": "2026-09-29"}
    assert pb.shown["to"] == fold("廣州")  # 顯示使用者的原文（簡體）
    assert iso(fold("下週五")) == iso("下週五") == ["2026-10-02"]
