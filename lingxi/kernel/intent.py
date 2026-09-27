"""意圖路由：規則打分為主，手冊指定優先，LLM 僅在拿不準時兜底（可關閉）。"""

from __future__ import annotations

from dataclasses import dataclass

from lingxi.hanzi import unify
from lingxi.kernel.profiles import PROFILES, Profile
from lingxi.knowledge.playbook import CompiledPlaybook
from lingxi.llm.client import ChatModel
from lingxi.llm.messages import Message

# 兩岸用語並列（程式碼 / 代碼、資料 / 數據），比對時兩邊都先 unify()，繁體、簡體任務都能命中
KEYWORDS: dict[str, list[str]] = {
    "web_query": ["查詢", "查一下", "查一查", "幫我查", "機票", "航班", "酒店", "火車票", "高鐵", "天氣", "價格",
                  "股價", "匯率", "多少錢", "營業時間", "快遞", "即時", "實時", "最新", "現在"],
    "research": ["報告", "分析", "調研", "調查", "對比", "比較", "競品", "研究", "綜述", "seo", "審查", "評估",
                 "趨勢", "洞察", "白皮書", "方案"],
    "coding": ["程式碼", "代碼", "腳本", "python", "程式設計", "編程", "寫程式", "函式", "函數", "bug", "csv",
               "excel", "json", "資料處理", "數據處理", "畫圖", "圖表", "計算", "統計", "正規表示式", "正則", "爬蟲"],
}
_UNIFIED = {name: [unify(k.lower()) for k in words] for name, words in KEYWORDS.items()}

_FEW_SHOT = """把任務歸入一個類別，只輸出類別名：web_query / research / coding / general
示例：
- 查一下明天上海的天氣 → web_query
- 寫一份新能源汽車出海歐洲的競品分析報告 → research
- 把 data.csv 按月份彙總並畫折線圖 → coding
- 幫我把這段話翻譯成英文 → general"""


@dataclass
class Route:
    profile: Profile
    reason: str
    scores: dict[str, int]


def rule_scores(task: str) -> dict[str, int]:
    low = unify(task.lower())
    return {name: sum(1 for k in words if k in low) for name, words in _UNIFIED.items()}


def route_by_rules(task: str, playbooks: list[CompiledPlaybook] | None = None) -> Route | None:
    scores = rule_scores(task)
    if playbooks and playbooks[0].playbook.profile in PROFILES:
        name = playbooks[0].playbook.profile
        return Route(PROFILES[name], f"手冊《{playbooks[0].title}》指定", scores)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    if top_score > 0 and top_score > second_score:
        return Route(PROFILES[top], f"關鍵詞命中 {top_score} 個", scores)
    return None  # 拿不準


async def route(task: str, playbooks: list[CompiledPlaybook] | None = None,
                llm: ChatModel | None = None, use_llm: bool = False) -> Route:
    decided = route_by_rules(task, playbooks)
    if decided:
        return decided
    scores = rule_scores(task)
    if use_llm and llm is not None:
        try:
            reply = await llm.chat([Message.system(_FEW_SHOT), Message.user(task)])
            name = reply.content.strip().split()[0].strip("`'\"。.") if reply.content.strip() else ""
            if name in PROFILES:
                return Route(PROFILES[name], "規則拿不準，LLM 判定", scores)
        except Exception:
            pass
    return Route(PROFILES["general"], "未命中明確類別，使用通用模式（兜底）", scores)
