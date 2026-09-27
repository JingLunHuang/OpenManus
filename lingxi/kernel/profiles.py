"""任務模式（Profile）：不同類型的任務，給不同的工具集、步數預算和工作指引。

對應課程答疑裡的結論："意圖識別——分類，類別不要太多；測試加 few-shot；做好兜底"。
靈犀只分四類，識別不出來就落到 general（兜底）。
工具集按模式裁剪，也順帶減少了每次請求攜帶的工具 Schema，省 token。
"""

from __future__ import annotations

from dataclasses import dataclass

WEB_ACTIONS = ("web_open", "web_click", "web_type", "web_select", "web_key", "web_scroll", "web_read",
               "web_nav", "web_wait", "web_look")
# 模式只裁剪內建技能；使用者自定義技能、MCP 工具、雲沙箱在所有模式下都可用
BUILTIN_SKILLS = frozenset(WEB_ACTIONS + ("web_search", "python_run", "files", "plan", "ask_human", "finish"))


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    base_budget: int
    date_preference: str  # future | nearest
    guidance: str
    skills: tuple[str, ...] | None = None  # None 表示全部技能

    def allows(self, skill_name: str) -> bool:
        if self.skills is None or skill_name not in BUILTIN_SKILLS or skill_name in ("finish", "ask_human"):
            return True
        return skill_name in self.skills


PROFILES: dict[str, Profile] = {
    "web_query": Profile(
        name="web_query",
        title="即時查詢",
        base_budget=16,
        date_preference="future",
        skills=WEB_ACTIONS + ("web_search", "python_run", "plan"),
        guidance=(
            "這是一次即時查詢。目標是儘快拿到準確的當前資料：\n"
            "- 有手冊直達地址就直接開啟，不要在首頁一步步點；\n"
            "- 結果出現在【正文摘要】或發現板裡之後，立即 finish，整理成清晰的列表並註明來源與查詢時間；\n"
            "- 不需要登入、預訂、下單，除非使用者明確要求。"
        ),
    ),
    "research": Profile(
        name="research",
        title="調研報告",
        base_budget=30,
        date_preference="nearest",
        guidance=(
            "這是一次調研/分析任務：\n"
            "- 第一步先用 plan 制定 3~7 項計劃，之後每完成一項就標記 done；\n"
            "- web_search 找線索，web_open + web_read 核實關鍵資料，事實會記入發現板；\n"
            "- 注意時效：以今天為基準，過舊的資料要標註；區分事實與推斷；\n"
            "- 反覆查證：關鍵數字至少找兩個獨立來源 web_read，讓發現板顯示 ✔ 多方印證；出現 ⚠ 矛盾時再找第三個來源釐清；\n"
            "- 用 files 把報告寫成 Markdown 檔案，再 finish 給出檔案路徑與核心結論。"
        ),
    ),
    "coding": Profile(
        name="coding",
        title="程式碼與資料",
        base_budget=20,
        date_preference="nearest",
        skills=("python_run", "files", "plan", "web_search", "web_open", "web_read"),
        guidance=(
            "這是一次程式設計/資料處理任務：\n"
            "- 用 python_run 小步驗證，每次只做一件事，用 print() 輸出關鍵結果；\n"
            "- 需要儲存的程式碼或資料用 files 寫入工作區；\n"
            "- 報錯時先讀錯誤資訊再修改，不要重複提交同樣的程式碼。"
        ),
    ),
    "general": Profile(
        name="general",
        title="通用",
        base_budget=24,
        date_preference="nearest",
        guidance="根據任務選擇最合適的工具組合；複雜任務先用 plan 拆解。",
    ),
}
