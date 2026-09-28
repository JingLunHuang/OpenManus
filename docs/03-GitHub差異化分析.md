# 與 GitHub 同類專案的差異化分析

> 調研時間：2026-09-27（同日二次查證，見第 5 節）。目的：確認靈犀在「同一門課的同學作品」和「更廣的瀏覽器智慧代理生態」裡分別處於什麼位置，
> 以及面試時被問到「你的專案和 OpenManus／別人的有什麼不同」時，哪些說法站得住。

## 1. 同源專案（OpenManus 及其衍生版）

搜尋「OpenManus 二次開發／GUI／攜程」能找到多個與本課程同源的作品。它們的共同點是：**保留 OpenManus 的 `app/` 目錄與四級繼承結構，在其上增加模組。**

| 專案 | 主要增強 | 架構 |
|---|---|---|
| [FoundationAgents/OpenManus](https://github.com/FoundationAgents/OpenManus) | 上游：ReAct 工具呼叫、browser-use、MCP、PlanningFlow、Docker／Daytona 沙箱 | 原始結構 |
| [wyplds/openmanus-browser-automation-agent](https://github.com/wyplds/openmanus-browser-automation-agent) | L0 知識庫直達 → L1 DOM 索引定位 → L2 元素語義分類輔助 → L3 視覺模型兜底的「多級降級鏈」；穩定性守衛；反檢測；143 項單元測試 | fork，沿用 `app/`，依賴 browser-use |
| [ckenkuo/manus-gui](https://github.com/ckenkuo/manus-gui)／[taofanwen-cell/manus-gui](https://github.com/taofanwen-cell/manus-gui) | 視覺 `gui_action` 兜底（dpr 座標換算）；CDP 接管真實 Chrome；成功經驗蒸餾為 RAG 配方（Faiss ＋ BM25 ＋ RRF）；後者定位為拼多多競品研究平台 | fork，沿用 `app/` |
| [Cecil09312/OpenManus-gui](https://github.com/Cecil09312/OpenManus-gui)、[Hank-Chromela/OpenManus-GUI](https://github.com/Hank-Chromela/OpenManus-GUI) | Web／Gradio 圖形介面、對話歷史 | fork |
| [mtaman/OpenManus-Web](https://github.com/mtaman/OpenManus-Web) | 透過猴子補丁 `Memory.add_message` 用 SSE 推送思考與工具執行 | fork |

**觀察**：同學們普遍採用「DOM 失敗 → 視覺兜底」的降級思路，經驗庫則是「文字注入」或「向量檢索配方」；答覆送出前都沒有獨立的查證步驟。

## 2. 靈犀與它們的區別

靈犀同樣以 OpenManus 開發框架為主體（分層、使用習慣、設定方式都延續），差別在於每一層的實作方式：

| 維度 | 同源衍生版的常見做法 | 靈犀 |
|---|---|---|
| 程式碼結構 | fork 上游，保留 `BaseAgent→ReAct→ToolCall→Manus` 繼承鏈 | 全新包 `lingxi`：`Kernel` ＋ `Hook` 組合，`RunContext` 共享狀態，約 5.7 千行 |
| 瀏覽器感知 | browser-use 元素列表（＋正則分類器／視覺） | 不依賴 browser-use：自研快照（cursor:pointer 啟發式、日期格月份上下文、遮擋圖層、穩定編號） |
| 視覺的角色 | 兜底：視覺給座標就點 | 神諭：視覺座標必須經 `elementFromPoint` 反查 DOM 並比對文字，一致才採納 |
| 動作結果 | 執行即成功 | 每個動作「定位 → 執行 → 驗證」，未觀察到變化會標註「未驗證」；輸入回讀值、列出聯想候選 |
| 歧義處理 | 取置信度最高的一個 | 分數接近時拒絕猜測，把候選編號交還給模型 |
| 經驗庫 | txt 注入／向量檢索配方 | 宣告式 TOML 手冊：填槽 ＋ 代碼表 ＋ **編譯出直達 URL**；預熱首頁在工具層強制執行 |
| 日期 | 提示詞寫當前時間／工具層修正攜程 URL | 任務進入模型前完成時間錨定並寫回任務；URL 日期修正按手冊宣告的參數名通用執行 |
| 步數 | 固定上限（20→25→30→40） | 進展感知預算：有可驗證進展才續航，打轉扣減，最後一步強制收尾 |
| 上下文成本 | 頁面元素每步追加進歷史 | 頁面只在本步簡報出現；舊觀察摺疊；發現板／計畫板儲存關鍵事實 |
| 答覆可信度 | 模型說完即結束 | **反覆查證**：發現板多源印證；finish 前逐條核對證據，無據退回補查，最多 N 輪 |
| 語言 | 簡體介面 | 繁體介面；比對繁簡通吃，繁體指令可定位簡體網頁 |
| 可觀測性 | 日誌 ＋ 臨時儲存 HTML；或猴子補丁推送 SSE | 黑匣子事件流是唯一事實來源，控制台／Web 介面／離線復盤報告都是訂閱者 |
| 測試 | 單元測試 ＋ 真實網站手測 | 離線「迷你攜程」夾具重現三大難點 ＋ `ScriptedLLM` 端到端（含查證循環、記憶、RSI），86 項，不花 token |
| 跨執行學習 | 無（每次從零開始） | LightMem 寫入 ＋ FluxMem 三層記憶圖；技能經 RSI 受保護評測才成為手冊（0.2） |

## 3. 放到更廣的生態裡：哪些是「獨創」，哪些不是

實事求是地說，靈犀裡**單個機制**在更大的智慧代理生態中多數都能找到相近的做法：

- **動作後驗證**：已有專案做「動作前後狀態對比、回讀輸入值」，例如 [AsHura-Wnd/browser-agent](https://github.com/AsHura-Wnd/browser-agent)、[esokullu/webbrain 的文字變更驗證](https://github.com/esokullu/webbrain/pull/339)、[ollama-client 的已驗證 DOM 變更動作](https://github.com/Shishir435/ollama-client/pull/361)。
- **視覺與 DOM 結合**：[hterzia/human-browser](https://github.com/hterzia/human-browser) 採用「視覺模型從帶標籤的元素列表裡選，再解析為座標」，方向與靈犀相反（靈犀是「視覺給座標，DOM 來核驗」）；browser-use 本身也有 Set-of-Mark 視覺標註。
- **預算感知**：學術界已有大量預算感知智慧代理研究（如 [BAGEN](https://arxiv.org/html/2606.00198v1)、[Budget-Aware Online Adaptation for Web Agents](https://arxiv.org/pdf/2609.05513)）。
- **逐條查證閘門**：研究報告類流程中已有「強制的逐條查證」，例如 [Rahul-Innv/stormworthy](https://github.com/Rahul-Innv/stormworthy) 以對抗式查證代理檢查引用；[GPT-Researcher](https://github.com/assafelovic/gpt-researcher) 等則以「邊寫邊引用」為主。
  靈犀的差異在於：查證放在**瀏覽器智慧代理的主迴圈內**，證據來自代理自己的執行軌跡（發現板＋工具輸出＋當前頁面），退回輪數與進展預算連動，並在瀏覽過程中就標記多源印證。

因此更準確的定位是：

> 靈犀的新穎之處不在於某一個技巧，而在於**以 OpenManus 開發框架為主體，把「證據」作為貫穿感知、行動、預算、記憶、查證、觀測的統一設計原則**，
> 並落地為一個可測試、可回放的小型框架；在 OpenManus 衍生專案中，目前沒有找到採用這種架構的作品。

面試時建議這樣講（不誇大、可追問）：
1. **問題驅動**：每個機制都能對應到一次真實故障（見 `01-課程筆記整理.md` 第 9 節）；
2. **取捨清楚**：為什麼不直接視覺兜底（座標不可驗證）、為什麼手冊要編譯而不是注入（不依賴模型記憶）、為什麼預算要看進展（固定上限要麼不夠要麼浪費）、
   為什麼查證員只看證據（它驗證「有沒有根據」，這件事可以被程式檢查）；
3. **可驗證**：離線夾具 ＋ 假模型端到端測試，任何人 clone 下來不需要 API Key 就能跑通。

## 4. 仍然可以繼續拉開差距的方向

- ~~從黑匣子自動生成手冊草稿~~：0.2 已由 FluxMem 技能蒸餾 ＋ RSI 手冊匯出實現，而且不是「人工審核」而是「受保護評測把關、可回滾」；
- 用真實模型做「有記憶 / 無記憶」的成功率與 token 對照實驗（目前所有數字都來自離線示範）；
- 同一任務多次執行的成功率／token 迴歸看板（把「穩定性」量化）；
- 查證時對「單一來源」的關鍵數字自動發起第二來源檢索（目前由模型依提示自行補查）。

## 5. 二次查證紀錄（2026-09-27）

本文件的事實性陳述在撰寫後逐條回到來源再查了一次：

| 陳述 | 來源 | 結果 |
|---|---|---|
| wyplds 版有 143 項單元測試、L0～L3 四級降級鏈、沿用 `app/`、依賴 browser-use | 該專案 README | ✔ 相符（README 原文為簡體，轉寫如下：「143 項單元測試」「知識庫直達／DOM 索引定位／元素語義分類輔助／視覺模型兜底」） |
| manus-gui 使用 Faiss ＋ BM25 ＋ RRF、以 CDP 接管真實 Chrome | taofanwen-cell/manus-gui README | ✔ 相符 |
| taofanwen-cell/manus-gui 有 163 項 CI 測試 | 專案描述 | ⚠ 只出現在 GitHub 專案簡介，README 正文未提及——因此本文不引用這個數字 |
| 研究流程中已有逐條查證閘門 | stormworthy README | ✔ 相符，已據此修正第 3 節，不宣稱查證閘門為首創 |
| 課程文件中的數字（8 個真實航班、截圖僅 10KB、tier3 才能上網、20 步硬上限、「2 天 500 塊」、免費額度只夠三四輪） | 原始簡報 PDF 抽取文字與課堂筆記 | ✔ 相符 |

## 6. 0.2 版新增能力的差異化查證（2026-09-28）

新增的自我學習記憶、RSI、KV 快取、LlamaIndex 檢索，同樣先查了 GitHub 上有沒有相近的作品：

| 相近的專案 | 它做了什麼 | 與靈犀的差別 |
|---|---|---|
| [zjunlp/LightMem](https://github.com/zjunlp/LightMem)（含 FluxMem）與其衍生（如 [LightMem2](https://github.com/jgx15638-creator/LightMem2)） | 記憶框架本身，評測於 LoCoMo、LongMemEval（對話）與 Mind2Web、GAIA（FluxMem） | 它們是**記憶元件**；靈犀把兩者**組合**成「LightMem 寫入 ＋ FluxMem 組織」並接進瀏覽器代理的主迴圈（動作簽名、站點經驗、Stage II 依動作成敗歸因）。沒有找到把 LightMem 或 FluxMem 接進 OpenManus 的專案 |
| [webscout9-png/rsi-forge](https://github.com/webscout9-png/rsi-forge) | 有界的 harness ＋ 技能庫 RSI：保留的評測集不給改進器看、版本控制與回滾 | **思路最接近**，因此不宣稱「受保護評測 ＋ 回滾」是首創。差別在於：靈犀依單一論文（2609.11873）的九要素與 L1–L5 實作並記錄自主歸屬；改進對象包含 KV 快取參數（以計費模型評分）與記憶蒸餾出的技能；而且嵌在瀏覽器代理裡 |
| [AlexWortega/OpenRsi](https://github.com/AlexWortega/OpenRsi) | 每次解題後把觀察寫成標記條目，下次召回注入提示 | 屬於「經驗記憶注入」，沒有評測閘門與版本化 |
| Hermes-Agent 系（如 [GarrettRoi/open-manus](https://github.com/GarrettRoi/open-manus)，名稱雖為 open-manus，實為 Nous Research 的 Hermes-Agent） | 代理自行整理記憶、複雜任務後自動建立技能、技能在使用中自我改進 | 公開文件沒有描述驗證閘門、版本或回滾；技能直接生效 |
| KV 快取友善的提示工程（[部落格](https://ankitbko.github.io/blog/2025/08/prompt-engineering-kv-cache/)、[context-caching](https://github.com/theketan26/context-caching)、[Hermes-Agent issue #13631](https://github.com/NousResearch/hermes-agent/issues/13631)） | 「穩定的內容放前面、動態內容放後面」是已知的最佳實務；Hermes-Agent 的 issue 正是「注入的上下文每 N 輪重建、讓快取失效」 | 不宣稱「穩定前綴」是首創。靈犀的差別在於**分段摺疊**（歷史摺疊與快取的衝突）、離線的 `PrefixMeter` 量測、以及把摺疊參數交給 RSI 用計費模型調整 |

**結論（不誇大、可追問）**：單看每一項——記憶框架、RSI 閘門、KV 快取提示工程、LlamaIndex 檢索——生態裡都有相近的做法或原始論文。
靈犀的差異在於**組合方式與落點**：在 OpenManus 衍生的瀏覽器代理裡，讓「執行經驗 → 記憶 → 技能 → 經受保護評測才繼承的新版本」形成閉環，
並且每一段都有離線可重現的測試與量測數字。在 OpenManus 衍生專案中，目前沒有找到這樣的組合。

面試時可以這樣講這一段：
1. **為什麼要閘門**：記憶會學錯（例：短觸發詞的手冊會搶走別本手冊的任務），所以學到的東西要先過受保護評測——而且真的擋下過一次；
2. **為什麼 KV 快取要在框架層處理**：前綴能不能重用由上下文佈局決定，推理端再快也救不了每一步都改寫歷史的框架；
3. **哪裡還不夠**：所有數字來自離線示範，下一步是用真實模型做有 / 無記憶的對照實驗。
