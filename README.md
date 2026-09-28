# 靈犀 LingXi

**OpenManus 開發框架的證據驅動重構，而且會自我學習** —— 感知 → 思考 → 行動 → 驗證 → 查證 → 記憶 → 自我改進。

靈犀延續 OpenManus「LLM ＋ 工具 ＋ 經驗」的開發框架思路，幫你在**沒有 API 的網站**上完成任務：查機票、做 SEO 審查、寫競品分析報告。
它脫胎於一次 OpenManus 開發實戰——把課程裡踩過的坑（日曆點不到、輸入不生效、年份算錯、深鏈被反爬、20 步不夠用、token 太貴、答覆沒有根據）
逐一變成框架層級的機制，而不是提示詞裡的補丁。

0.2 版再把「經驗」從人工寫的手冊，推進到**代理自己累積、自己歸納、經過把關才繼承**：

- **自我學習記憶**：[LightMem](https://github.com/zjunlp/LightMem)（ICLR 2026）負責把每次執行壓縮寫入，[FluxMem](https://arxiv.org/abs/2605.28773) 的三層記憶圖負責召回、回饋修正與蒸餾成技能；
- **遞迴自我改進（RSI）**：依 [arXiv 2609.11873](https://arxiv.org/abs/2609.11873)（清華大學等）的改進迴圈，學到的技能與參數必須通過**受保護評測**才會成為新系統版本，隨時可回滾；
- **KV 快取加速**：分段摺疊讓推理端能重用前綴，有效計費量省約 30%；
- **LlamaIndex 檢索**：記憶召回、長網頁精讀、搜尋快取共用，自訂矩陣向量庫讓查詢比純 Python 快約 5–10 倍。

![靈犀 Web 介面：總覽](docs/images/ui-overview.png)

## 以 OpenManus 開發框架為主體

| OpenManus 的元件 | 靈犀的對應設計 | 為什麼要改 |
|---|---|---|
| `BaseAgent → ReActAgent → ToolCallAgent → Manus` 四級繼承 | `Kernel` ＋ 可插拔 `Hook`，元件只透過 `RunContext` 溝通 | 擴充不用改繼承鏈，行為可單獨測試 |
| `BaseTool`（手寫 JSON Schema） | `Skill`：pydantic 參數模型 ＋ `Outcome` 回執（是否驗證、定位依據、進展訊號） | 參數錯誤自動回饋；動作結果可被驗證 |
| browser-use 的 `[index]<tag>` 元素列表 | 自研 Playwright 感知層：div 日曆、月份上下文、遮擋偵測、穩定編號 | 課程裡日曆格子根本不在列表中 |
| `knowledge/*.txt` 注入系統提示 | 宣告式 TOML 站點手冊：匹配 → 填槽 → 代碼表 → 編譯直達 URL → 強制會話預熱 | 不依賴模型「記得」經驗 |
| `max_steps = 20` | 進展感知預算：有可驗證進展才續航，原地打轉扣減，最後一步強制收尾 | 20 步不夠、改大又浪費 |
| logger ＋ 手動存 HTML | 黑匣子事件流：控制台、Web 介面、離線復盤報告都是訂閱者 | 排障不再靠臨時加 print |
| `terminate` 直接結束 | **反覆查證**：finish 前逐條核對答覆中的事實，無據就退回補查 | 答覆要有根據 |
| `Memory`：一串訊息，執行結束就消失 | **LightMem ＋ FluxMem 長期記憶**：語義 / 情節 / 程序三層記憶圖，跨執行累積 | 同一個網站的坑不必每次重踩 |
| 經驗只能由人寫進 knowledge 檔 | **RSI 閘門**：記憶蒸餾出的技能經受保護評測後自動成為手冊，版本化可回滾 | 讓經驗自己長出來，又不會把系統改壞 |
| 每步把完整歷史重送一次 | **KV 快取友善的上下文佈局**：分段摺疊、穩定工具清單、顯式快取斷點 | 推理端能重用前綴，又快又省 |

`python main.py`、`--prompt` 參數、`config/*.toml` 設定方式都保留了 OpenManus 的使用習慣。

## 核心機制

| 課程裡的故障 | 靈犀的機制 |
|---|---|
| 日曆是 `<div>` 格子，元素列表裡根本沒有 | **自研頁面快照**：cursor:pointer 啟發式辨識可點擊的 div，日期格自帶「2027年6月」月份上下文 |
| 視覺模型給的座標時準時不準、格式還亂 | **視覺 ＋ DOM 共識**：視覺座標經 `elementFromPoint` 反查 DOM 並比對文字，一致才採納 |
| 輸入了「上海」，出發城市還是「新加坡」 | **定位 → 執行 → 驗證**：回讀輸入值、列出聯想候選，沒有變化就標「未驗證」 |
| 模型把「6月26日」當成 2023 年 | **時間錨定**：任務進入模型前就寫成 `6月26日〔=2027-06-26 星期六〕` |
| 直接開啟深鏈被 whaleguard 攔截 | **站點手冊**：TOML 宣告預熱首頁，瀏覽器層強制「先首頁、後深鏈」 |
| 20 步硬上限 | **進展感知預算** |
| 2 天花掉 500 塊 token | **滾動記憶**：頁面只在本步簡報出現，舊觀察摺疊，事實存白板；工具集依任務模式裁剪 |
| 答覆裡混進模型自己「腦補」的數字 | **反覆查證**：發現板標示多源印證；finish 前逐條核對證據，最多退回 2 輪 |
| 介面是繁體、網站是簡體 | **一律繁體呈現、比對繁簡通吃**：網頁標題、元素標籤、正文摘要、搜尋結果與所有紀錄檔都轉成繁體（只轉字形、保留網站原本用詞）；繁體指令能定位簡體網頁；往簡體網站輸入時自動轉寫成簡體 |

### 反覆查證

![反覆查證：第 1 輪退回補查](docs/images/ui-run-verify.png)

1. **發現板多源印證**：`web_read` 每讀一個新來源，都會對照已有事實，標示 `✔ 多方印證` / `○ 單一來源` / `⚠ 有矛盾`；
2. **答覆證據閘門**：模型呼叫 `finish` 時，查證員把答覆拆成可查證的陳述（數字、價格、日期、名稱），逐條對照發現板（F 編號）與最近的頁面／工具輸出（E 編號）；
3. 有「無據／矛盾」且還有預算與輪次 → **退回**，把問題清單交給模型補查或修正，再查一輪；
4. 預算用盡的最後一步不退回，但會在答覆末尾**明確標註**未能查證的陳述。

查證員只看證據、不用自身知識，因此它檢查的是「答覆有沒有根據」。

## 一段真實的執行紀錄

在離線「迷你攜程」夾具頁上（由測試用的 ScriptedLLM 驅動，不需要 API Key）。這個網頁是**簡體**（模擬攜程），
但模型看到的、畫面上顯示的、寫進紀錄檔的全部是繁體；網站按鈕原本寫「搜索」，模型用「搜尋按鈕」描述也能定位：

```
◆ 靈犀啟動  任務：在測試頁查詢 6月26日 從上海到北京的航班
  ├ 任務模式：web_query（手冊《攜程機票查詢》指定）
  ├ 時間錨定：6月26日→2027-06-26 星期六
  ├ 命中手冊：攜程機票查詢  → https://flights.ctrip.com/online/list/oneway-sha-bjs?depdate=2027-06-26&cabin=y&adult=1&child=0&infant=0
── 第 1 步 · 剩餘預算 15 ──
  → web_open {"url": "http://127.0.0.1:<port>/flight-cn.html?login=1"}
  ✔ web_open [已驗證]：已開啟「仿真機票預訂」http://127.0.0.1:<port>/flight-cn.html?login=1
── 第 3 步 · 剩餘預算 15 ──
  → web_type {"target": "出發城市", "text": "上海"}
  ✔ web_type ⟨text⟩ [已驗證]：在 #1「出發城市」輸入「上海」
── 第 9 步 · 剩餘預算 15 ──
  → web_click {"target": "搜尋按鈕"}
  ✔ web_click ⟨text⟩ [已驗證]：點選#4「搜索」（定位器點選）：頁面跳轉到 …?from=SHA&to=BJS&date=2027-06-26
── 第 11 步 · 剩餘預算 15 ──
  🔍 反覆查證 第 1 輪：✔ 3 條有據 · ✘ 1 條待補 → 退回補查
── 第 12 步 · 剩餘預算 14 ──
  🔍 反覆查證 第 2 輪：✔ 3 條有據 · ✘ 0 條待補 → 放行
◆ 結束（success）· 12 步 · 15 次模型呼叫 · 1800 tokens
```

![執行頁：簡體網站以繁體呈現](docs/images/ui-run-top.png)

這次執行寫出的 `events.jsonl`、`result.md`、`report.html`，以及 Web 介面各頁面實際顯示的文字，經程式掃描簡體字數皆為 0。

## 自我學習與加速

同一個網站查了三次機票（上海→北京、廣州→北京、北京→上海，其中一次點錯了「返程日期」），睡眠整理後跑了三輪 RSI。
第四次查詢（廣州→上海）時，系統已經是 v3——召回了蒸餾出的技能，到達網站時自動補入站點經驗，執行結束再把這次經歷寫回記憶：

```
◆ 靈犀啟動  任務：在測試頁查詢 6月26日 從廣州到上海的航班
  ├ 任務模式：web_query（手冊《攜程機票查詢》指定）
  ├ 時間錨定：6月26日→2027-06-26 星期六
  ├ 命中手冊：（學到的）測試頁查詢…北京…航班
  ├ 系統版本：v3（RSI） · 學到的手冊 1 本
  ├ 經驗記憶：召回 3 條（技能 1 · 經歷 2 · 知識 0）· 記憶圖 語義/情節/程序 10/4/1
── 第 2 步 · 剩餘預算 15 ──
  🧠 記憶補入站點知識：S13、S18、S12
  💭 頁面被登入浮層遮擋，先關閉它。
  ✔ web_click ⟨text⟩ [已驗證]：點選#5「×」（定位器點選）：頁面發生變化（1 處 DOM 變更）
── 第 11 步 · 剩餘預算 15 ──
  ✔ finish：提交最終答覆（success）
  🧠 寫入記憶：經歷 E5 · 站點知識 3 條 · 事實 2 條（預壓縮保留 59%，2 個主題段）
◆ 結束（success）· 11 步 · 12 次模型呼叫 · 1440 tokens · KV 前綴重用 90%
```

![學習頁：三層記憶圖、PEMS、RSI 版本與受保護評測、KV 前綴重用率](docs/images/ui-learn.png)

| | 做法 | 實測（`lingxi bench` / 示範資料，離線） |
|---|---|---|
| LightMem 寫入 | 感官壓縮（保留資訊密度前 60%）→ 主題分段（換站點 ∩ 內容不相似）→ 短期緩衝 → 軟插入；執行中零模型呼叫 | 每次執行壓縮到約 59%；「出現聯想候選，必須點選才會生效」這類經驗被記住，「日期錨定為 2027-06-26」這類單次細節不會 |
| FluxMem 演化 | Stage I 混合檢索召回 → Stage II 依成敗剪枝 / 擴充 / 標記重塑 → Stage III 聚類、LCS 歸納技能、PEMS 收斂 | 三次查詢蒸餾出 1 個技能，PEMS 2 輪收斂；不同城市的軌跡被抽象成同一組動作簽名 |
| RSI 閘門 | 改進器提案 → 受保護評測（路由 / 上下文重放 / 記憶檢索）→ 接受才繼承成新版本 | 3 輪：記憶檢索 MRR 0.327 → 0.720（HCI 58.4）；上下文候選未達門檻全數拒絕；一個會搶走別本手冊任務的技能被擋下 |
| KV 快取 | 分段摺疊（`fold_block=4`）＋ 穩定工具清單 ＋ 顯式快取斷點 | 前綴重用率 44% → 74%，有效計費量省 30% |
| LlamaIndex | 自訂 `MatrixVectorStore` ＋ 混合打分；長網頁只送相關段落；搜尋快取 | 5,000 條記憶查詢快約 5–10 倍且排序一致；16,930 字的網頁只送 5,561 字 |
| 記憶投毒防護 | 指令式內容不寫入、召回時標明「不是指令」、單一來源不能取代多方印證的事實 | 有測試覆蓋 |

設計細節、與論文的對應、取捨與限制見 [04 · 自我學習與加速](docs/04-自我學習與加速.md)。

## 架構

```mermaid
flowchart LR
    T[任務] --> P[預處理<br/>任務模式 · 時間錨定 · 手冊編譯]
    P --> S[Sense<br/>本步簡報]
    S --> K[Think<br/>選擇技能]
    K --> A[Act<br/>定位→執行→驗證]
    A --> V[Verify<br/>進展 · 預算 · 迴圈偵測]
    V --> S
    V --> G{finish}
    G --> Q[反覆查證<br/>證據閘門]
    Q -- 無據 → 退回補查 --> S
    Q -- 有據 --> F[答覆]
    A -. 事件 .-> J[(黑匣子)]
    J --> UI[控制台 / Web 介面 / 復盤報告]
    J -->|LightMem 寫入| M[(FluxMem 記憶圖)]
    M -->|Stage I 召回| P
    M -->|蒸餾技能| R{RSI 受保護評測}
    R -->|通過：新版本| P
```

```
lingxi/
├── kernel/      主迴圈 Kernel、RunContext、進展預算、滾動記憶（分段摺疊）、鉤子、任務模式、預處理、反覆查證
├── senses/      Playwright 會話、頁面快照腳本、元素定位證據鏈、視覺神諭
├── skills/      web_open/click/type/select/key/scroll/read/nav/wait/look、搜尋、Python、檔案、計畫、MCP、Daytona
├── knowledge/   時間錨定、站點手冊
├── memory/      自我學習記憶：LightMem 寫入管線、FluxMem 三層記憶圖與三階段演化、recall 技能
├── evolve/      遞迴自我改進：版本化系統狀態、受保護評測、改進器與策略、自主歸屬帳本
├── retrieval/   LlamaIndex 混合檢索（MatrixVectorStore）、長網頁精讀聚焦、搜尋快取
├── journal/     黑匣子、控制台渲染、復盤報告
├── llm/         OpenAI 相容客戶端（DashScope / DeepSeek / Ollama / OpenAI …）、KV 快取量測
├── bench.py     lingxi bench 加速基準
├── hanzi.py     繁簡處理：一律繁體呈現、比對繁簡通吃（對照表由 scripts/gen_hanzi.py 產生）
└── web/         FastAPI ＋ SSE 單頁介面
playbooks/       攜程機票 / SEO 審查 / 競品分析 手冊（新增場景＝新增一個 TOML；學到的手冊放在 evolve/playbooks/vN）
tests/           離線「迷你攜程」夾具 ＋ 假模型端到端測試（86 項）
docs/            課程筆記整理 · 架構設計 · GitHub 差異化分析 · 自我學習與加速
```

## 快速開始

```bash
git clone https://github.com/JingLunHuang/OpenManus.git
```

```bash
cd OpenManus
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -e ".[web,rag,dev]"
```

```bash
.venv\Scripts\python -m playwright install chromium
```

設定金鑰（二選一）：複製 `.env.example` 為 `.env` 填入 `DASHSCOPE_API_KEY`，或設定同名環境變數。
要換模型／端點時，複製 `config/lingxi.example.toml` 為 `config/lingxi.toml` 修改。

```bash
.venv\Scripts\python -m lingxi doctor
```

執行任務（和 OpenManus 一樣也可以用 `python main.py`）：

```bash
.venv\Scripts\python -m lingxi run "查詢 6月26日 從上海到北京的機票"
```

啟動 Web 介面（http://127.0.0.1:8765）：

```bash
.venv\Scripts\python -m lingxi serve
```

### 不需要 API Key 也能體驗的部分

```bash
.venv\Scripts\python -m lingxi playbooks "查詢 6月26日 從上海到北京的機票"
```

試跑知識層：看到時間錨定、任務模式、命中的手冊和編譯好的直達 URL（Web 介面「手冊」頁也能試跑）。

```bash
.venv\Scripts\python -m lingxi bench
```

加速基準：KV 前綴重用、LlamaIndex 與純 Python 檢索、長網頁精讀聚焦、搜尋快取（確定性，不呼叫模型）。

```bash
.venv\Scripts\python -m pytest -q
```

86 項測試：在離線夾具頁上重現攜程的 div 日曆、聯想回滾、登入遮罩；繁體指令定位簡體網頁；
用 `ScriptedLLM` 跑完整任務與「退回 → 補查 → 放行」的查證循環；記憶的寫入、睡眠整理、跨執行召回、並行寫入與投毒防護；
RSI 的接受、拒絕、繼承與回滾——全部不產生任何模型費用。

## Web 介面

設計語彙參考 [joshhu/uitest](https://github.com/joshhu/uitest) 的 **AI-Native**（紫色漸層、非對稱圓角對話泡泡、思考中跳點）與 **Bento Box**（大圓角磚塊網格），
以開發框架為主體組織成六個頁面：

| 頁面 | 內容 |
|---|---|
| 總覽 | 任務輸入、技能／手冊／紀錄／查證統計、核心迴圈、與 OpenManus 的對照、模型與環境 |
| 執行 | 即時時間線：思考、技能呼叫、定位方式與驗證徽章、反覆查證卡片、步數預算環、發現板 |
| 技能 | 全部內建技能與參數說明（由框架自我描述 API 產生） |
| 手冊 | 試跑知識層、任務模式、站點手冊 |
| 紀錄 | 黑匣子中的每一次執行，點開即可回放 |
| 學習 | 三層記憶圖、試召回、程序技能與 PEMS 曲線、RSI 版本時間線（可回滾）、每輪候選與受保護評測、自主等級、KV 前綴重用率、加速基準 |

支援淺色／深色、手機寬度。

![手冊頁：試跑知識層（深色）](docs/images/ui-playbooks-dark.png)

## 命令一覽

| 命令 | 作用 |
|---|---|
| `lingxi run "任務"` | 執行任務（`--headless` 無頭，`--debug` 每步存截圖與 HTML） |
| `lingxi serve` | Web 介面 |
| `lingxi playbooks ["任務"]` | 列出手冊／試跑知識層 |
| `lingxi replay [latest\|<執行目錄>] --open` | 把一次執行渲染成離線復盤報告 |
| `lingxi runs` | 最近的執行紀錄 |
| `lingxi doctor` | 檢查設定、金鑰、Chromium、LlamaIndex |
| `lingxi memory [stats\|recall "任務"\|ingest\|sleep --mode rules\|llm]` | 記憶統計、試召回、補寫歷史執行、睡眠整理 |
| `lingxi evolve [status\|round\|rollback N]` | RSI：查看版本、跑一輪自我改進、回滾 |
| `lingxi bench` | 加速基準（離線） |

## 擴充

```python
from lingxi import LingXi
from lingxi.kernel.hooks import default_hooks

agent = LingXi(extra_skills=[MySkill()], hooks=[MyHook(), *default_hooks()])
result = await agent.run("任務")
```

- 技能：一個 pydantic 參數模型 ＋ 一個 `run()`，Schema 與校驗自動完成；
- 鉤子：`on_start()` 往系統提示加內容（整次不變，KV 快取友善），`brief()` 往每步簡報裡加內容，`after_step()` 觀察每步結果，`on_finish()` 做收尾（記憶寫入就是一個鉤子）；
- 手冊：`playbooks/` 下新增 TOML；MCP：設定 `[mcp.servers.<name>]`；雲沙箱：設定 `DAYTONA_API_KEY`；
- 查證：`[verify]` 可調整退回輪數與適用的任務模式。

範例見 [`examples/`](examples/)：`extend_lingxi.py`（自訂技能與鉤子）、`vision_grounding.py`（視覺定位）、`daytona_hello.py`（雲沙箱）。

## 文件

- [01 · 課程筆記整理](docs/01-課程筆記整理.md) —— OpenManus 的結構、執行流程、三個實戰案例與除錯過程、兩種增強策略、課堂問答
- [02 · 靈犀架構設計](docs/02-靈犀架構設計.md) —— 每個機制的設計動機、取捨與程式碼位置
- [03 · GitHub 差異化分析](docs/03-GitHub差異化分析.md) —— 與同源衍生專案、更廣生態的對比（含二次查證紀錄）
- [04 · 自我學習與加速](docs/04-自我學習與加速.md) —— LightMem ＋ FluxMem、RSI、KV 快取、LlamaIndex 的設計、與論文的對應、實測與限制


## License

MIT
