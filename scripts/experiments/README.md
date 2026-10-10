# W1 · 有記憶／無記憶對照實驗（實驗基建，里程碑 M1）

一個命令跑完「同一批任務 × {memory off, memory on}」兩組對照，量測成功率、步數、
模型呼叫數、token（含快取命中）、查證輪數與牆鐘時間，並產出
`experiments/report.md` 對照報告。對應 `docs/05-下一階段規劃書.md` 的 W1。

**預設完全離線**：模型由 `lingxi/evaluation/scripted.py` 的 ScriptedLLM（假模型）依固定劇本扮演，
網頁由 `lingxi/evaluation/fixtures` 的夾具頁經本地 HTTP 伺服器提供——不需要任何 API key、
不產生任何付費呼叫。離線數字證明的是「實驗管線正確、可重現」；
「記憶對真實模型的增益」要等 M2 接真實端點後才有意義（報告中會誠實註明）。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `tasks.py` | 任務集：main（12 題，含 ≥6 題同站重複）、smoke（2 題冒煙，測試用） |
| `run_experiment.py` | 執行器：跑任務 × 兩組對照、雙重成本護欄、寫紀錄 |
| `report.py` | 報表：把一場實驗的紀錄寫成 Markdown 對照報告 |
| `../../lingxi/evaluation/` | 共用離線劇本模型與網頁夾具，不依賴測試套件 |

## 離線乾跑（不需要任何設定）

```powershell
# 冒煙：2 題、不需瀏覽器，秒級跑完
.venv\Scripts\python scripts\experiments\run_experiment.py --suite smoke

# 完整任務集：12 題 × 兩組（瀏覽器任務需要 Chromium：python -m playwright install chromium）
.venv\Scripts\python scripts\experiments\run_experiment.py --suite main

# 只重新產生報告（不重跑實驗）
.venv\Scripts\python scripts\experiments\report.py            # 自動找最新一場
.venv\Scripts\python scripts\experiments\report.py --run-dir experiments\runs\<時間戳>
```

產出在 `experiments/runs/<時間戳>/`：

```
meta.json        實驗配置、護欄上限、是否中止
records.jsonl    每次執行一筆（含黑匣子 run_dir，可用 lingxi replay 復盤）
summary.json     兩組彙總與逐題明細
report.md        這一場的對照報告（同時更新 experiments/report.md）
site/            夾具網站
memory-off/      無記憶組的 runs、workspace、evolve（各自隔離）
memory-on/       有記憶組的 runs、workspace、memory（記憶圖）、evolve
```

## 接真實模型（M2）

前提：`config/lingxi.toml` 已配好 `[llm]`，且環境變數裡有 API key
（`DASHSCOPE_API_KEY`，或 `LINGXI_API_KEY`，或 `[llm] api_key_env` 指定的變數）。

```powershell
# provider 非 scripted 即走真實端點；每題跑 3 次、成功率取多數（W1 正式配置）
.venv\Scripts\python scripts\experiments\run_experiment.py --suite main --provider dashscope --repeats 3

# 臨時換模型或端點
.venv\Scripts\python scripts\experiments\run_experiment.py --suite main --provider deepseek `
    --base-url https://api.deepseek.com --model deepseek-chat --api-key-env DEEPSEEK_API_KEY

# 也可用環境變數切換（預設 provider）
$env:LINGXI_EXPERIMENT_PROVIDER = "dashscope"
```

真實模型模式下沿用 `config/lingxi.toml` 的全部行為（查證閘門、記憶睡眠用模型整理等），
找不到金鑰會直接报错退出，不會誤跑成半套配置。

## 成本護欄（雙重）

| 層級 | 旗標 | 預設 | 行為 |
| --- | --- | --- | --- |
| 單次執行 | `--max-calls-per-run` | 80 | 模型呼叫數超限：中止該次執行，`budget_exceeded` 記原因 |
| 單次執行 | `--max-tokens-per-run` | 300000 | token 超限：同上 |
| 單次執行 | `--max-seconds-per-run` | 600 | 牆鐘超時：同上 |
| 整個實驗 | `--max-total-calls` | 800 | 總呼叫超限：中止後續所有執行，`meta.json.aborted` 記原因 |
| 整個實驗 | `--max-total-tokens` | 3000000 | 總 token 超限：同上 |

M2 開跑前請依預算調整這五個數字（規劃書 §4：總預算上限由你決定）。
護欄中止不是失敗掩蓋：被中止的執行照常寫進 `records.jsonl`，報告會列出原因。

## 其他常用旗標

- `--arms off,on`：只跑某一組（如 `--arms on`）；
- `--repeats N`：每題重複次數（正式實驗建議 3，多數決見 `summary.json` 的 `majority`）；
- `--evolve-dir <路徑>`：兩組共用一個 RSI 狀態目錄（預設各自從 v0 開始、互相隔離）；
- `--headed`：瀏覽器不以無頭模式跑（除錯用）；
- `--out-dir <路徑>`：改產出根目錄（測試用）。

## 設計備註

- **兩組隔離**：memory-off / memory-on 各有自己的 workspace、runs、記憶圖與 evolve 狀態，
  都在該場實驗目錄內，絕不寫進專案目錄；memory-on 組的記憶在整場實驗中跨題累積
  （同站重複任務先跑過的經驗，後面的題目才召回得到——任務順序是刻意安排的）。
- **離線劇本**：每題附一段 ScriptedLLM 劇本（`tasks.py`）；`web_read` 內部的資訊提取呼叫
  （無工具）以 callable 劇本項攔截。
  離線模式關閉了查證閘門（它會多消耗一次模型呼叫，劇本未為它編排），真實模式則照常開啟。
- **成功判定**：`status == success` 且答覆包含該題全部 `expect` 子串，且未觸發護欄。
- **只透過公開 API**：執行器只用 `LingXi` / `Settings` / `load_settings` 與
  `Journal.load` 讀黑匣子，不碰 lingxi 內部。
