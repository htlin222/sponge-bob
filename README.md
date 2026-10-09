# sponge-bob

**把一整個 Google Drive 資料夾的海綿寶寶（台灣配音）自動加上繁體中文 SRT 字幕。全部在雲端跑。**

[![ci](https://github.com/htlin222/sponge-bob/actions/workflows/ci.yml/badge.svg)](https://github.com/htlin222/sponge-bob/actions/workflows/ci.yml)
[![batch](https://github.com/htlin222/sponge-bob/actions/workflows/batch.yml/badge.svg)](https://github.com/htlin222/sponge-bob/actions/workflows/batch.yml)

建立在 [polish-screen-record](https://github.com/htlin222/polish-screen-record) 的架構上：**先斷句、再定時**，讓 LLM 永遠碰不到時間碼。
這個 repo 把它從「一個 issue 一支影片」改成「一個資料夾 228 集」。

```
GitHub Actions（排程 / 手動）
  │
  ├─ 列出 Drive 資料夾 ── 影片旁已有 .zh-Hant.srt 的集數跳過（Drive 狀態 = 待辦清單）
  │
  ├─ 開一台 Colab T4，整批共用 ── VM 直接從 Drive 抓影片，runner 不碰 GB 級檔案
  │    └ 每集：faster-whisper large-v3 → words.json（立刻寫進 _psr/，GPU 成果先落地）
  │
  ├─ 加標點 ── DeepSeek 只插入標點，去標點後須與原文逐字相同
  ├─ 對回時間 → 修整 → 驗證 ── 純函式
  │
  └─ 寫回 Drive ── <集名>.zh-Hant.srt 放在影片旁邊，最後才寫（存在 = 這集完成）
```

## 跟 polish-screen-record 的差異

| | polish-screen-record | sponge-bob |
| --- | --- | --- |
| 觸發 | issue + label，一次一支 | `workflow_dispatch` + 每 6 小時排程 |
| 待辦清單 | issue 內文 | Drive 資料夾本身 |
| Colab | 每支影片開一次 T4 | 一個 session 跑整批，模型留在 kernel 裡 |
| 中斷續跑 | 重跑整支 | 轉錄先落地 `_psr/`，下一輪只補加標點 |
| 失敗隔離 | — | 單集失敗繼續；連續 3 集失敗或 GPU 配額用完就停，等下一輪 |
| 術語表 | 講課用詞 | 角色與地名（全部取自本資料夾集名） |
| Groq 備援 | 有 | 選用（有 `GROQ_API_KEY` 才啟用） |

公開 repo 沒有 issue 觸發面：外人無法讓 workflow 執行，資料夾 ID 也放在 secret 裡。

## Drive 上的產物

```
海綿寶寶/
├── 海綿寶寶_S01_ep001_急徵店員_海底清潔隊_我的好朋友.mp4
├── 海綿寶寶_S01_ep001_急徵店員_海底清潔隊_我的好朋友.zh-Hant.srt   ← 播放器會自動載入
├── …
└── _psr/                                    ← 中間產物，除錯與續跑用
    ├── <集名>.words.json      逐字時間戳
    ├── <集名>.raw.srt         未加標點的機械斷句（對照用）
    └── <集名>.manifest.json   stage key、引擎、GPU、耗時、成本
```

## 使用

**Actions → batch → Run workflow**，或等排程。

| 輸入 | 說明 |
| --- | --- |
| `limit` | 本輪最多幾集，`0` = 全部（受 5 小時時間預算限制） |
| `dry_run` | 不寫入 Drive，報告裡附第一集前 6 條預覽。調術語表時用 |

每輪結果寫在該次執行的 **Summary**：每集字幕數、加標點失敗塊數、原文覆蓋率、驗證違規數。

本機也能跑同一個指令（運算一樣在 Colab）：

```bash
uv sync --frozen
DRIVE_FOLDER_ID=... DEEPSEEK_API_KEY=... DRY_RUN=true uv run psr batch --limit 1
```

## 憑證

| Secret | 內容 | 壽命 |
| --- | --- | --- |
| `GOOGLE_OAUTH_TOKEN` | Drive OAuth token（`drive.readonly` + `drive.file`） | 長效，見下方 |
| `GCP_ADC_JSON` | `gcloud auth application-default login` 產生的 ADC，給 Colab CLI | 長效 |
| `DEEPSEEK_API_KEY` | 加標點 | — |
| `DRIVE_FOLDER_ID` | 要處理的資料夾 | — |
| `GROQ_API_KEY` | 選用，Colab 拿不到 GPU 時的備援 | — |

### 為什麼 Drive token 會「每週壞一次」，以及修法

OAuth 同意畫面停在 **測試中** 時，Google 發的 refresh token **7 天就失效**（`invalid_grant`）。
排程批次一定會撞到。修法是把 app 發佈到 **實際運作中**：自用、少於 100 位使用者不需要通過驗證，只是授權時會看到「未經驗證」警告。

這些瀏覽器操作都用 ego-browser 在已登入的 Google 帳號裡完成，不用手動點 Console：

1. GCP Console → Google Auth Platform → 品牌：補上首頁、隱私權政策與授權網域，否則「發布應用程式」按鈕是灰的。
2. 目標對象 → 發布應用程式 → 確認。
3. 本機跑 `uv run python -u scripts/reauth.py`，把印出的 `AUTH_URL` 交給 ego 打開：進階 → 前往（不安全）→ 全選範圍 → 繼續。
4. `gh secret set GOOGLE_OAUTH_TOKEN < ~/.config/polish-screen-record/token.json`

之後只有在 token 被撤銷或換帳號時才需要重做第 3、4 步。

## 已知限制

- 加標點階段**不改字**。角色名誤聽靠 `glossary.yml` 的 `wrong` 在加標點前逐字修正（只接受與正字等長的對照，時間軸不動）；片尾浮水印類幻覺靠 `hallucinations` 清單刪除。兩者都只收實測看過的，從 `_psr/*.raw.srt` 統計後再補。
- **刻意關閉 VAD**：Silero VAD 會把配樂底下的卡通對白大量砍掉（S01E01 實測少掉 59%）。代價是片頭曲會被轉成（常聽錯的）歌詞，靜音段偶有幻覺字句，靠 `condition_on_previous_text=False` 把影響限制在單一窗內。
- Colab 免費層每日 GPU 配額不固定。用完時本輪提前停，下一輪排程接手，整個資料夾大約需要 2–3 輪。
- `colab exec` 是否沿用同一個 kernel 未經實測；若不沿用，模型每集重載一次（多 30–60 秒），結果不變。

## 授權

MIT。核心 pipeline 來自 polish-screen-record。
