# sponge-bob：給 Claude Code 的工作說明

海綿寶寶（台灣配音）字幕 pipeline ＋ 金句庫 `bob`。使用者面向的說明在 README；這裡寫的是改這個 repo 時要守的規則。

## 規則

- **公開 repo，台詞有版權**：字幕、台詞、參考庫、截圖、備份只放 Turso、Drive、`scratch/`（gitignore）或 scratchpad。
  不寫進 repo、測試 fixture、commit 訊息、Actions log 或 artifact。workflow 只印計數與統計。
- **秘密不落地**：`DEEPSEEK_API_KEY`、`DRIVE_FOLDER_ID` 只在 GitHub secret；本機沒有 DeepSeek key，要用 DeepSeek 就觸發 workflow。
  設 secret 用管線（`gh secret set X < file`），不要把值印出來。
- 直接在 `develop` commit、push，不開分支或 PR。
- 用 `uv`（`uv run …`、`uv sync --frozen`）；刪檔用 `rip`；ffmpeg 用 `command ffmpeg`。
- 測試的期望值來自規格，不是先跑程式再把輸出抄進去。

## 指令

```bash
uv run pytest -q                      # 全部測試（CI 跑這個，另檢查純函式模組沒引入網路相依）
uv run ruff check <改到的檔案>         # 既有檔案有舊的 ruff 錯誤，只檢查自己動到的
uv run psr batch --limit 1            # 字幕 pipeline（Colab T4 + DeepSeek；平常由 batch workflow 跑）
uv run bob --help                     # 金句庫指令，完整列表見 README「金句庫 bob」
```

本機憑證：

- Turso：`~/.config/sponge-bob/turso.json`（`{"url","token"}`），或 `TURSO_DATABASE_URL`／`TURSO_AUTH_TOKEN`。
- Drive：`~/.config/polish-screen-record/token.json`（`psr.cli.TOKEN_PATH`）。

## 架構

| 位置                                           | 內容                                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/psr/batch.py`、`cli.py`                   | `psr`：列 Drive → Colab Whisper → 加標點 → 對時間 → 驗證 → 寫回 `.zh-Hant.srt`        |
| `src/psr/bob.py`                               | `bob` 所有子指令（`cmd_*`）                                                           |
| `src/psr/quotedb.py`                           | Turso／libSQL 存取（Hrana HTTP）、schema、`dump` 備份                                 |
| `src/psr/quotes.py`                            | 金句標記 prompt、schema、`accept` 驗證；`PROMPT_VERSION` 改了會整批重標               |
| `src/psr/fix.py`、`refaudit.py`、`glossary.py` | 誤聽修正、`bob scan`、角色名對照表                                                    |
| `src/psr/llm.py`                               | `deepseek-*` 走 DeepSeek API，其他模型名走本機 `claude -p`                            |
| `src/psr/still.py`                             | `bob still` 截圖卡                                                                    |
| `glossary.yml`、`taxonomy.yml`                 | 誤聽對照表、封閉標籤表（都要人工審過才改）                                            |
| `.github/workflows/`                           | `ci`、`batch`（每 6 小時）、`library`（batch 後）、`health`（每天）、`backup`（每週） |
| `.claude/skills/`                              | `spongebob-quote`（找台詞）、`spongebob-still`（做圖卡）                              |

### 資料流

```
Drive 影片 ─batch─▶ <集名>.zh-Hant.srt ─bob fix─▶ 修正後 SRT（原檔留 _psr/<集名>.asr.srt）
                                        ─bob index─▶ Turso quotes/quote_tags ─bob embed─▶ quote_vecs
```

`library` workflow 在雲端對新集數跑 `fix --quiet` → `index`（DeepSeek）。

### 資料庫（Turso `bob`）

主要表：`episodes`、`cues`（全部字幕，`start_s`/`end_s`）、`quotes`（`cue_from..cue_to`、`speaker`、`quotability`、`use_when`）、
`tags`／`quote_tags`、`quote_vecs`（`use_when` 的 512 維 bge-small-zh 向量）。
修正相關：`srt_fixes`、`fix_log`、`episode_fixes`、`scan_verdicts`、`refs`、`audit_issues`、`tag_suggestions`、`punctuations`、`resegments`。

- 金句文字一律依編號從字幕切出，模型只回編號。
- `speaker` 是模型猜的、常錯；截圖時從畫面確認。
- Hrana 回傳的 blob 是沒有 padding 的 base64（`quotedb._decode` 已處理）。

## 換模型的注意事項

- Haiku 用 `quotes.SYSTEM_PROMPT`；DeepSeek 用 `quotes.system_prompt(model)`，會另加星等與 `use_when` 長度的校準。
- 調 DeepSeek prompt 後，用 `library` workflow 的 `compare` 模式對照 Turso 現有金句（只印統計、不寫入）：
  `gh workflow run library.yml -f mode=compare -f episodes="ep161 ep098 ep194 ep067 ep229"`。
  目標：★4+ 約 29%，`use_when` 平均 20 字以上。
