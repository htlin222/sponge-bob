"""把 ASR 逐字稿丟給 LLM 潤稿，並用結構化的方式限制它能造成的傷害。

整體策略：把逐字稿切成不重疊的窗口分別送給 DeepSeek，每個窗口的輸出都
用「刪除/插入比例分開量測」的方式驗收；不過關就重試，重試三次仍不過關
就讓這個窗口整個放棄（呼叫端改用 psr.segment.raw_segment 走無 LLM 的
決定性降級路徑）。每一項設計都是實測 87 分鐘教學影片逐字稿後留下的結果，
細節理由寫在各自的函式旁邊，不要為了「看起來更聰明」而改回去。
"""

from __future__ import annotations

import difflib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import OpenAI

from psr.glossary import Glossary
from psr.models import Word
from psr.text import normalize

MODEL = "deepseek-v4-flash"

# 每個窗口約 1800 個 word 文字字元，不重疊。窗口銜接處的斷句品質，交給
# 後製的 merge_short_cues 去補——那一步本來就是跨窗口邊界運作的。
WINDOW_CHARS = 1800

# 用於 context-echo 剝除的相鄰窗口文字長度（見 _strip_context_echo）。
CONTEXT_CHARS = 250

# 刪除比例上限 12%、插入比例上限 15%——刻意不對稱。單一對稱的 edit
# ratio 會把「修掉贅字」跟「憑空捏造」混成同一個數字，但這兩者的風險
# 完全不同：刪除會掉內容、插入會捏造內容。插入的容忍度刻意設得比刪除
# 高，因為 Whisper 常把英文單字切成 BPE 子詞片段（prompt → Promp），
# 模型把缺的字母補回來是正確行為，會在插入比例上正常地偏高。
MAX_DELETION_RATIO = 0.12
MAX_INSERTION_RATIO = 0.15

MAX_ATTEMPTS = 3
# 固定退避秒數，不加 jitter——重試次數少（3 次）、窗口彼此獨立，
# jitter 帶來的「避免同時打 API」好處在這個規模下可忽略，固定值反而
# 讓行為好預測、好測試。
RETRY_BACKOFF_SECONDS = (3, 10, 30)

# 去重只處理「完全相同」的行，且只處理正規化後長度 >= 8 的行。
# 原本想法是「內容已出現在前面某行就刪」，但「好的」「那」這種短行
# 幾乎必然是某個長行的子字串，會被誤刪；內容缺一塊之後 align 就對不上，
# 整個窗口跟著降級。長度門檻把這類合法短行留住，只清掉模型偶爾在窗口
# 尾端整段重述前文的情形。
DUPLICATE_MIN_NORM_LEN = 8

SYSTEM_PROMPT = """你是繁體中文字幕潤稿員。把語音辨識的逐字稿整理成通順、好讀的字幕行。

必須遵守：
1. 輸出繁體中文（台灣用語）。
2. 補上標點符號（，。？！），讓句子讀起來自然。
3. 依語意斷行。每行 12～20 個全形字寬，一行放一個完整的意群。
   不要切得太碎——「大家好」「我是林醫師」這種三五個字一行是錯的，
   應該合併成「大家好，我是林醫師」。
4. 修正明顯的同音錯字（例如「旨令」→「指令」、「派森」→ Python）。
5. 可以修掉少量口語贅字，讓句子讀起來乾淨——但只限這些：
   語助詞（呃、嗯、欸）、無意義的口頭禪（那個、就是說、對、然後那個）、
   結巴造成的立即重複（「我我我覺得」→「我覺得」）。
   拿捏原則：刪掉之後語意必須完全不變，讀者不會察覺少了什麼。

絕對禁止：
- 不可刪除有內容的重複。講者重述同一個論點或示範步驟是刻意強調，
  那是真實內容不是冗餘。原文講了三次就保留三次。
- 不可改寫、濃縮或摘要。除了上面第 5 點列的贅字，其餘一字不動。
- 不可憑空補充原文沒有的資訊。
- 不可輸出任何時間碼、數字編號或說話者標籤。
- 英文技術術語保持英文原形，不要翻譯也不要音譯。

只輸出 JSON：{"lines": ["第一行", "第二行", ...]}"""


def _norm(s: str) -> str:
    return normalize(s)[0]


def _build_system_prompt(glossary: Glossary) -> str:
    hint = glossary.polish_hint()
    if not hint:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + (
        "\n\n這份逐字稿的專有名詞對照表（左邊是正確寫法，右邊是語音辨識常見的"
        "錯誤，看到右邊的請改成左邊）：\n" + hint
    )


def edit_budget(original: str, polished: str) -> tuple[float, float]:
    """回傳 (刪除比例, 插入比例)，兩者都相對 normalize 後的原文長度計算。

    對稱的 1-ratio 把「刪掉贅字」和「憑空捏造」混成同一個數字，但兩者的
    風險完全不同：刪除會掉內容、插入會捏造內容，容忍度本來就該不一樣。
    分開計量才能說「可以修掉一點贅言，但不准無中生有」。

    autojunk=False 是必要的：預設的 autojunk 啟發式會把長序列裡出現
    頻率 >1% 的元素當雜訊丟掉，中文逐字稿裡「的」「是」這類高頻字幾乎
    必中，一旦被丟掉，equal 區塊會靜默地變少變爛，比例算出來會不準
    但不會報錯。
    """
    na, nb = _norm(original), _norm(polished)
    if not na:
        return 0.0, 0.0
    matcher = difflib.SequenceMatcher(None, na, nb, autojunk=False)
    deleted = inserted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
        elif tag == "replace":
            deleted += max(0, (i2 - i1) - (j2 - j1))
            inserted += max(0, (j2 - j1) - (i2 - i1))
    return deleted / len(na), inserted / len(na)


def make_windows(words: list[Word], window_chars: int = WINDOW_CHARS) -> list[list[Word]]:
    """把 word 清單切成不重疊的窗口，每個窗口的 word 文字總長度約
    window_chars 字元（累積到 >= window_chars 就收）。刻意不重疊——
    重疊窗口會讓同一段內容被潤兩次，銜接處要判斷「哪個版本才是準的」，
    徒增一類全新的 bug；窗口接縫的斷句品質由後製的 merge_short_cues
    負責，那一步本來就是跨窗口邊界運作的。"""
    windows: list[list[Word]] = []
    current: list[Word] = []
    length = 0
    for word in words:
        current.append(word)
        length += len(word.text)
        if length >= window_chars:
            windows.append(current)
            current, length = [], 0
    if current:
        windows.append(current)
    return windows


def _strip_context_echo(lines: list[str], before: str, after: str) -> list[str]:
    """剝掉上下文回音。

    這個窗口送給模型的 user prompt **不含**任何前後文（見
    _polish_one_window 裡的說明），但模型偶爾仍會在開頭或結尾吐出一段
    跟相鄰窗口內容雷同的文字。這段文字在「這個窗口的原文」裡找不到
    對應，align 會因為找不到錨點而讓整個窗口降級。既然相鄰窗口的原文
    是已知的，用它做確定性比對、直接剝除，比修 prompt 去求模型「不要
    這樣做」可靠——prompt 上的約束模型不保證會聽。
    """
    nb, na = _norm(before), _norm(after)
    out = list(lines)
    while out and nb and _norm(out[0]) and _norm(out[0]) in nb:
        out.pop(0)
    while out and na and _norm(out[-1]) and _norm(out[-1]) in na:
        out.pop()
    return out


def _dedupe_exact_lines(lines: list[str]) -> list[str]:
    """只刪「完全相同」的行，且只處理正規化後長度 >= DUPLICATE_MIN_NORM_LEN
    的行。理由見模組頂端 DUPLICATE_MIN_NORM_LEN 的註解。"""
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = _norm(line)
        if key and (len(key) < DUPLICATE_MIN_NORM_LEN or key not in seen):
            seen.add(key)
            out.append(line)
    return out


@dataclass(frozen=True, slots=True)
class WindowResult:
    """單一窗口的潤稿結果。lines 為 None 代表這個窗口三次重試後仍未過關，
    呼叫端應改用 psr.segment.raw_segment 做決定性降級。"""

    index: int
    lines: list[str] | None
    deletion_ratio: float | None
    insertion_ratio: float | None
    attempts: int
    error: str | None
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class PolishResult:
    """整趟潤稿的彙總資訊，供 CLI 回報成本與降級窗口數。"""

    windows: list[list[str] | None]
    degraded: list[tuple[int, str]]
    prompt_tokens: int
    completion_tokens: int

    @property
    def degraded_count(self) -> int:
        return len(self.degraded)

    def estimated_cost_usd(self) -> float:
        """DeepSeek 定價：輸入 $0.14/1M tokens、輸出 $0.28/1M tokens。"""
        return self.prompt_tokens / 1e6 * 0.14 + self.completion_tokens / 1e6 * 0.28


def _polish_one_window(
    index: int,
    windows: list[list[Word]],
    system_prompt: str,
    client,
) -> WindowResult:
    ws = windows[index]
    body = "".join(w.text for w in ws)
    before = "".join(w.text for w in windows[index - 1])[-CONTEXT_CHARS:] if index > 0 else ""
    after = (
        "".join(w.text for w in windows[index + 1])[:CONTEXT_CHARS]
        if index + 1 < len(windows)
        else ""
    )
    # 刻意不送前後文。實測（2026-08-16）：加了唯讀上下文之後模型會崩潰——
    # 該回 105 行只回 2 行，或直接吐出格式錯誤的 JSON。本來是想讓窗口接縫
    # 處的斷句自然一點，但代價是整個窗口陣亡，划不來。接縫品質由後製的
    # merge_short_cues 補償，那一步本來就跨接縫運作。上面算出的 before/
    # after 不會出現在送給模型的訊息裡，只用於下面 _strip_context_echo
    # 的防禦性比對。
    user_content = body

    last_error: str | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                top_p=1,
                seed=42 + attempt,
                max_tokens=8000,
                response_format={"type": "json_object"},
                # 這個模型是 reasoning model，實測 reasoning 吃掉 completion
                # token 預算的 81%，導致空回應。關掉之後單趟快 10 倍、
                # 便宜 6 倍。
                extra_body={"thinking": {"type": "disabled"}},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens

            lines = json.loads(response.choices[0].message.content)["lines"]
            lines = [line.strip() for line in lines if line.strip()]
            lines = _strip_context_echo(lines, before, after)
            lines = _dedupe_exact_lines(lines)

            deletion_ratio, insertion_ratio = edit_budget(body, "".join(lines))
            if deletion_ratio <= MAX_DELETION_RATIO and insertion_ratio <= MAX_INSERTION_RATIO:
                return WindowResult(
                    index=index,
                    lines=lines,
                    deletion_ratio=deletion_ratio,
                    insertion_ratio=insertion_ratio,
                    attempts=attempt + 1,
                    error=None,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                )
            last_error = (
                f"刪除 {deletion_ratio:.1%}(上限{MAX_DELETION_RATIO:.0%}) "
                f"插入 {insertion_ratio:.1%}(上限{MAX_INSERTION_RATIO:.0%})"
            )
        except Exception as e:  # noqa: BLE001 - 任何一種失敗都該重試，不分類
            last_error = f"{type(e).__name__}: {str(e)[:80]}"

        if attempt + 1 < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    # token 用量無論成敗都要計入——重試打出去的每一次 API call 都要付錢，
    # 不能因為窗口最終降級就假裝那些 token 沒被消耗。
    return WindowResult(
        index=index,
        lines=None,
        deletion_ratio=None,
        insertion_ratio=None,
        attempts=MAX_ATTEMPTS,
        error=last_error,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )


def _default_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未設定，且未注入 client。"
            "測試請一律注入假 client，不要依賴這條路徑打真實網路。"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def polish_words(
    words: list[Word],
    glossary: Glossary,
    client=None,
    max_workers: int = 6,
) -> tuple[list[list[str] | None], PolishResult]:
    """把 words 切窗、逐窗潤稿，回傳 (每窗的 lines-or-None 清單, 彙總結果)。

    每個窗口彼此獨立、互不依賴（No read-only context，見
    _polish_one_window），所以可以平行送出去；用 max_workers 控制並行度，
    測試裡把它設成 1 可以拿到決定性的執行順序。client 預設用真正的
    DeepSeek OpenAI client，但一律應該注入假 client 做測試——這個函式
    本身不判斷「這是不是測試」，注入什麼就用什麼。
    """
    if client is None:
        client = _default_client()

    system_prompt = _build_system_prompt(glossary)
    windows = make_windows(words)
    if not windows:
        return [], PolishResult(windows=[], degraded=[], prompt_tokens=0, completion_tokens=0)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda i: _polish_one_window(i, windows, system_prompt, client),
                range(len(windows)),
            )
        )
    results.sort(key=lambda r: r.index)

    lines_per_window: list[list[str] | None] = []
    degraded: list[tuple[int, str]] = []
    prompt_tokens = completion_tokens = 0
    for result in results:
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        lines_per_window.append(result.lines)
        if result.lines is None:
            degraded.append((result.index, result.error or "未知錯誤"))

    polish_result = PolishResult(
        windows=lines_per_window,
        degraded=degraded,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return lines_per_window, polish_result
