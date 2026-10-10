"""第一階段：替原始逐字稿加上標點。

這一步**只加標點，不改任何字**。這個限制看似綁手綁腳，實際上是整個架構
的地基：因為「去掉標點後與原文逐字相同」是可以嚴格驗證的，第二階段把
文字對回時間就變成**查表**而不是猜測——不需要 diff、不需要錨點、不可能
漂移、不會有窗口降級。

對照先前的做法：讓 LLM 同時潤稿與斷句，然後用 difflib 把結果貼回時間軸。
那條路要處理斷點映射、錨點品質、整窗降級，而且模型的斷句失控時（實測
出現過整個窗口只回 2 行、或 55% 的字幕不足兩個字元）沒有東西救得回來。
"""

import difflib
import re
import time

from psr.text import normalize, to_traditional

MODEL = "deepseek-v4-flash"
CHUNK_CHARS = 1500
# 相似度門檻。0.995 在實測中會擋掉三塊只差幾個字的正常輸出（模型偶爾會
# 修掉一個明顯的口誤），0.98 仍足以攔住任何有意義的內容遺失。
MIN_SIMILARITY = 0.98
RETRY_DELAYS = (3, 10, 30)

# 換人記號：模型判斷換了角色開口時插在新說話者的第一個字前面。它是標點類
# 字元，normalize 會丟掉，所以「只插標點」的驗證照常成立；timeline 遇到它
# 就強制斷句，一條字幕不會再混進兩個角色的話。
TURN = "／"
# 字間停頓超過這個秒數就在送給模型的文字裡換行，當作「這裡有停頓」的提示。
PAUSE_HINT_S = 0.5

SYSTEM = (
    "你替卡通的語音辨識逐字稿加標點。逐字稿是多個角色輪流說話的對白；"
    "換行代表說話中有停頓（不一定是換人）。\n\n"
    "絕對規則：只能插入標點符號 ，。！？、 與換人記號 ／，其他一個字都不可以改、刪、加、調換，"
    "也不要修正錯字。\n\n"
    "標點：\n"
    "- 一句話說完就用句號、問號或驚嘆號，不要用逗號把好幾句串在一起。"
    "疑問句用？，喊叫、驚呼、感嘆用！。\n"
    "- 句子內部的停頓才用逗號。\n"
    "換人記號：\n"
    "- 判斷換了另一個角色開口時，在新說話者的第一個字前面插入 ／（緊接在前一句的句尾標點之後）。"
    "依稱呼、問答、語氣判斷；沒有把握就不要插。\n\n"
    "只輸出加好標點的文字本身，不要任何說明、不要 JSON、不要編號；換行可以省略。"
)


def _canonical(text: str) -> str:
    """比對用的正規形式。

    必須用 psr.text.normalize 而不是自己寫個「去標點」——它會折疊繁簡。
    實測模型會在輸出中途把繁體轉成簡體，天真的逐字比對會把那判成「改了字」
    （相似度只有 0.77–0.95），但那其實只是字形轉換，內容一個字都沒動。
    """
    return normalize(text)[0]


def similarity(original: str, punctuated: str) -> float:
    return difflib.SequenceMatcher(
        None, _canonical(original), _canonical(punctuated), autojunk=False
    ).ratio()


def make_chunks(words, chunk_chars: int = CHUNK_CHARS, pause_s: float = PAUSE_HINT_S) -> list[str]:
    """把 words 串成送給模型的文字塊。字間停頓 ≥ pause_s 秒的地方換行，提示
    模型這裡可能是句尾或換人；塊也盡量切在停頓處，避免把一句話劈成兩塊。
    換行是空白，normalize 會丟掉，不影響逐字驗證。"""
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    prev_end = None
    for word in words:
        paused = prev_end is not None and word.start - prev_end >= pause_s
        if buf and size >= chunk_chars and (paused or size >= 2 * chunk_chars):
            chunks.append("".join(buf).strip())
            buf, size = [], 0
        elif paused and buf:
            buf.append("\n")
        buf.append(word.text)
        size += len(word.text)
        prev_end = word.end
    if buf:
        chunks.append("".join(buf).strip())
    return chunks


_HALFWIDTH = {",": "，", "?": "？", "!": "！", ";": "；", ":": "："}
_HALFWIDTH_AFTER_CJK = re.compile(r"(?<=[\u3400-\u9fff])[,?!;:]")


def fullwidth(text: str) -> str:
    """中文字後面的半形標點改成全形。模型偶爾混用，字幕裡「你好?」很刺眼。"""
    return _HALFWIDTH_AFTER_CJK.sub(lambda m: _HALFWIDTH[m.group(0)], text)


MIN_SPLIT_CHARS = 200


def split_at_pause(chunk: str) -> tuple[str, str] | None:
    """從最靠近中間的停頓（換行）切成兩半；太短或沒有停頓就回 None。"""
    if len(chunk) < MIN_SPLIT_CHARS:
        return None
    breaks = [i for i, ch in enumerate(chunk) if ch == "\n"]
    if not breaks:
        return None
    mid = min(breaks, key=lambda i: abs(i - len(chunk) / 2))
    return chunk[:mid], chunk[mid:]


def punctuate_chunk(chunk: str, client, sleep=time.sleep):
    """回傳 (加了標點的文字 | None, prompt_tokens, completion_tokens, 說明)。

    失敗時回傳 None，呼叫端應退回原文——沒有標點的字幕仍然可讀，
    內容被竄改的字幕不行。
    """
    prompt_tokens = completion_tokens = 0
    ratio = 0.0
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=6000,
            # 這個模型預設會推理，而推理 token 佔掉八成的輸出額度導致回傳空白。
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

        out = fullwidth(to_traditional((response.choices[0].message.content or "").strip()))
        ratio = similarity(chunk, out)
        if ratio >= MIN_SIMILARITY:
            return out, prompt_tokens, completion_tokens, f"相似度 {ratio:.4f}"
        if delay is not None:
            sleep(delay)
    return None, prompt_tokens, completion_tokens, f"三次都不符（相似度 {ratio:.4f}）"
