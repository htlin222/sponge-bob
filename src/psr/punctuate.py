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
import time

from psr.text import normalize, to_traditional

MODEL = "deepseek-v4-flash"
CHUNK_CHARS = 1500
# 相似度門檻。0.995 在實測中會擋掉三塊只差幾個字的正常輸出（模型偶爾會
# 修掉一個明顯的口誤），0.98 仍足以攔住任何有意義的內容遺失。
MIN_SIMILARITY = 0.98
RETRY_DELAYS = (3, 10, 30)

SYSTEM = (
    "你的唯一任務是替中文逐字稿加上標點符號。\n"
    "絕對規則：除了插入標點（，。！？、）之外，一個字都不可以改、刪、加。\n"
    "把標點放在語意自然停頓處：完整句子結束用句號，句中停頓用逗號。\n"
    "只輸出加了標點的文字本身，不要任何說明、不要 JSON、不要編號。"
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


def make_chunks(words, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for word in words:
        buf.append(word.text)
        size += len(word.text)
        if size >= chunk_chars:
            chunks.append("".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("".join(buf))
    return chunks


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

        out = to_traditional((response.choices[0].message.content or "").strip())
        ratio = similarity(chunk, out)
        if ratio >= MIN_SIMILARITY:
            return out, prompt_tokens, completion_tokens, f"相似度 {ratio:.4f}"
        if delay is not None:
            sleep(delay)
    return None, prompt_tokens, completion_tokens, f"三次都不符（相似度 {ratio:.4f}）"
