"""第二階段：把加了標點的文字對回 word 時間軸，並斷成字幕。

這是純函式，零網路。因為第一階段保證「去掉標點後與原文逐字相同」，這裡
只需要走一遍指標：每消耗一個非標點字元就往前推進一個原文字元，它所屬的
word 就給出時間。**不需要 diff、不需要錨點、不可能漂移。**
"""

from psr.models import Cue, Word
from psr.text import display_width, normalize

SENTENCE_END = "。！？!?…"
SOFT_BREAK = "，,、；;：:"
PUNCTUATION = SENTENCE_END + SOFT_BREAK

MAX_WIDTH = 30.0
SOFT_MIN_WIDTH = 14.0
# 字幕內部若跨越比這更長的靜音就切開。只看標點會產生橫跨整段停頓的字幕
# ——實測出現過一條 29.9 秒的，講者早就講完下一句了它還掛在螢幕上。
MAX_INTERNAL_GAP = 1.5


def _char_index(words: list[Word]) -> tuple[list[tuple[float, float]], list[str]]:
    """原文的每個字元 → (起始時間, 結束時間)。

    時間在 word 內線性內插，而不是整個 word 共用一組時間。理由是斷點會落在
    word 內部：Whisper 對中文輸出的是多字詞，而斷句依標點走，兩者不對齊是
    常態。若前後兩條字幕都引用同一個 word 的 start/end，就會直接重疊。
    內插讓「上一條的結尾」與「下一條的開頭」在建構上就不可能交叉。

    正規化後會消失的字元（空白）不佔位——第一階段的比對也是在正規化後的
    文字上做的，兩邊必須用同一把尺。
    """
    spans: list[tuple[float, float]] = []
    chars: list[str] = []
    for word in words:
        kept = [c for c in word.text if normalize(c)[0]]
        if not kept:
            continue
        step = (word.end - word.start) / len(kept)
        for offset, char in enumerate(kept):
            spans.append((word.start + offset * step, word.start + (offset + 1) * step))
            chars.append(char)
    return spans, chars


def build_cues(
    words: list[Word],
    punctuated: str,
    max_width: float = MAX_WIDTH,
    soft_min: float = SOFT_MIN_WIDTH,
    max_gap: float = MAX_INTERNAL_GAP,
) -> list[Cue]:
    """把加了標點的文字切成字幕，時間直接取自 word。

    斷點的三個來源，依序判斷：

      句尾標點（。！？）  一句話講完就換一條。
      句中標點（，、；）  且累積寬度已達 soft_min——太早在逗號斷會切出
                          讀不完整的半句。
      靜音                 下一個字距離目前結尾超過 max_gap 秒。
      寬度上限             真的沒有標點可依循時的最後手段。
    """
    spans, _ = _char_index(words)
    cues: list[Cue] = []
    buffer: list[str] = []
    start_time: float | None = None
    end_time = 0.0
    cursor = 0

    def flush() -> None:
        nonlocal buffer, start_time
        text = "".join(buffer).strip()
        if text and start_time is not None:
            cues.append(Cue(len(cues) + 1, start_time, end_time, text))
        buffer, start_time = [], None

    for char in punctuated:
        if char in PUNCTUATION:
            if start_time is None:
                continue                      # 開頭的孤兒標點，直接丟掉
            buffer.append(char)
            if char in SENTENCE_END or display_width("".join(buffer)) >= soft_min:
                flush()
            continue

        if not normalize(char)[0]:
            continue                          # 空白等在正規化後消失的字元

        if cursor >= len(spans):
            break

        char_start, char_end = spans[cursor]
        # 跨越長靜音就先收尾，否則字幕會橫跨整段停頓掛在螢幕上。
        if start_time is not None and char_start - end_time > max_gap:
            flush()

        if start_time is None:
            start_time = char_start
        end_time = char_end
        buffer.append(char)
        cursor += 1

        if display_width("".join(buffer)) >= max_width:
            flush()

    flush()
    return cues


def coverage(words: list[Word], cues: list[Cue]) -> float:
    """字幕實際涵蓋了多少比例的原文字元。

    這是這個架構最有力的健康指標：低於 1.0 就代表有內容在對回時間的過程中
    掉了。先前用 diff 對齊時無法這樣檢查，因為那條路本來就允許文字與原文
    不同。
    """
    total = len(_char_index(words)[1])
    if not total:
        return 1.0
    used = sum(len(normalize(c.text)[0]) for c in cues)
    return min(1.0, used / total)
