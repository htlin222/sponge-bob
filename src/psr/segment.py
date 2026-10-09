import re
import unicodedata

from psr.models import Word, Cue
from psr.text import display_width, to_traditional

# 兩個東亞寬字元之間的空白。Whisper 的中文 word token 會帶前導空格
# （實測 ' 來'），直接串接會讓字幕出現「導演 來個特寫」這種夾雜空格的結果。
# 中文字幕不用空格分隔，斷句靠標點與換行。但英文術語兩側的空格必須保留
# （「安裝 Python 環境」），所以只能刪除「兩側都是寬字元」的那些空白。
_CJK_SPACE = re.compile(r"(?<=\S)\s+(?=\S)")


def _is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def clean_text(text: str) -> str:
    """整理串接 word 後的字幕文字：收斂連續空白、刪除中日韓字元之間的空白、
    去除頭尾空白，並統一成繁體中文（台灣用語）。

    Whisper 對中文會在同一份逐字稿裡混輸繁簡（實測「我们说」與「我們就會」
    並存），所以繁化不能只靠潤稿 prompt——降級路徑根本不經過 LLM。放在這裡
    才能保證**兩條路徑的輸出都是繁體**。"""
    text = to_traditional(_CJK_SPACE.sub(lambda m: " ", text).strip())
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == " " and 0 < i < len(text) - 1 and _is_wide(text[i - 1]) and _is_wide(text[i + 1]):
            continue
        out.append(ch)
    return "".join(out)


def raw_segment(words: list[Word], max_width: float = 20.0, max_gap: float = 0.6) -> list[Cue]:
    """在沒有潤稿（或潤稿降級）時，用固定規則把 word 直接切成字幕。
    純規則、決定性、不呼叫 LLM：累積寬度會超過 max_width，或跟下一個字
    之間的間隔超過 max_gap 秒，就在此處斷句。"""
    if not words:
        return []

    cues: list[Cue] = []
    current_words: list[Word] = [words[0]]

    for word in words[1:]:
        gap = word.start - current_words[-1].end
        candidate_text = "".join(w.text for w in current_words) + word.text

        if display_width(candidate_text) > max_width or gap > max_gap:
            head, carry = _split_without_breaking_ascii(current_words, word)
            cues.append(_flush(cues, head))
            current_words = carry + [word]
        else:
            current_words.append(word)

    cues.append(_flush(cues, current_words))
    return cues


def _splits_ascii_run(left: str, right: str) -> bool:
    """在 left 與 right 之間斷開，是否會切斷一個連續的 ASCII 英數串？"""
    return bool(left) and bool(right) \
        and left[-1].isascii() and left[-1].isalnum() \
        and right[0].isascii() and right[0].isalnum()


def _split_without_breaking_ascii(
    current: list[Word], nxt: Word
) -> tuple[list[Word], list[Word]]:
    """決定 current 這條字幕實際該在哪裡結束。

    Whisper 把英文輸出成 BPE 子詞片段——實測 'prompt' 是 ['prom','pt']、
    'Agent' 是 ['A','gent']、'2026' 是 ['20','26']。逐 word 累積再依寬度斷行，
    就會產生「...好的prom」／「pt」這種把英文單字劈成兩半的字幕。實測 87 分鐘
    的教學影片有 116 處。

    因此斷點若會切開一段連續的 ASCII 英數串，就往回退到該串的起點，把整串
    移到下一條字幕。整條字幕本身就是一整串英文時（退無可退）才照原樣斷開。
    """
    if not _splits_ascii_run(current[-1].text, nxt.text):
        return current, []

    i = len(current) - 1
    while i > 0 and _splits_ascii_run(current[i - 1].text, current[i].text):
        i -= 1
    if i == 0:
        return current, []          # 整條都是同一串，退無可退
    return current[:i], current[i:]


def _flush(cues: list[Cue], words_in_cue: list[Word]) -> Cue:
    return Cue(
        index=len(cues) + 1,
        start=words_in_cue[0].start,
        end=words_in_cue[-1].end,
        text=clean_text("".join(w.text for w in words_in_cue)),
    )
