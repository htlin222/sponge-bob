"""轉錄後、加標點前的確定性清理。純函式，逐 word 操作，**不碰時間**。

兩件事：

1. **移除已知幻覺**。Whisper 在片尾字卡、純配樂段會吐出訓練資料裡的
   浮水印，例如「優優獨播劇場 YoYo Television Series Exclusive」
   （S01E01 實測出現在最後兩條字幕）。這些字句一字不差地重複出現在
   不同影片裡，用固定清單攔截比任何啟發式都可靠。
2. **修正角色名誤聽**。只接受「錯字與正字等長」的對照，逐字替換：
   word 的切分與時間戳完全不動，第一階段「去標點後與原文逐字相同」的
   不變量在清理之後的文字上依然成立，第二階段的查表定時不受影響。
   不等長的對照會讓字元與 word 的對應失準，所以在這裡直接略過。

比對一律在 `psr.text.normalize` 之後進行：繁簡、全半形、大小寫、標點與
空白都被折疊，「優優獨播劇場」與「优优独播剧场」、「張宇哥」與「张宇哥」
都命中同一條規則。
"""

from __future__ import annotations

from psr.models import Word
from psr.text import normalize


def _owners(words: list[Word]) -> tuple[str, list[int]]:
    """串起所有 word 的文字，回傳 (全文, 每個字元所屬的 word 索引)。"""
    owner: list[int] = []
    for i, w in enumerate(words):
        owner.extend([i] * len(w.text))
    return "".join(w.text for w in words), owner


def _matches(norm: str, needle: str):
    start = norm.find(needle)
    while needle and start != -1:
        yield start, start + len(needle)
        start = norm.find(needle, start + len(needle))


def drop_hallucinations(words: list[Word], phrases: list[str]) -> tuple[list[Word], int]:
    """刪掉被任一幻覺字句覆蓋到的 word，回傳 (剩下的 words, 刪掉的 word 數)。"""
    text, owner = _owners(words)
    norm, orig = normalize(text)
    doomed: set[int] = set()
    for phrase in phrases:
        for a, b in _matches(norm, normalize(phrase)[0]):
            doomed.update(owner[orig[k]] for k in range(a, b))
    return [w for i, w in enumerate(words) if i not in doomed], len(doomed)


def apply_corrections(words: list[Word], pairs: list[tuple[str, str]]) -> tuple[list[Word], int]:
    """把等長的誤聽逐字換成正字，回傳 (新的 words, 修正次數)。"""
    text, owner = _owners(words)
    chars = list(text)
    norm, orig = normalize(text)
    count = 0
    for wrong, correct in pairs:
        needle = normalize(wrong)[0]
        if len(needle) != len(correct):
            continue
        for a, b in _matches(norm, needle):
            # 正規化後的每個字元必須各自對到一個不同的原始字元，才能逐字替換。
            positions = [orig[k] for k in range(a, b)]
            if len(set(positions)) != len(positions):
                continue
            for pos, ch in zip(positions, correct):
                chars[pos] = ch
            count += 1
    if not count:
        return words, 0

    rebuilt: list[list[str]] = [[] for _ in words]
    for pos, ch in enumerate(chars):
        rebuilt[owner[pos]].append(ch)
    return [Word("".join(r), w.start, w.end) for r, w in zip(rebuilt, words)], count
