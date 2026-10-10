"""照語音修正誤聽：讓 Haiku 找出同音／近音字，用拼音相似度把關後套回字幕。

Whisper 的錯幾乎都是「聽對了音、選錯了字」：謝老闆（蟹老闆）、好媽雞（好麻吉）、
平價橋市（平價超市）。這類錯從文字上下文就能修回來，而且修正前後**唸起來一樣**。
所以規則是：

- 只在 cue 文字層級替換，cue 的起訖時間完全不動。
- 每筆修正必須是該條字幕裡原樣出現的片段，且前後拼音（不含聲調）相似度
  ≥ SOUND_THRESHOLD。模型想「順便潤飾」或憑劇情猜台詞（例如把「我做不得白大姨」
  改成「派大星」，拼音相似度 0.42）都會被擋掉——那不是聽錯，是改寫。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from pypinyin import lazy_pinyin

from psr.models import Cue
from psr.text import to_traditional

FIX_VERSION = "4"  # 2：先轉繁（加標點失敗的塊漏了繁化）；3：先套 glossary 的 wrong 對照；4：glossary 反覆套用到穩定
# 由實測誤聽校準：佳音哥→章魚哥 0.57 要放行，我做不得白大姨→派大星 0.42 要擋。
SOUND_THRESHOLD = 0.55
MAX_LENGTH_CHANGE = 2
MAX_GLOSSARY_PASSES = 3

SYSTEM_PROMPT = """你是《海綿寶寶》台灣配音版的字幕校對。字幕由語音辨識產生，常把字音聽對、字選錯。

輸入每行格式為「編號|台詞」。請找出「聽錯的字」並給出正確寫法：
- 只修同音或近音的誤聽：角色名、地名、專有名詞、慣用語、明顯選錯的同音字。修正前後唸起來必須相同或非常接近。
- wrong 必須是該條字幕裡一字不差出現的片段，盡量短（只包含錯字與必要的相鄰字）。
- 不要潤飾、不要改語氣詞、不要改標點、不要補漏字、不要憑劇情改寫整句。沒把握就不要改。
- 片頭曲歌詞若能認出原詞，可以照原詞修正。

台灣配音版的正確譯名：{terms}"""


SCHEMA = {
    "type": "object",
    "properties": {"fixes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "cue": {"type": "integer"},
            "wrong": {"type": "string"},
            "right": {"type": "string"},
        },
        "required": ["cue", "wrong", "right"],
    }}},
    "required": ["fixes"],
}


def system_prompt(terms: list[str]) -> str:
    return SYSTEM_PROMPT.format(terms="、".join(terms))


def user_prompt(cues: list[Cue]) -> str:
    return "\n".join(f"{c.index}|{c.text.replace(chr(10), ' ')}" for c in cues)


def sound_similarity(a: str, b: str) -> float:
    pa, pb = " ".join(lazy_pinyin(a)), " ".join(lazy_pinyin(b))
    return SequenceMatcher(None, pa, pb).ratio()


@dataclass(frozen=True)
class Fix:
    cue: int
    wrong: str
    right: str


def traditionalize(cues: list[Cue]) -> list[Cue]:
    """確定性地把字形統一成台灣繁體（s2tw），補救 pipeline 漏掉繁化的字幕。"""
    return [replace(c, text=to_traditional(c.text)) for c in cues]


def apply_glossary(cues: list[Cue], pairs: list[tuple[str, str]]) -> tuple[list[Cue], list[Fix]]:
    """把 glossary 的 (誤聽, 正字) 確定性地套到每條字幕，回傳 (新字幕, 修正紀錄)。

    Haiku 逐集判斷，同一個錯字常常這集修、那集漏；反覆出現的誤聽改由對照表統一處理。
    長的先換，「美味謝寶」才不會先被「謝寶」吃掉一半。時間不變，也不受拼音門檻限制——
    對照表是人工審過的。
    """
    ordered = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
    out: list[Cue] = []
    applied: list[Fix] = []
    for c in cues:
        text = c.text
        # 替換會連鎖：「每位謝寶」先被「謝寶→蟹堡」換成「每位蟹堡」，下一輪才輪到它。
        for _ in range(MAX_GLOSSARY_PASSES):
            before = text
            for wrong, right in ordered:
                n = text.count(wrong)
                if n:
                    text = text.replace(wrong, right)
                    applied.extend([Fix(c.index, wrong, right)] * n)
            if text == before:
                break
        out.append(replace(c, text=text))
    return out, applied


def apply_fixes(cues: list[Cue], proposals: list[dict]) -> tuple[list[Cue], list[Fix], list[Fix]]:
    """套用通過檢查的修正，回傳 (新字幕, 已套用, 被擋下)。時間不變。"""
    texts = {c.index: c.text for c in cues}
    applied: list[Fix] = []
    skipped: list[Fix] = []
    for p in proposals:
        fix = Fix(p.get("cue"), (p.get("wrong") or "").strip(), (p.get("right") or "").strip())
        ok = (fix.cue in texts and fix.wrong and fix.right and fix.wrong != fix.right
              and fix.wrong in texts[fix.cue]
              and abs(len(fix.wrong) - len(fix.right)) <= MAX_LENGTH_CHANGE
              and sound_similarity(fix.wrong, fix.right) >= SOUND_THRESHOLD)
        if not ok:
            skipped.append(fix)
            continue
        texts[fix.cue] = texts[fix.cue].replace(fix.wrong, fix.right)
        applied.append(fix)
    return [replace(c, text=texts[c.index]) for c in cues], applied, skipped
