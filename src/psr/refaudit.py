"""用拼音找「字選錯」的誤聽：先演算法篩嫌疑，只把嫌疑交給模型。

`bob fix` 讓 Haiku 逐集讀整份字幕，但它不知道台灣觀眾怎麼稱呼劇中的東西：
「老師說，小熊」每個字都通順，它就放過了——那其實是派大星的玩偶「老實說熊熊」。
這裡改走便宜的兩段式：

1. 參考庫比對：從公開資料收集台灣配音版的台詞、名稱與迷因（存 Turso，不進 repo），
   在字幕裡找「唸起來像參考句、字卻不同」的片段。
2. 全庫同音異寫：同一串音在全部字幕裡多數寫成 A、少數寫成 B，少數那邊多半是誤聽
   （派大星 ×3000 vs 拍打星 ×4）。

兩者都只用拼音字串運算，幾秒跑完全部集數；模型只看被標出來的幾百個片段。
拼音一律去聲調，並把台灣口音與 Whisper 常混的音併在一起（zh/z、ch/c、sh/s、
-ing/-in、-eng/-en），所以「老師說」與「老實說」是同一個音。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from pypinyin import lazy_pinyin

_CJK = re.compile(r"[㐀-鿿]+")
_INITIALS = (("zh", "z"), ("ch", "c"), ("sh", "s"))
_FINALS = (("ing", "in"), ("eng", "en"))

MIN_REF_SYLLABLES = 3
MIN_REF_SCORE = 0.85
# 只差在這些字的同音異寫不算誤聽（他／她／它、的／得…），全庫比對時略過。
INTERCHANGEABLE = [
    set("他她它牠祂"),
    set("的得地"),
    set("你妳"),
    set("那哪"),
    set("麼么"),
    set("嗎嘛"),
    set("裡里"),
    set("做作"),
    set("在再"),
    set("啦喇"),
    set("哦喔噢"),
    set("呀啊阿"),
    set("吧罷"),
    set("惟唯"),
    set("彩采"),
    set("份分"),
]


def fuzzy(syllable: str) -> str:
    """去聲調後，把容易混淆的聲母韻母併成同一個。"""
    for a, b in _INITIALS:
        if syllable.startswith(a):
            syllable = b + syllable[len(a) :]
            break
    for a, b in _FINALS:
        if syllable.endswith(a):
            syllable = syllable[: -len(a)] + b
            break
    return syllable


def syllables(text: str) -> list[tuple[str, str, int]]:
    """文字裡每個漢字的 (模糊拼音, 字, 位置)。標點與非漢字略過，位置指回原字串。"""
    out = []
    for m in _CJK.finditer(text):
        for i, py in enumerate(lazy_pinyin(m.group())):
            out.append((fuzzy(py), m.group()[i], m.start() + i))
    return out


def _sound(sylls) -> str:
    return " ".join(s for s, _, _ in sylls)


@dataclass(frozen=True)
class Suspect:
    """一處嫌疑：cue 裡的 heard 可能其實是 proposed。"""

    key: tuple  # 比對用的鍵（集名, cue 編號）
    heard: str
    proposed: str
    context: str
    score: float
    reason: str  # "ref" 或 "homophone"


def ref_suspects(
    cues: list[tuple[tuple, str]], refs: list[str], min_score: float = MIN_REF_SCORE
) -> list[Suspect]:
    """在字幕裡找唸起來像參考句、但寫法不同的片段。

    cues 是 [(鍵, 文字)]。先用倒排索引投票找出可能的起點，才逐一算相似度，
    否則十萬條字幕 × 數百句參考是幾億次比對。
    """
    seqs = [syllables(text) for _, text in cues]
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ci, seq in enumerate(seqs):
        for pos, (s, _, _) in enumerate(seq):
            index[s].append((ci, pos))

    out = []
    for ref in dict.fromkeys(refs):
        rs = syllables(ref)
        n = len(rs)
        if n < MIN_REF_SYLLABLES:
            continue
        ref_chars = "".join(c for _, c, _ in rs)
        ref_sound = _sound(rs)
        votes: Counter = Counter()
        for j, (s, _, _) in enumerate(rs):
            for ci, pos in index.get(s, ()):
                votes[(ci, pos - j)] += 1
        need = (n + 1) // 2
        best: dict[int, Suspect] = {}
        for (ci, start), v in votes.items():
            if v < need:
                continue
            seq = seqs[ci]
            key, text = cues[ci]
            if ref_chars in "".join(c for _, c, _ in seq):
                continue
            # 只比同樣字數的片段：差一個字的多半是另一句話（「我準備要」不是「我準備好了」），
            # 而誤聽的典型是字數相同、逐字換成同音字。
            if start < 0 or start + n > len(seq):
                continue
            win = seq[start : start + n]
            if "".join(c for _, c, _ in win) == ref_chars:
                continue
            score = SequenceMatcher(None, ref_sound, _sound(win)).ratio()
            if score < min_score:
                continue
            heard = text[win[0][2] : win[-1][2] + 1]
            if ci not in best or score > best[ci].score:
                best[ci] = Suspect(key, heard, ref, text, round(score, 3), "ref")
        out.extend(best.values())
    return out


def _interchangeable(a: str, b: str) -> bool:
    return all(
        x == y or any(x in g and y in g for g in INTERCHANGEABLE) for x, y in zip(a, b)
    )


def homophone_suspects(
    cues: list[tuple[tuple, str]],
    *,
    min_len: int = 3,
    max_len: int = 5,
    min_major: int = 10,
    ratio: float = 10.0,
    max_minor: int = 3,
) -> list[Suspect]:
    """全庫同音異寫：同一串音多數寫成 A、少數寫成 B，就把 B 標成嫌疑。

    只看 3–5 字的片段（兩個字的同音詞太多正常用法），並要求多數寫法夠常見、
    少數寫法夠罕見，才不會把「是不是／事不是」這類兩邊都常見的情況全倒給模型。
    """
    counts: Counter = Counter()
    where: dict[str, list[int]] = defaultdict(list)
    sound_of: dict[str, str] = {}
    for ci, (_, text) in enumerate(cues):
        seq = syllables(text)
        seen = set()
        for size in range(min_len, max_len + 1):
            for i in range(len(seq) - size + 1):
                win = seq[i : i + size]
                if win[-1][2] - win[0][2] != size - 1:  # 中間隔了標點就不算一個片段
                    continue
                chars = "".join(c for _, c, _ in win)
                counts[chars] += 1
                sound_of[chars] = _sound(win)
                if chars not in seen and len(where[chars]) < max_minor:
                    where[chars].append(ci)
                seen.add(chars)
    groups: dict[str, list[str]] = defaultdict(list)
    for chars, sound in sound_of.items():
        groups[sound].append(chars)

    out = []
    for forms in groups.values():
        if len(forms) < 2:
            continue
        forms.sort(key=lambda f: -counts[f])
        major = forms[0]
        if counts[major] < min_major:
            continue
        for minor in forms[1:]:
            if counts[minor] > max_minor or counts[major] < ratio * counts[minor]:
                continue
            if _interchangeable(major, minor):
                continue
            for ci in where[minor]:
                key, text = cues[ci]
                out.append(
                    Suspect(
                        key,
                        minor,
                        major,
                        text,
                        round(counts[major] / (counts[major] + counts[minor]), 3),
                        "homophone",
                    )
                )
    return _drop_nested(out)


def _drop_nested(suspects: list[Suspect]) -> list[Suspect]:
    """同一條字幕裡互相包含的嫌疑（「蟹堡蜜」「美味蟹堡蜜」）只留最長的。"""
    by_key: dict[tuple, list[Suspect]] = defaultdict(list)
    for s in suspects:
        by_key[s.key].append(s)
    out = []
    for group in by_key.values():
        group.sort(key=lambda s: -len(s.heard))
        kept: list[Suspect] = []
        for s in group:
            if not any(s.heard in k.heard for k in kept):
                kept.append(s)
        out.extend(kept)
    return out


CONFIRM_SYSTEM = """你是《海綿寶寶》台灣配音版的字幕校對。字幕由語音辨識產生，常把字音聽對、字選錯。

每一題給你：字幕原文（context）、可疑片段（heard）、候選正字（proposed）。
proposed 來自台灣觀眾熟知的台詞與譯名，或是同一個音在全部字幕裡最常見的寫法。
請判斷 heard 在這句台詞裡是不是 proposed 的誤聽：
- 是：ok=true，wrong 填 context 裡一字不差出現的最短錯誤片段，right 填改正後的寫法。
- 不是（heard 本來就是正確的另一個詞、或改了反而不通）：ok=false。
沒把握就填 false。"""

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "wrong": {"type": "string"},
                    "right": {"type": "string"},
                },
                "required": ["id", "ok"],
            },
        }
    },
    "required": ["verdicts"],
}


def confirm_prompt(batch: list[tuple[int, Suspect]]) -> str:
    return "\n".join(
        f"{i}|context={s.context.replace(chr(10), ' ')}|heard={s.heard}|proposed={s.proposed}"
        for i, s in batch
    )
