from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    """ASR 輸出的單一詞，帶絕對時間（秒）。"""
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Cue:
    """一條字幕。index 從 1 起算。"""
    index: int
    start: float
    end: float
    text: str
