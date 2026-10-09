import json
from pathlib import Path

from psr.models import Word
from psr.align import align
from psr.refine import enforce_duration
from psr.srt import render

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# 這是「潤稿後」的字幕行。Phase 1 沒有真的呼叫 LLM，這份清單是手動模擬
# LLM 潤稿會輸出的內容：加上標點、把 ASR 誤聽的「旨令」修正回「指令」
# （words.json 裡第 25 個字其實是「旨」，這裡故意寫成正確的「指」，
# 用來驗證 align() 真的能吸收潤稿修正，而不是要求逐字相同）。
POLISHED_LINES = [
    "大家好，今天我們來安裝 Python",
    "首先打開終端機。",
    "然後輸入安裝指令。",
    "接著我們要把程式碼推上 GitHub。",
    "這樣團隊成員就能透過 API 互相溝通。",
]


def _load_words() -> list[Word]:
    raw = json.loads((FIXTURES_DIR / "words.json").read_text(encoding="utf-8"))
    return [Word(text=w["text"], start=w["start"], end=w["end"]) for w in raw]


def test_golden_fixture_byte_for_byte():
    words = _load_words()
    aligned = align(words, POLISHED_LINES)
    assert aligned is not None

    final = enforce_duration(aligned, words)
    actual_srt = render(final)

    expected_srt = (FIXTURES_DIR / "expected.srt").read_text(encoding="utf-8")
    assert actual_srt == expected_srt
