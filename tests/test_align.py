from psr.models import Word, Cue
from psr.align import align


def _tutorial_words():
    return [
        Word("我", 0.0, 0.2), Word("們", 0.2, 0.4), Word("先", 0.4, 0.6),
        Word("來", 0.6, 0.8), Word("安", 0.8, 1.0), Word("裝", 1.0, 1.2),
        Word("Python", 1.2, 1.8),
    ]


def test_align_identity_polish():
    words = _tutorial_words()
    lines = ["我們先來安裝Python"]
    result = align(words, lines)
    assert result == [Cue(index=1, start=0.0, end=1.8, text="我們先來安裝Python")]


def test_align_with_correction_still_maps_full_span():
    # ASR 把「安」聽成同音字「後」，潤稿修正回「安」。修正是行內部的單字替換，
    # 不影響行首/行尾這兩個邊界點的映射，所以整行時間應該和沒有修正時一樣。
    words = [
        Word("我", 0.0, 0.2), Word("們", 0.2, 0.4), Word("先", 0.4, 0.6),
        Word("來", 0.6, 0.8), Word("後", 0.8, 1.0), Word("裝", 1.0, 1.2),
        Word("Python", 1.2, 1.8),
    ]
    lines = ["我們先來安裝Python"]
    result = align(words, lines)
    assert result == [Cue(index=1, start=0.0, end=1.8, text="我們先來安裝Python")]


def test_align_unanchored_line_degrades_to_none():
    words = _tutorial_words()
    lines = ["今天天氣真好嗎"]  # 與原文完全無關，找不到任何長度 >=4 的 equal 區塊
    assert align(words, lines) is None


def test_align_empty_words_returns_none():
    assert align([], ["某些文字"]) is None


def test_align_empty_lines_returns_none():
    assert align(_tutorial_words(), []) is None


def _multichar_words():
    """Whisper 對中文輸出的是多字詞，不是單字。上面的 fixture 全是單字元 word，
    因此永遠踩不到「斷行點落在 word 內部」這條路徑——而那對中文是常態。"""
    return [
        Word("今天我們", 0.0, 2.0),
        Word("先來安裝", 2.0, 4.0),
        Word("Python環境", 4.0, 7.0),
        Word("這邊要注意", 7.0, 10.0),
        Word("版本必須是", 10.0, 13.0),
    ]


def test_align_line_break_inside_word_does_not_overlap():
    # 迴歸測試：斷行點切在「先來安|裝」與「這邊要注意|版本」的 word 內部。
    # 舊實作讓前一行取該 word 的 end、後一行取同一個 word 的 start，
    # 產生 cue1.end=4.0 > cue2.start=2.0 的時間軸重疊。
    words = _multichar_words()
    lines = ["今天我們先來安", "裝Python環境這邊要注意", "版本必須是"]

    cues = align(words, lines)
    assert cues is not None, "這是合法的重切，不該降級"
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end <= later.start, (
            f"時間軸重疊：{earlier.end} > {later.start}"
        )


def test_align_adjacent_cues_share_the_boundary_instant():
    # 每個斷點只有一個時間，所以相鄰 cue 必然首尾相接——這是建構上的保證。
    words = _multichar_words()
    lines = ["今天我們先來安", "裝Python環境這邊要注意", "版本必須是"]

    cues = align(words, lines)
    assert cues is not None
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end == later.start


def test_align_spans_full_audio_when_lines_cover_everything():
    words = _multichar_words()
    lines = ["今天我們先來安", "裝Python環境這邊要注意", "版本必須是"]

    cues = align(words, lines)
    assert cues is not None
    assert cues[0].start == words[0].start
    assert cues[-1].end == words[-1].end


def test_align_accepts_lines_shorter_than_the_anchor_length():
    # 迴歸測試：一份逐字完全相同的潤稿，只因為其中一行是「好的」這種兩字
    # 短句，就因為裝不下 4 字錨點而讓整個窗口降級。錨點要求必須隨跨度縮小。
    words = [Word("今天我們", 0.0, 2.0), Word("先來安裝", 2.0, 4.0),
             Word("好的", 4.0, 5.0), Word("這邊要注意", 5.0, 8.0)]
    lines = ["今天我們先來安裝", "好的", "這邊要注意"]

    cues = align(words, lines)
    assert cues is not None, "逐字相同的潤稿不該降級"
    assert [c.text for c in cues] == lines


def test_align_still_rejects_a_short_line_that_does_not_match():
    words = [Word("今天我們", 0.0, 2.0), Word("先來安裝", 2.0, 4.0),
             Word("好的", 4.0, 5.0), Word("這邊要注意", 5.0, 8.0)]
    # 中間換成完全無關的兩個字，仍必須降級——放寬錨點長度不等於放棄檢查
    assert align(words, ["今天我們先來安裝", "貓咪", "這邊要注意"]) is None
