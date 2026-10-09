from psr.models import Cue, Word
from psr.text import display_width
from psr.timeline import build_cues, coverage


def _words(pairs):
    return [Word(t, s, e) for t, s, e in pairs]


def test_times_come_straight_from_the_words_no_guessing():
    words = _words([("今天我們", 0.0, 2.0), ("先來安裝", 2.0, 4.0), ("Python", 4.0, 5.0)])
    cues = build_cues(words, "今天我們先來安裝Python。")
    assert len(cues) == 1
    assert cues[0].start == 0.0 and cues[0].end == 5.0


def test_breaks_at_sentence_end():
    words = _words([("今天很好", 0.0, 2.0), ("明天更好", 2.0, 4.0)])
    cues = build_cues(words, "今天很好。明天更好。")
    assert [c.text for c in cues] == ["今天很好。", "明天更好。"]
    assert cues[0].end == 2.0 and cues[1].start == 2.0


def test_soft_break_waits_until_the_clause_is_long_enough():
    # 太早在逗號斷會切出「那我想，」這種讀不完整的半句
    words = _words([("那我想", 0.0, 1.0), ("這個部分真的非常重要喔", 1.0, 4.0)])
    cues = build_cues(words, "那我想，這個部分真的非常重要喔。", soft_min=14.0)
    assert cues[0].text.startswith("那我想，這個")


def test_breaks_across_a_long_silence():
    # 只看標點會產生橫跨整段停頓的字幕——實測出現過一條 29.9 秒的
    words = _words([("前面這句", 0.0, 2.0), ("後面這句", 30.0, 32.0)])
    cues = build_cues(words, "前面這句後面這句", max_gap=1.5)
    assert len(cues) == 2
    assert cues[0].end == 2.0 and cues[1].start == 30.0


def test_no_cue_ever_spans_more_than_the_gap_allows():
    words = _words([("甲", 0.0, 1.0), ("乙", 20.0, 21.0), ("丙", 40.0, 41.0)])
    cues = build_cues(words, "甲乙丙", max_gap=1.5)
    assert len(cues) == 3


def test_leading_orphan_punctuation_is_dropped():
    words = _words([("你好世界", 0.0, 2.0)])
    cues = build_cues(words, "，你好世界。")
    assert cues[0].text == "你好世界。"


def test_every_original_character_is_accounted_for():
    words = _words([("今天我們", 0.0, 2.0), ("先來安裝", 2.0, 4.0), ("環境設定", 4.0, 6.0)])
    cues = build_cues(words, "今天我們，先來安裝。環境設定。")
    assert coverage(words, cues) == 1.0


def test_timeline_is_monotonic_and_non_overlapping():
    words = _words([(f"字{i}", float(i), float(i) + 0.9) for i in range(20)])
    cues = build_cues(words, "".join(f"字{i}" for i in range(20)))
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start
        assert a.start < a.end
