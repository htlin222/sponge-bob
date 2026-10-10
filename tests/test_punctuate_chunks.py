from psr.models import Word
from psr.punctuate import make_chunks, similarity


def test_pause_inserts_newline_hint_that_does_not_count_as_an_edit():
    words = [Word("你好", 0.0, 0.5), Word("嗎", 0.5, 0.7), Word("我很好", 1.5, 2.0)]
    chunks = make_chunks(words, chunk_chars=100, pause_s=0.5)
    assert chunks == ["你好嗎\n我很好"]
    assert similarity("你好嗎我很好", chunks[0]) == 1.0


def test_chunks_prefer_to_end_at_a_pause():
    words = [Word("一二三", 0.0, 1.0), Word("四五", 1.0, 2.0), Word("六七", 3.0, 4.0)]
    # 累積到 5 字已達上限，但「四五」與「六七」之間才有停頓，所以在那裡切。
    assert make_chunks(words, chunk_chars=4, pause_s=0.5) == ["一二三四五", "六七"]


def test_fullwidth_only_touches_punctuation_after_chinese():
    from psr.punctuate import fullwidth
    assert fullwidth("你好?我很好!OK, fine") == "你好？我很好！OK, fine"


def test_split_at_pause_picks_the_newline_closest_to_the_middle():
    from psr.punctuate import split_at_pause
    chunk = "甲" * 150 + "\n" + "乙" * 40 + "\n" + "丙" * 160
    left, right = split_at_pause(chunk)
    assert left == "甲" * 150 + "\n" + "乙" * 40 and right == "\n" + "丙" * 160


def test_split_at_pause_refuses_short_or_unbroken_text():
    from psr.punctuate import split_at_pause
    assert split_at_pause("甲\n乙") is None and split_at_pause("甲" * 500) is None
