from psr.cleanup import apply_corrections, drop_hallucinations
from psr.models import Word


def _w(*texts):
    return [Word(t, float(i), float(i) + 0.5) for i, t in enumerate(texts)]


# S01E01 片尾實際出現的幻覺，跨越兩條字幕。
HALLUCINATIONS = ["優優獨播劇場", "YoYo Television Series Exclusive"]


def test_drops_words_covered_by_observed_credit_hallucination():
    words = _w("乾", "杯", "，", "優優", "獨播", "劇場", "Yo", "Yo", "Television", "Series", "Exclusive")
    kept, dropped = drop_hallucinations(words, HALLUCINATIONS)
    assert [w.text for w in kept] == ["乾", "杯", "，"]
    assert dropped == 8


def test_matches_simplified_and_spaced_variants():
    words = _w("优优", "独播", "剧场", " YoYo", " Television", " Series", " Exclusive")
    kept, _ = drop_hallucinations(words, HALLUCINATIONS)
    assert kept == []


def test_keeps_untouched_words_with_their_original_timing():
    words = _w("你", "聞到", "了", "沒有")
    kept, dropped = drop_hallucinations(words, HALLUCINATIONS)
    assert kept == words and dropped == 0


def test_corrects_misheard_name_without_moving_word_boundaries():
    words = _w("快來", "救我", "，", "張宇", "哥")
    fixed, n = apply_corrections(words, [("張宇哥", "章魚哥")])
    assert [w.text for w in fixed] == ["快來", "救我", "，", "章魚", "哥"]
    assert [(w.start, w.end) for w in fixed] == [(w.start, w.end) for w in words]
    assert n == 1


def test_corrects_simplified_form_and_every_occurrence():
    words = _w("张宇哥", "先生", "再见", "鄭", "宇哥")
    fixed, n = apply_corrections(words, [("張宇哥", "章魚哥"), ("鄭宇哥", "章魚哥")])
    assert [w.text for w in fixed] == ["章魚哥", "先生", "再见", "章", "魚哥"]
    assert n == 2


def test_correction_skips_across_punctuation_inside_original_text():
    # 正規化會丟掉標點，「張宇，哥」在比對時也是「張宇哥」——仍逐字替換，
    # 標點留在原位，字元數不變。
    words = _w("張宇", "，", "哥")
    fixed, _ = apply_corrections(words, [("張宇哥", "章魚哥")])
    assert [w.text for w in fixed] == ["章魚", "，", "哥"]


def test_unequal_length_pairs_are_ignored():
    words = _w("海綿", "寶")
    fixed, n = apply_corrections(words, [("海綿寶", "海綿寶寶")])
    assert fixed == words and n == 0


def test_collapse_repeats_drops_whisper_loops_but_keeps_short_laughs():
    from psr.cleanup import collapse_repeats
    words = [Word("哈哈哈", 0.0, 1.0), Word("來" * 200, 1.0, 9.0), Word("好", 9.0, 9.5)]
    out, dropped = collapse_repeats(words)
    assert [w.text for w in out] == ["哈哈哈", "來" * 6, "好"] and dropped == 194
    assert [(w.start, w.end) for w in out] == [(0.0, 1.0), (1.0, 9.0), (9.0, 9.5)]


def test_collapse_repeats_counts_runs_across_words_and_drops_emptied_words():
    from psr.cleanup import collapse_repeats
    words = [Word("來來來", 0.0, 1.0), Word("來來來", 1.0, 2.0), Word("來", 2.0, 2.5), Word("好", 2.5, 3.0)]
    out, dropped = collapse_repeats(words)
    assert [w.text for w in out] == ["來來來", "來來來", "好"] and dropped == 1
