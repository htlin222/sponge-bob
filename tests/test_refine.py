import pytest

from psr.models import Word, Cue
from psr.refine import enforce_duration, merge_short_cues


def test_enforce_duration_splits_at_largest_gap():
    # 用中文而非 ASCII 單字母：ASCII 半形寬 0.5，"AB" 只有 1.0 寬，會被
    # _split_one 的碎片守衛擋下。那個守衛是對的——一條只有 "AB" 的字幕
    # 在螢幕上讀不到——但它讓這個測試原本的合成資料失去代表性。
    words = [Word("甲", 0, 1), Word("乙", 1, 2), Word("丙", 5, 6), Word("丁", 6, 7)]
    cue = Cue(index=1, start=0, end=7, text="甲乙丙丁")
    result = enforce_duration([cue], words, min_s=0.5, max_s=5.0)
    assert result == [
        Cue(index=1, start=0, end=2, text="甲乙"),
        Cue(index=2, start=5, end=7, text="丙丁"),
    ]


def test_enforce_duration_borrows_when_gap_alone_is_not_enough():
    words = [Word("A", 0, 0.2), Word("B", 0.3, 1.0)]
    cues = [Cue(index=1, start=0, end=0.2, text="A"), Cue(index=2, start=0.3, end=1.0, text="B")]
    result = enforce_duration(cues, words, min_s=0.5, max_s=7.0)

    # 空檔（0.2→0.3）只夠補 0.1 秒，剩下的 0.2 秒向下一條借。
    # 兩條都達到 min_s，且仍不重疊。
    # 用容差比較：借用量是浮點減法算出來的（0.7 - 0.5 = 0.19999999999999996），
    # 精確相等在這裡註定失敗，而這個誤差量級對字幕毫無意義。
    assert result[0].end == pytest.approx(0.5)
    assert result[0].end <= result[1].start
    assert result[1].end - result[1].start >= 0.5 - 1e-9

    # 代價要講清楚：B 實際在 0.3 秒開口，字幕卻從 0.5 秒才出現，晚了 0.2 秒。
    # 這個取捨是刻意的——留一條 0.3 秒的字幕只會閃一下、根本讀不到，
    # 字幕規範上最短時長優先於精確同步。可借的上限就是「捐出者仍保有 min_s」，
    # 所以延遲量有界，不會累積。
    assert result[1].start - words[1].start == pytest.approx(0.2)


def test_enforce_duration_never_creates_overlap():
    words = [Word("A", 0, 0.1), Word("B", 0.15, 0.3), Word("C", 0.35, 0.5)]
    cues = [
        Cue(index=1, start=0, end=0.1, text="A"),
        Cue(index=2, start=0.15, end=0.3, text="B"),
        Cue(index=3, start=0.35, end=0.5, text="C"),
    ]
    result = enforce_duration(cues, words, min_s=0.5, max_s=7.0)
    for a, b in zip(result, result[1:]):
        assert a.end <= b.start


def test_extend_short_cue_borrows_from_neighbour_when_contiguous():
    # align() 的輸出首尾相接、空檔為零。舊實作只能往空檔延伸，
    # 因此中間這條 0.2 秒的字幕會原封不動通過，接著被 validate 判為違規。
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="今天我們先來安裝"),
        Cue(index=2, start=1.0, end=1.2, text="好"),
        Cue(index=3, start=1.2, end=3.0, text="這邊要注意版本必須是"),
    ]
    words = [
        Word("今天我們先來安裝", 0.0, 1.0),
        Word("好", 1.0, 1.2),
        Word("這邊要注意版本必須是", 1.2, 3.0),
    ]

    out = enforce_duration(cues, words)
    assert out[1].end - out[1].start >= 0.5 - 1e-9, "過短的字幕沒有被延長"


def test_borrowing_never_creates_overlap_or_gap():
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="今天我們先來安裝"),
        Cue(index=2, start=1.0, end=1.2, text="好"),
        Cue(index=3, start=1.2, end=3.0, text="這邊要注意版本必須是"),
    ]
    words = [
        Word("今天我們先來安裝", 0.0, 1.0),
        Word("好", 1.0, 1.2),
        Word("這邊要注意版本必須是", 1.2, 3.0),
    ]

    out = enforce_duration(cues, words)
    for earlier, later in zip(out, out[1:]):
        assert abs(earlier.end - later.start) < 1e-9, "借用時間破壞了首尾相接"
    assert out[0].start == 0.0 and out[-1].end == 3.0, "整段的頭尾不該被改動"


def test_borrowing_leaves_donor_at_or_above_min_duration():
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="今天我們先來安裝"),
        Cue(index=2, start=1.0, end=1.2, text="好"),
        Cue(index=3, start=1.2, end=3.0, text="這邊要注意版本必須是"),
    ]
    words = [
        Word("今天我們先來安裝", 0.0, 1.0),
        Word("好", 1.0, 1.2),
        Word("這邊要注意版本必須是", 1.2, 3.0),
    ]

    out = enforce_duration(cues, words)
    for cue in out:
        assert cue.end - cue.start >= 0.5 - 1e-9, f"cue{cue.index} 被借到低於 min_s"


def test_split_never_produces_empty_text_when_polish_shortened_the_line():
    # 對抗性案例：ASR 逐字稿 20 字，潤稿後只剩 3 字。
    # 舊實作用 ASR 字元數（10）去索引潤稿文字（3 字），
    # cue.text[:10] 回傳整串、cue.text[10:] 回傳 ''，
    # 產出一條佔著數秒螢幕時間卻沒有字的空白字幕。
    words = [Word("字", float(i), float(i) + 0.5) for i in range(20)]
    cue = Cue(index=1, start=0.0, end=20.0, text="精簡版")

    out = enforce_duration([cue], words, min_s=0.5, max_s=7.0)
    assert all(c.text.strip() for c in out), f"產生了空白字幕：{out}"


def test_split_preserves_outer_boundaries():
    words = [Word("字", float(i), float(i) + 0.5) for i in range(20)]
    cue = Cue(index=1, start=0.0, end=20.0, text="一" * 20)

    out = enforce_duration([cue], words, min_s=0.5, max_s=7.0)
    assert out[0].start == 0.0
    assert out[-1].end == 20.0


def test_last_cue_is_not_extended_past_audio_end():
    # 最後一條過短時會被延長，但不能延過音訊結尾，否則 validate 的邊界檢查
    # 會判違規。實測影片就踩到這個情況。
    words = [Word("字", 0.0, 20.48), Word("出", 20.48, 20.78)]
    cues = [Cue(index=1, start=0.0, end=20.48, text="字"),
            Cue(index=2, start=20.48, end=20.78, text="出")]

    out = enforce_duration(cues, words, min_s=0.5, max_s=7.0, audio_duration=20.78)
    assert out[-1].end <= 20.78, f"延長超過音訊結尾：{out[-1].end}"

    # 未提供 audio_duration 時維持原本行為（延到 min_s）
    out2 = enforce_duration(cues, words, min_s=0.5, max_s=7.0)
    assert out2[-1].end == pytest.approx(20.98)


def test_strip_leading_punctuation_drops_cues_that_become_empty():
    # 整條只有標點的字幕，剝完就變空的。空白字幕佔著螢幕時間卻什麼都不顯示，
    # 而且時長合法、不重疊、行寬 0、閱讀速度 0——其他規則全都抓不到它。
    # 實測第四次 CI 執行產出 6 條這樣的字幕。
    from psr.refine import strip_leading_punctuation
    cues = [Cue(1, 0.0, 1.0, "正常字幕"), Cue(2, 1.0, 1.2, "，"), Cue(3, 1.2, 2.0, "也正常")]
    out = strip_leading_punctuation(cues)
    assert [c.text for c in out] == ["正常字幕", "也正常"]
    assert [c.index for c in out] == [1, 2], "刪掉之後序號必須重新連續"


def test_split_refuses_when_a_half_would_be_whitespace_only():
    # 'I ' 長度是 2，會通過「長度不足 2」的檢查，然後被切成 'I' 與 ' '。
    # 後者就是一條佔著螢幕時間卻沒有字的空白字幕。
    words = [Word("I", 0.0, 4.0), Word(" ", 4.0, 12.0)]
    cue = Cue(index=1, start=0.0, end=12.0, text="I ")
    out = enforce_duration([cue], words, min_s=0.5, max_s=7.0)
    assert all(c.text.strip() for c in out), f"產生了空白字幕：{[c.text for c in out]}"


def test_fragments_merge_even_across_a_long_pause():
    # 潤稿模型在少數窗口會退回逐字元切分。一條只有 'I' 的字幕永遠是錯的，
    # 讓它因為跨越靜音而留在原地，只是把模型的失誤原樣呈現給觀眾。
    cues = [Cue(1, 0.0, 1.0, "前面這句話夠長了"),
            Cue(2, 5.0, 5.2, "I"),          # 與前後都隔著長靜音
            Cue(3, 9.0, 10.0, "模型")]
    out = merge_short_cues(cues)
    assert all(len(c.text.strip()) >= 2 for c in out), \
        f"仍有單字元碎片：{[c.text for c in out]}"


def test_absorb_fragments_leaves_none_behind():
    from psr.refine import absorb_fragments
    cues = [Cue(1, 0.0, 1.0, "這是一句完整的話。"),
            Cue(2, 5.0, 5.2, "I"),        # 跟在句號後面，merge 不會處理
            Cue(3, 9.0, 9.1, " "),
            Cue(4, 9.1, 10.0, "模型")]
    out = absorb_fragments(cues)
    from psr.text import display_width
    assert all(display_width(c.text.strip()) >= 2 for c in out), \
        f"仍有碎片：{[c.text for c in out]}"
    assert [c.index for c in out] == list(range(1, len(out) + 1))


def test_absorb_fragments_preserves_the_overall_span():
    from psr.refine import absorb_fragments
    cues = [Cue(1, 0.0, 1.0, "完整的一句話。"), Cue(2, 5.0, 5.2, "I"), Cue(3, 9.0, 10.0, "模型")]
    out = absorb_fragments(cues)
    assert out[0].start == 0.0 and out[-1].end == 10.0


def test_split_refuses_when_a_half_would_be_an_unreadable_fragment():
    # 「所以當」切成「所以」與「當」，後者寬度 1，在螢幕上閃一下就沒了。
    # 這道檢查必須在切分處——absorb_fragments 跑在 enforce_duration 之前，
    # 看不到切分產生的新碎片。
    words = [Word("所以", 0.0, 6.0), Word("當", 6.0, 14.0)]
    cue = Cue(index=1, start=0.0, end=14.0, text="所以當")
    out = enforce_duration([cue], words, min_s=0.5, max_s=7.0)
    from psr.text import display_width
    assert all(display_width(c.text.strip()) >= 2 for c in out), \
        f"切出了碎片：{[c.text for c in out]}"
