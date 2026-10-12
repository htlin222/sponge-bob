from psr.still import caption_lines, frame_times, match_cues, segment_title, source_line


def test_frames_are_evenly_spaced_around_the_cue_middle():
    assert frame_times(10.0, 12.0) == [9.75, 10.25, 10.75, 11.25, 11.75, 12.25]


def test_frames_shrink_spacing_to_stay_within_window():
    # 間距 2 秒排 6 張要 10 秒，超過 [7, 13]，改成均分 6 秒
    assert frame_times(10.0, 10.0, step=2.0, window=3.0) == [
        7.0,
        8.2,
        9.4,
        10.6,
        11.8,
        13.0,
    ]


def test_frames_never_before_zero():
    assert min(frame_times(0.2, 0.6)) >= 0.0


def test_segment_title_picks_the_half_the_cue_is_in():
    assert segment_title("月球遊記／蟹老闆渡假去", 396.8, 1320.0) == "月球遊記"
    assert segment_title("月球遊記／蟹老闆渡假去", 900.0, 1320.0) == "蟹老闆渡假去"
    assert segment_title("失蹤記", 900.0, 1320.0) == "失蹤記"


def test_caption_lines_quote_and_indent_continuations():
    assert caption_lines("拜託海綿寶寶，你沒問題的，\n就像開玉米罐一樣。") == [
        "「拜託海綿寶寶，你沒問題的，",
        "　就像開玉米罐一樣。」",
    ]
    assert caption_lines("我準備好了！") == ["「我準備好了！」"]


def test_source_line_with_and_without_speaker():
    assert (
        source_line("珊迪", 8, 161, "月球遊記", 396.82)
        == "—— 珊迪｜《海綿寶寶》S08 ep161〈月球遊記〉6:36"
    )
    assert (
        source_line("", 8, 161, "月球遊記", 396.82)
        == "—— 《海綿寶寶》S08 ep161〈月球遊記〉6:36"
    )


CUES = [
    (1, 393.5, 394.78, "天哪，不好吧！"),
    (2, 396.82, 400.52, "拜託海綿寶寶，你沒問題的，\n就像開玉米罐一樣。"),
    (3, 400.52, 403.22, "我不知道玉米是什麼，但是我愛罐頭。"),
]


def test_match_ignores_punctuation_and_line_breaks():
    assert [c[0] for c in match_cues(CUES, "你沒問題的就像開玉米罐")] == [2]


def test_match_can_span_cues():
    assert [c[0] for c in match_cues(CUES, "一樣。我不知道")] == [2, 3]


def test_match_returns_empty_when_absent():
    assert match_cues(CUES, "蟹堡王") == []
