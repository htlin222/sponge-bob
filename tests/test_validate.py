from psr.models import Cue
from psr.validate import (
    validate,
    _check_intervals_valid,
    _check_monotonic,
    _check_bounds,
    _check_line_width,
    _check_duration,
    _check_reading_speed,
    _check_contiguous_indices,
)


def test_intervals_valid_pass():
    assert _check_intervals_valid([Cue(1, 0.0, 1.0, "A")]) == []


def test_intervals_valid_fail():
    assert _check_intervals_valid([Cue(1, 1.0, 1.0, "A")]) != []


def test_monotonic_pass():
    cues = [Cue(1, 0.0, 1.0, "A"), Cue(2, 1.0, 2.0, "B")]
    assert _check_monotonic(cues) == []


def test_monotonic_fail():
    cues = [Cue(1, 0.0, 1.5, "A"), Cue(2, 1.0, 2.0, "B")]
    assert _check_monotonic(cues) != []


def test_bounds_pass():
    cues = [Cue(1, 0.0, 1.0, "A")]
    assert _check_bounds(cues, audio_duration=10.0) == []


def test_bounds_fail_negative_start():
    cues = [Cue(1, -0.5, 1.0, "A")]
    assert _check_bounds(cues, audio_duration=10.0) != []


def test_bounds_fail_end_exceeds_audio_duration():
    cues = [Cue(1, 0.0, 11.0, "A")]
    assert _check_bounds(cues, audio_duration=10.0) != []


def test_line_width_pass():
    cues = [Cue(1, 0.0, 1.0, "你好")]
    assert _check_line_width(cues) == []


def test_line_width_fail():
    cues = [Cue(1, 0.0, 1.0, "你" * 21)]
    assert _check_line_width(cues) != []


def test_duration_pass():
    cues = [Cue(1, 0.0, 1.0, "A")]
    assert _check_duration(cues) == []


def test_duration_fail_too_short():
    cues = [Cue(1, 0.0, 0.1, "A")]
    assert _check_duration(cues) != []


def test_duration_fail_too_long():
    # 上限從 7 秒放寬到 12 秒：講課語速慢，一個完整句子常需 8-10 秒，
    # 為了守 7 秒而把句子切在詞中間是本末倒置。
    cues = [Cue(1, 0.0, 15.0, "A")]
    assert _check_duration(cues) != []


def test_duration_allows_a_long_sentence():
    # 8 秒的字幕在放寬前會被判違規，導致整窗降級。實測講課語速下，
    # 一個完整句子常落在 8-10 秒。
    assert _check_duration([Cue(1, 0.0, 8.0, "一個完整的句子")]) == []


def test_reading_speed_pass():
    cues = [Cue(1, 0.0, 2.0, "你好")]  # 2 全形字 / 2 秒 = 1 <= 9
    assert _check_reading_speed(cues) == []


def test_reading_speed_fail():
    cues = [Cue(1, 0.0, 1.0, "你" * 10)]  # 10 全形字 / 1 秒 = 10 > 9
    assert _check_reading_speed(cues) != []


def test_contiguous_indices_pass():
    cues = [Cue(1, 0.0, 1.0, "A"), Cue(2, 1.0, 2.0, "B")]
    assert _check_contiguous_indices(cues) == []


def test_contiguous_indices_fail():
    cues = [Cue(1, 0.0, 1.0, "A"), Cue(3, 1.0, 2.0, "B")]
    assert _check_contiguous_indices(cues) != []


def test_validate_passing_cues_returns_empty_list():
    cues = [Cue(1, 0.0, 1.0, "你好"), Cue(2, 1.0, 2.0, "世界")]
    assert validate(cues, audio_duration=2.0) == []


def test_validate_collects_multiple_violations():
    cues = [Cue(1, 0.0, 0.1, "你" * 25)]
    violations = validate(cues, audio_duration=10.0)
    assert len(violations) >= 2  # 時長太短 + 行寬超標


def test_validate_rejects_empty_text():
    # 空白字幕能通過其他所有規則：時長合法、不重疊、行寬 0、閱讀速度 0。
    cues = [Cue(index=1, start=0.0, end=2.0, text="")]
    violations = validate(cues, audio_duration=3.0)
    assert any("為空" in v for v in violations)
