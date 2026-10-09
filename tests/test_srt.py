import pytest

from psr.models import Cue
from psr.srt import format_timestamp, parse, render


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_with_hours_minutes_seconds():
    assert format_timestamp(3661.5) == "01:01:01,500"


def test_format_timestamp_millisecond_rounding():
    # 1.0005 秒在 IEEE 754 double 中實際儲存為略小於 1.0005 的值，
    # round(seconds * 1000) 對剛好落在 .5 的浮點數採「banker's rounding」
    # （四捨五入到最近的偶數），1000 是偶數，所以捨去而非進位。
    # 這不是 bug，是刻意選擇 round() 而非手動 +0.5 取整的結果——
    # 手動 +0.5 在浮點數上反而更容易因為誤差被放大而不穩定，
    # round() 對整數毫秒運算則是穩定、可重現的。
    assert format_timestamp(1.0005) == "00:00:01,000"


def test_format_timestamp_negative_raises():
    with pytest.raises(ValueError):
        format_timestamp(-1.0)


def test_render_two_cues_exact_string():
    cue1 = Cue(index=1, start=0.0, end=1.5, text="你好")
    cue2 = Cue(index=2, start=1.5, end=3.0, text="世界")
    expected = (
        "1\n00:00:00,000 --> 00:00:01,500\n你好\n"
        "\n"
        "2\n00:00:01,500 --> 00:00:03,000\n世界\n"
    )
    assert render([cue1, cue2]) == expected


def test_render_empty_list():
    assert render([]) == ""


def test_parse_two_cues():
    text = (
        "1\n00:00:00,000 --> 00:00:01,500\n你好\n"
        "\n"
        "2\n00:00:01,500 --> 00:00:03,000\n世界\n"
    )
    cues = parse(text)
    assert cues == [
        Cue(index=1, start=0.0, end=1.5, text="你好"),
        Cue(index=2, start=1.5, end=3.0, text="世界"),
    ]
