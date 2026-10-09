from psr.models import Word
from psr.align import word_stream, line_stream


def test_word_stream_concatenates_and_maps_char_to_word():
    words = [Word(text="AB", start=0.0, end=1.0), Word(text="C", start=1.0, end=1.5)]
    text, char_to_word = word_stream(words)
    assert text == "ABC"
    assert char_to_word == [0, 0, 1]


def test_word_stream_empty():
    assert word_stream([]) == ("", [])


def test_line_stream_concatenates_and_tracks_boundaries():
    lines = ["AB", "CDE"]
    text, char_to_line, boundaries = line_stream(lines)
    assert text == "ABCDE"
    assert char_to_line == [0, 0, 1, 1, 1]
    assert boundaries == [0, 2]


def test_line_stream_empty():
    assert line_stream([]) == ("", [], [])
