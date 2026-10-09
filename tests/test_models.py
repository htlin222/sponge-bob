import pytest
from dataclasses import FrozenInstanceError

from psr.models import Word, Cue


def test_word_construction():
    w = Word(text="你好", start=0.0, end=0.5)
    assert w.text == "你好"
    assert w.start == 0.0
    assert w.end == 0.5


def test_word_is_frozen():
    w = Word(text="你好", start=0.0, end=0.5)
    with pytest.raises(FrozenInstanceError):
        w.text = "再見"


def test_cue_construction():
    c = Cue(index=1, start=0.0, end=1.5, text="你好世界")
    assert c.index == 1
    assert c.text == "你好世界"


def test_cue_is_frozen():
    c = Cue(index=1, start=0.0, end=1.5, text="你好世界")
    with pytest.raises(FrozenInstanceError):
        c.index = 2
