from psr.bob import _accept
from psr.refaudit import Suspect


def _s(reason, context="什麼世界都一樣"):
    return Suspect(("e", 1), "什麼世", "什麼事", context, 0.9, reason)


def test_homophone_fix_must_keep_length():
    assert _accept(_s("homophone"), {"ok": True, "wrong": "什麼世界", "right": "什麼事"})[0] is False


def test_fix_may_not_add_punctuation():
    s = Suspect(("e", 1), "溺水了你們", "溺水啦你們", "溺水了你們最會溺水", 0.9, "ref")
    assert _accept(s, {"ok": True, "wrong": "溺水了你們", "right": "溺水啦，你們"})[0] is False


def test_same_length_homophone_fix_passes():
    s = Suspect(("e", 1), "北汶蟹堡", "美味蟹堡", "北汶蟹堡好好吃", 0.9, "homophone")
    assert _accept(s, {"ok": True, "wrong": "北汶蟹堡", "right": "美味蟹堡"}) == (True, "北汶蟹堡", "美味蟹堡")
