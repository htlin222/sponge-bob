from psr.bob import _accept
from psr.refaudit import Suspect


def _s(reason, context="什麼世界都一樣"):
    return Suspect(("e", 1), "什麼世", "什麼事", context, 0.9, reason)


def test_fix_may_change_length_by_at_most_two():
    s = Suspect(("e", 1), "海的大峰裡", "海的大鳳梨", "深海的大峰裡", 0.9, "homophone")
    assert _accept(s, {"ok": True, "wrong": "海的大峰裡", "right": "海的大鳳梨裡"})[0] is True
    assert _accept(s, {"ok": True, "wrong": "深海的大峰裡", "right": "海"})[0] is False


def test_model_spaces_map_back_to_newlines():
    s = Suspect(("e", 1), "素食餐廳", "速食餐廳", "最美味的\n素食餐廳", 0.9, "ref")
    assert _accept(s, {"ok": True, "wrong": "最美味的 素食餐廳", "right": "最美味的 速食餐廳"}) == (
        True, "最美味的\n素食餐廳", "最美味的\n速食餐廳")


def test_fix_may_not_add_punctuation():
    s = Suspect(("e", 1), "溺水了你們", "溺水啦你們", "溺水了你們最會溺水", 0.9, "ref")
    assert _accept(s, {"ok": True, "wrong": "溺水了你們", "right": "溺水啦，你們"})[0] is False


def test_same_length_homophone_fix_passes():
    s = Suspect(("e", 1), "北汶蟹堡", "美味蟹堡", "北汶蟹堡好好吃", 0.9, "homophone")
    assert _accept(s, {"ok": True, "wrong": "北汶蟹堡", "right": "美味蟹堡"}) == (True, "北汶蟹堡", "美味蟹堡")
