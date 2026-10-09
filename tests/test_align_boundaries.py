from psr.align import map_boundaries, has_anchor


def test_map_boundaries_identity():
    assert map_boundaries("abcdef", "abcdef", [0, 3, 6]) == [0, 3, 6]


def test_map_boundaries_snaps_across_inserted_text():
    # poly 在 orig 的 "abc" 之後插入了 "XXX"，boundary 落在插入區塊內時
    # 一律吸附到最近的 equal 邊界。
    orig = "abcdef"
    poly = "abcXXXdef"
    assert map_boundaries(orig, poly, [4]) == [3]


def test_map_boundaries_across_deleted_text():
    orig = "abcZZZdef"
    poly = "abcdef"
    assert map_boundaries(orig, poly, [3]) == [6]


def test_map_boundaries_tie_break_takes_preceding_edge():
    # orig="abcXYdef", poly="abcPQdef"：XY 被換成 PQ（同長度 replace）。
    # poly 上 b=4（P 和 Q 之間）與左邊 equal 區塊尾端（j=3）、右邊 equal 區塊
    # 開頭（j=5）等距，平手時固定取「前者」——回傳 i=3，不是 i=5。
    orig = "abcXYdef"
    poly = "abcPQdef"
    assert map_boundaries(orig, poly, [4]) == [3]


def test_map_boundaries_completely_unrelated_strings_are_all_none():
    assert map_boundaries("abc", "xyz", [0, 1, 2, 3]) == [None, None, None, None]


def test_has_anchor_overlap_exactly_min_len_is_true():
    opcodes = [("equal", 0, 4, 0, 4)]
    assert has_anchor(opcodes, 0, 4, min_len=4) is True


def test_has_anchor_partial_overlap_below_min_len_is_false():
    # 跨度夠長（10 字元），但唯一的 equal 區塊只涵蓋其中 3 個 —— 錨點不足。
    # 這才是這條規則要擋的情況：內容大體上已經對不上原文了。
    opcodes = [("equal", 0, 3, 0, 3), ("replace", 3, 10, 3, 10)]
    assert has_anchor(opcodes, 0, 10, min_len=4) is False


def test_has_anchor_requirement_shrinks_to_fit_a_short_span():
    # 跨度只有 3 個字元時，不可能容納 4 字元的錨點。完全吻合就該算通過，
    # 否則「好的」這種兩字短句永遠無法錨定，一份逐字相同的潤稿也會被判失敗。
    opcodes = [("equal", 0, 3, 0, 3)]
    assert has_anchor(opcodes, 0, 3, min_len=4) is True


def test_has_anchor_picks_best_among_multiple_equal_blocks():
    # 第一個 equal 區塊太短（長度 2），第二個夠長（長度 5），只要有一個
    # 符合門檻就算數。
    opcodes = [
        ("equal", 0, 2, 0, 2),
        ("replace", 2, 3, 2, 3),
        ("equal", 3, 8, 3, 8),
    ]
    assert has_anchor(opcodes, 0, 8, min_len=4) is True


def test_has_anchor_no_overlap_with_query_range_is_false():
    # equal 區塊存在，但完全落在查詢範圍之外
    opcodes = [("equal", 0, 10, 0, 10)]
    assert has_anchor(opcodes, 20, 30, min_len=4) is False
