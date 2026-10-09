from psr.fix import apply_fixes, sound_similarity
from psr.models import Cue

CUES = [Cue(1, 0.0, 1.5, "謝老闆，老闆，快點"), Cue(2, 1.5, 3.0, "我們可以變成好媽雞"),
        Cue(3, 3.0, 4.0, "我做不得白大姨，請問你要去哪裡？")]


def _p(cue, wrong, right):
    return {"cue": cue, "wrong": wrong, "right": right}


def test_homophone_fix_changes_text_but_not_timing():
    fixed, applied, skipped = apply_fixes(CUES, [_p(1, "謝老闆", "蟹老闆"), _p(2, "好媽雞", "好麻吉")])
    assert [c.text for c in fixed] == ["蟹老闆，老闆，快點", "我們可以變成好麻吉", CUES[2].text]
    assert [(c.index, c.start, c.end) for c in fixed] == [(c.index, c.start, c.end) for c in CUES]
    assert len(applied) == 2 and skipped == []


def test_rewrite_that_does_not_sound_alike_is_blocked():
    # 「我做不得白大姨」→「派大星」是憑劇情猜，不是聽錯。
    fixed, applied, skipped = apply_fixes(CUES, [_p(3, "我做不得白大姨", "派大星")])
    assert fixed[2].text == CUES[2].text and applied == [] and len(skipped) == 1


def test_wrong_text_must_appear_in_that_cue():
    _, applied, skipped = apply_fixes(CUES, [_p(2, "謝老闆", "蟹老闆")])
    assert applied == [] and len(skipped) == 1


def test_unknown_cue_empty_or_identical_fixes_are_blocked():
    proposals = [_p(9, "謝老闆", "蟹老闆"), _p(1, "", "蟹"), _p(1, "謝老闆", "謝老闆")]
    _, applied, skipped = apply_fixes(CUES, proposals)
    assert applied == [] and len(skipped) == 3


def test_length_change_over_two_is_blocked():
    _, applied, _ = apply_fixes(CUES, [_p(1, "謝老闆", "蟹老闆蟹老闆")])
    assert applied == []


def test_sound_similarity_is_one_for_pure_homophones():
    assert sound_similarity("素食", "速食") == 1.0


def test_traditionalize_converts_leftover_simplified_without_touching_timing():
    from psr.fix import traditionalize
    cues = [Cue(7, 1.0, 2.0, "醒醒孩子你在烧我的钱"), Cue(8, 2.0, 3.0, "蟹老闆")]
    out = traditionalize(cues)
    assert [c.text for c in out] == ["醒醒孩子你在燒我的錢", "蟹老闆"]
    assert [(c.index, c.start, c.end) for c in out] == [(7, 1.0, 2.0), (8, 2.0, 3.0)]
