from psr.refaudit import fuzzy, homophone_suspects, ref_suspects, syllables, widen


def test_fuzzy_merges_retroflex_and_nasal_finals():
    assert fuzzy("shi") == "si"
    assert fuzzy("zhong") == "zong"
    assert fuzzy("xing") == "xin"
    assert fuzzy("cheng") == "cen"
    assert fuzzy("xiong") == "xiong"


def test_syllables_skip_punctuation_but_keep_positions():
    out = syllables("老師，說")
    assert [c for _, c, _ in out] == ["老", "師", "說"]
    assert [p for _, _, p in out] == [0, 1, 3]
    assert out[1][0] == "si"


def test_ref_match_finds_misheard_name_across_punctuation():
    cues = [(("ep79", 1), "快告訴老師說，小熊，快！")]
    found = ref_suspects(cues, ["老實說熊熊"])
    assert len(found) == 1
    assert found[0].heard == "老師說，小熊"
    assert found[0].proposed == "老實說熊熊"
    assert found[0].key == ("ep79", 1)


def test_ref_match_ignores_cues_already_correct():
    assert ref_suspects([(("e", 1), "老實說熊熊！")], ["老實說熊熊"]) == []


def test_ref_match_ignores_unrelated_lines():
    assert ref_suspects([(("e", 1), "我準備好了！")], ["老實說熊熊"]) == []


def test_homophone_flags_rare_spelling_of_common_name():
    cues = [(("e", i), "派大星你好") for i in range(12)] + [(("e", 99), "拍大星你好")]
    found = homophone_suspects(cues)
    hits = {(s.heard, s.proposed, s.key) for s in found}
    # 互相包含的嫌疑只留最長的一段
    assert hits == {("拍大星你好", "派大星你好", ("e", 99))}


def test_homophone_ignores_interchangeable_pronouns():
    cues = [(("e", i), "他們走了") for i in range(12)] + [(("e", 99), "她們走了")]
    assert all("她們" not in s.heard for s in homophone_suspects(cues))


def test_widen_extends_single_char_fix_to_neighbours():
    assert widen("我要去謝寶王上班", "謝寶王", "寶", "堡") == ("謝寶王", "謝堡王")


def test_widen_anchors_on_the_suspect_not_the_first_occurrence():
    # 「寶」在句中出現三次，要改的是嫌疑片段「謝寶王」裡那個
    assert widen("海綿寶寶去謝寶王", "謝寶王", "寶", "堡") == ("謝寶王", "謝堡王")


def test_widen_keeps_long_fixes_unchanged():
    assert widen("快告訴老師說小熊快", "老師說小熊", "老師說小熊", "老實說熊熊") == ("老師說小熊", "老實說熊熊")


def test_widen_gives_up_when_punctuation_blocks():
    assert widen("寶！", "寶", "寶", "堡") is None


def test_ref_match_rejects_different_character_names_sharing_a_suffix():
    # 蟹老闆 / 皮老闆 只共用「老闆」，xie 與 pi 不像
    assert ref_suspects([(("e", 1), "蟹老闆你好")], ["皮老闆你好"]) == []


def test_ref_match_rejects_spelling_variants_of_the_same_word():
    assert ref_suspects([(("e", 1), "美味蟹堡祖傳秘方")], ["美味蟹堡祖傳祕方"]) == []


def test_ref_match_skips_phrases_that_are_common_in_the_corpus():
    cues = [(("e", i), "海綿寶寶你好") for i in range(9)]
    assert ref_suspects(cues, ["海綿寶貝你好"]) == []
