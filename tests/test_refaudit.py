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


def test_confirm_prompt_includes_surrounding_cues_and_title():
    from psr.refaudit import Suspect, confirm_prompt
    s = Suspect(("ep1", 5), "幾張電源", "急徵店員", "我們在幾張電源", 0.9, "ref")
    out = confirm_prompt([(0, s)], lambda key: ("急徵店員", ["蟹堡王", "缺人"], ["我準備好了"]))
    assert "集名：急徵店員" in out
    assert "前文：蟹堡王／缺人" in out
    assert "本句：我們在幾張電源" in out
    assert "後文：我準備好了" in out
    assert "可疑：幾張電源　候選：急徵店員" in out


def test_audit_suspect_takes_quoted_fragment_found_in_cue():
    from psr.refaudit import audit_suspects
    rows = [(("e", 3), "我們去找謝老闆吧", "「謝老闆」應為蟹老闆", "蟹老闆")]
    [s] = audit_suspects(rows)
    assert (s.key, s.heard, s.proposed, s.context, s.reason) == (
        ("e", 3), "謝老闆", "蟹老闆", "我們去找謝老闆吧", "audit")


def test_audit_suspect_skips_quotes_not_in_cue_and_equal_to_suggestion():
    from psr.refaudit import audit_suspects
    rows = [(("e", 3), "叫拍大星過來", "「派大星」與本集「拍大星」不一致", "派大星")]
    assert audit_suspects(rows)[0].heard == "拍大星"


def test_audit_suspect_without_quote_flags_whole_cue():
    from psr.refaudit import audit_suspects
    rows = [(("e", 3), "天空吃了月亮的鞋", "語意不通，辨識失真", "")]
    [s] = audit_suspects(rows)
    assert (s.heard, s.proposed) == ("天空吃了月亮的鞋", "")


def test_audit_suspect_dropped_when_already_fixed():
    from psr.refaudit import audit_suspects
    rows = [(("e", 3), "我們去找蟹老闆吧", "「謝老闆」應為蟹老闆", "蟹老闆")]
    assert audit_suspects(rows) == []


def test_confirm_prompt_asks_model_to_infer_when_no_candidate():
    from psr.refaudit import Suspect, confirm_prompt
    s = Suspect(("ep1", 5), "天空吃了月亮的鞋", "", "天空吃了月亮的鞋", 0.0, "audit")
    out = confirm_prompt([(0, s)])
    assert "候選：（無，依前後文與讀音推斷）" in out


def test_insertion_fix_is_ambiguous_when_word_appears_elsewhere():
    from psr.refaudit import ambiguous_insertion
    # 只該補一處，但整集替換會把其他「等一下」都改掉
    assert ambiguous_insertion("等一下", "等一下下", "等一下，好嗎\n你等一下") is True


def test_insertion_fix_ok_when_other_hits_are_already_correct():
    from psr.refaudit import ambiguous_insertion
    assert ambiguous_insertion("綿寶寶", "海綿寶寶", "海綿寶寶，綿寶寶\n海綿寶寶") is False


def test_substitution_fix_is_never_ambiguous():
    from psr.refaudit import ambiguous_insertion
    assert ambiguous_insertion("派大西", "派大星", "派大西\n派大西") is False
