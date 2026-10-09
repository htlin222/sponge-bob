import pytest

from psr import quotes
from psr.models import Cue
from psr.quotes import Taxonomy, TaxonomyError, accept, load_taxonomy, parse_episode_name

TAX = Taxonomy({
    "situations": ("上班", "上學考試"),
    "emotions": ("開心", "崩潰"),
    "functions": ("鼓勵", "吐槽"),
    "characters": ("海綿寶寶", "章魚哥"),
})
CUES = [Cue(i, float(i * 2), float(i * 2 + 1.5), f"第{i}句") for i in range(1, 13)]


def _q(a, b, score=4, **tags):
    return {"from": a, "to": b, "speaker": "海綿寶寶", "quotability": score,
            "use_when": "需要打氣時", "situations": [], "emotions": [], "functions": [],
            "characters": [], **tags}


def test_parses_season_episode_and_title_from_drive_name():
    meta = parse_episode_name("海綿寶寶_S07_ep142_洞窟歷險記（深海鄉巴佬）_海綿火山爆發（火山爆發記）.zh-Hant.srt")
    assert (meta.season, meta.episode) == (7, 142)
    assert meta.title == "洞窟歷險記（深海鄉巴佬）／海綿火山爆發（火山爆發記）"
    assert meta.stem == "海綿寶寶_S07_ep142_洞窟歷險記（深海鄉巴佬）_海綿火山爆發（火山爆發記）"


def test_quote_text_comes_from_subtitles_not_from_the_model():
    raw = {"quotes": [{**_q(3, 4), "text": "模型自己編的台詞"}], "issues": [], "new_tags": []}
    (q,) = accept(raw, CUES, TAX).quotes
    assert q.text == "第3句\n第4句"
    assert (q.start, q.end) == (6.0, 9.5)


def test_low_scores_are_skipped_but_not_counted_as_rejected():
    raw = {"quotes": [_q(1, 1, score=2), _q(2, 2, score=3)], "issues": [], "new_tags": []}
    out = accept(raw, CUES, TAX)
    assert [q.cue_from for q in out.quotes] == [2]
    assert out.rejected == 0


@pytest.mark.parametrize("bad", [
    _q(0, 1),                       # 編號不存在
    _q(12, 13),                     # 超出最後一條
    _q(5, 4),                       # from > to
    _q(1, 9),                       # 跨 9 條，超過上限 8
    _q(1, 1, emotions=["焦慮"]),     # 分類表外的標籤
])
def test_invalid_segments_are_rejected_and_counted(bad):
    out = accept({"quotes": [bad], "issues": [], "new_tags": []}, CUES, TAX)
    assert out.quotes == () and out.rejected == 1


def test_span_of_exactly_eight_cues_is_allowed():
    out = accept({"quotes": [_q(1, 8)], "issues": [], "new_tags": []}, CUES, TAX)
    assert len(out.quotes) == 1


def test_tags_keep_facet_and_drop_duplicates():
    raw = {"quotes": [_q(1, 1, situations=["上班", "上班"], characters=["章魚哥"])],
           "issues": [], "new_tags": []}
    (q,) = accept(raw, CUES, TAX).quotes
    assert q.tags == (("situations", "上班"), ("characters", "章魚哥"))


def test_duplicate_segments_are_stored_once():
    out = accept({"quotes": [_q(1, 2), _q(1, 2)], "issues": [], "new_tags": []}, CUES, TAX)
    assert len(out.quotes) == 1


def test_issues_pointing_at_missing_cues_are_dropped():
    raw = {"quotes": [], "new_tags": [], "issues": [
        {"cue": 3, "type": "人名", "detail": "張宇哥", "suggestion": "章魚哥"},
        {"cue": 99, "type": "錯字", "detail": "x", "suggestion": ""},
    ]}
    assert [i.cue for i in accept(raw, CUES, TAX).issues] == [3]


def test_new_tag_proposals_already_in_taxonomy_are_ignored():
    raw = {"quotes": [], "issues": [], "new_tags": [
        {"facet": "emotions", "name": "開心", "reason": ""},
        {"facet": "emotions", "name": "焦慮", "reason": "考前常用"},
    ]}
    assert accept(raw, CUES, TAX).new_tags == (("emotions", "焦慮", "考前常用"),)


def test_schema_only_offers_taxonomy_tags():
    schema = quotes.output_schema(TAX)
    item = schema["properties"]["quotes"]["items"]["properties"]
    assert item["situations"]["items"]["enum"] == ["上班", "上學考試"]
    assert item["characters"]["items"]["enum"] == ["海綿寶寶", "章魚哥"]


def test_user_prompt_lists_cue_number_minute_and_text():
    meta = parse_episode_name("海綿寶寶_S01_ep001_急徵店員.zh-Hant.srt")
    prompt = quotes.user_prompt(meta, [Cue(7, 65.4, 66.0, "我準備好了\n我準備好了")])
    assert prompt.splitlines()[-1] == "7|01:05|我準備好了 我準備好了"


def test_repository_taxonomy_loads():
    tax = load_taxonomy(quotes.Path(__file__).resolve().parents[1] / "taxonomy.yml")
    assert set(tax.facets) == set(quotes.FACETS)


def test_taxonomy_rejects_duplicates(tmp_path):
    p = tmp_path / "t.yml"
    p.write_text("situations: [上班, 上班]\nemotions: [a]\nfunctions: [b]\ncharacters: [c]\n")
    with pytest.raises(TaxonomyError):
        load_taxonomy(p)


def test_index_version_changes_with_taxonomy():
    other = Taxonomy({**TAX.facets, "emotions": ("開心",)})
    assert quotes.index_version(TAX) != quotes.index_version(other)
