from psr import quotedb
from psr.models import Cue
from psr.quotes import Indexed, Issue, Quote, Taxonomy, parse_episode_name

TAX = Taxonomy({
    "situations": ("上班", "上學考試"),
    "emotions": ("開心", "崩潰"),
    "functions": ("鼓勵", "吐槽"),
    "characters": ("海綿寶寶", "章魚哥"),
})
META = parse_episode_name("海綿寶寶_S01_ep001_急徵店員.zh-Hant.srt")
CUES = [Cue(1, 0.0, 1.0, "我準備好了"), Cue(2, 1.0, 2.0, "又是星期一"),
        Cue(3, 2.0, 3.0, "100%完美")]


def _quote(a, b, text, use_when, *tags, score=4):
    return Quote(a, b, text, float(a), float(b), "海綿寶寶", score, use_when, tuple(tags))


def _db(*quotes_, issues=()):
    db = quotedb.SqliteDb()
    quotedb.init(db, TAX)
    quotedb.write_episode(db, META, drive_id="d1", srt_md5="m1", version="v1", cues=CUES,
                          result=Indexed(tuple(quotes_), tuple(issues), (), 0))
    return db


def test_rewriting_an_episode_replaces_its_rows():
    db = _db(_quote(1, 1, "我準備好了", "上班前", ("situations", "上班")),
             issues=[Issue(2, "錯字", "x", "y")])
    quotedb.write_episode(db, META, drive_id="d1", srt_md5="m2", version="v2", cues=CUES,
                          result=Indexed((_quote(2, 2, "又是星期一", "週一"),), (), (), 0))
    assert db.execute("SELECT cue_from FROM quotes") == [(2,)]
    assert db.execute("SELECT count(*) FROM quote_tags") == [(0,)]
    assert db.execute("SELECT count(*) FROM audit_issues") == [(0,)]
    assert db.execute("SELECT count(*) FROM cues") == [(3,)]
    assert quotedb.indexed_versions(db) == {META.stem: ("m2", "v2")}


def test_init_is_idempotent():
    db = _db()
    quotedb.init(db, TAX)
    assert db.execute("SELECT count(*) FROM tags") == [(8,)]


def test_search_scores_tags_two_text_three_use_when_two():
    db = _db(
        _quote(1, 1, "我準備好了", "上班前打氣", ("situations", "上班"), ("functions", "鼓勵")),
        _quote(2, 2, "又是星期一", "週一不想上班", ("situations", "上班")),
    )
    hits = quotedb.search(db, tags=[("situations", "上班"), ("functions", "鼓勵")],
                          keywords=["準備"])
    # 第 1 段：兩個標籤 4 分 + 台詞命中 3 分 = 7；第 2 段：一個標籤 2 分。
    assert [(h["text"], h["score"]) for h in hits] == [("我準備好了", 7), ("又是星期一", 2)]
    assert hits[0]["tags"] in ("上班、鼓勵", "鼓勵、上班")


def test_search_omits_quotes_without_any_hit():
    db = _db(_quote(1, 1, "我準備好了", "上班前"))
    assert quotedb.search(db, tags=[("emotions", "崩潰")], keywords=["考試"]) == []


def test_like_wildcards_in_keywords_are_literal():
    db = _db(_quote(1, 1, "我準備好了", "x"), _quote(3, 3, "100%完美", "y"))
    assert [h["text"] for h in quotedb.search(db, tags=[], keywords=["%"])] == ["100%完美"]


def test_ties_prefer_higher_quotability():
    db = _db(_quote(1, 1, "我準備好了", "上班", score=3),
             _quote(2, 2, "又是星期一", "上班", score=5))
    assert [h["text"] for h in quotedb.search(db, tags=[], keywords=["上班"])] == ["又是星期一", "我準備好了"]


def test_turso_values_round_trip():
    for v in (None, 3, 2.5, "海綿寶寶"):
        assert quotedb._decode(quotedb._encode(v)) == v


def test_refresh_text_keeps_tags_and_reslices_quotes():
    db = _db(_quote(1, 2, "我準備好了\n又是星期一", "上班前", ("situations", "上班")))
    fixed = [Cue(1, 0.0, 1.0, "我準備好了！"), Cue(2, 1.0, 2.0, "又是星期一"), Cue(3, 2.0, 3.0, "百分百完美")]
    quotedb.refresh_text(db, META.stem, srt_md5="m9", cues=fixed)
    assert db.execute("SELECT text FROM quotes") == [("我準備好了！\n又是星期一",)]
    assert db.execute("SELECT text FROM cues ORDER BY idx")[2] == ("百分百完美",)
    assert db.execute("SELECT count(*) FROM quote_tags") == [(1,)]
    assert quotedb.indexed_versions(db) == {META.stem: ("m9", "v1")}


def test_cue_timings_lists_index_start_end_in_order():
    db = _db()
    assert quotedb.cue_timings(db, META.stem) == [(1, 0.0, 1.0), (2, 1.0, 2.0), (3, 2.0, 3.0)]


def test_punctuation_cache_is_keyed_by_version():
    db = _db()
    assert quotedb.cached_punctuation(db, META.stem, "1") is None
    quotedb.cache_punctuation(db, META.stem, "1", "你好。／我很好！")
    assert quotedb.cached_punctuation(db, META.stem, "1") == "你好。／我很好！"
    assert quotedb.cached_punctuation(db, META.stem, "2") is None
