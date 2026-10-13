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


def test_refs_load_is_idempotent_and_listed():
    db = _db()
    quotedb.load_refs(db, [{"phrase": "老實說熊熊", "kind": "meme", "source": "u"}])
    quotedb.load_refs(db, [{"phrase": "老實說熊熊", "kind": "name", "source": "v"},
                           {"phrase": "野海熊防護圈"}])
    assert quotedb.ref_phrases(db) == ["老實說熊熊", "野海熊防護圈"]


def test_scan_verdicts_round_trip():
    db = _db()
    quotedb.record_scan(db, [("ep79", "老師說，小熊", "老實說熊熊", True, "老師說，小熊", "老實說熊熊", "ref"),
                             ("ep1", "我的防", "我的房", False, "", "", "homophone")])
    assert quotedb.scan_verdicts(db) == {
        ("ep79", "老師說，小熊", "老實說熊熊"): (True, "老師說，小熊", "老實說熊熊"),
        ("ep1", "我的防", "我的房"): (False, "", ""),
    }


def test_episode_fixes_grouped_by_stem():
    db = _db()
    quotedb.add_episode_fixes(db, [("ep79", "老蜥蜴小熊", "老實說熊熊", "scan"),
                                   ("ep79", "老師說小熊", "老實說熊熊", "scan"),
                                   ("ep80", "阿密巴", "阿米巴", "scan")])
    got = quotedb.episode_fixes(db)
    assert sorted(got["ep79"]) == [("老師說小熊", "老實說熊熊"), ("老蜥蜴小熊", "老實說熊熊")]
    assert got["ep80"] == [("阿密巴", "阿米巴")]


def test_vectors_are_listed_saved_and_dropped_with_their_episode():
    db = _db(_quote(1, 1, "我準備好了", "上班前"))
    [(qid, text, use_when, digest)] = quotedb.quote_docs(db)
    assert (text, use_when, digest) == ("我準備好了", "上班前", None)
    quotedb.save_vectors(db, [(qid, "d1", [1.0, 0.0])])
    assert quotedb.quote_docs(db)[0][3] == "d1"
    quotedb.write_episode(db, META, drive_id="d1", srt_md5="m2", version="v2", cues=CUES,
                          result=Indexed((_quote(2, 2, "又是星期一", "週一"),), (), (), 0))
    assert db.execute("SELECT count(*) FROM quote_vecs") == [(0,)]


def test_nearest_ranks_by_cosine_similarity():
    db = _db(_quote(1, 1, "我準備好了", "上班前"), _quote(2, 2, "又是星期一", "週一"))
    ids = {text: qid for qid, text, _, _ in quotedb.quote_docs(db)}
    quotedb.save_vectors(db, [(ids["我準備好了"], "a", [1.0, 0.0]), (ids["又是星期一"], "b", [0.0, 1.0])])
    hits = quotedb.nearest(db, [0.2, 0.9], limit=2)
    assert [h["text"] for h in hits] == ["又是星期一", "我準備好了"]
    assert hits[0]["title"] == META.title and hits[0]["use_when"] == "週一"


def test_grep_cues_matches_any_term_in_episode_order():
    db = _db()
    rows = quotedb.grep_cues(db, ["星期一", "完美"])
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("海綿寶寶_S01_ep001_急徵店員", 1.0, "又是星期一"),
        ("海綿寶寶_S01_ep001_急徵店員", 2.0, "100%完美"),
    ]


def test_grep_cues_treats_percent_literally():
    db = _db()
    assert [r[2] for r in quotedb.grep_cues(db, ["%"])] == ["100%完美"]


def test_dump_copies_every_table_row_for_row(tmp_path):
    db = _db(_quote(1, 1, "我準備好了", "上班前", ("situations", "上班")))
    (qid, *_), = quotedb.quote_docs(db)
    quotedb.save_vectors(db, [(qid, "d", [1.0, 0.5])])
    dest = tmp_path / "backup.sqlite"
    counts = quotedb.dump(db, dest, page=2)  # 每頁 2 筆，cues 3 筆要翻頁

    import sqlite3
    out = sqlite3.connect(dest)
    for table in ("episodes", "cues", "quotes", "quote_tags", "quote_vecs", "tags"):
        want = db.execute(f"SELECT * FROM {table} ORDER BY rowid")
        assert out.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() == want, table
        assert counts[table] == len(want)


def test_turso_blobs_decode_without_base64_padding():
    # Hrana 回傳的 BLOB 是不補 = 的 base64：b"\x01\x02" 是 "AQI"
    assert quotedb._decode({"type": "blob", "base64": "AQI"}) == b"\x01\x02"
    assert quotedb._decode({"type": "blob", "base64": "AQI="}) == b"\x01\x02"


def test_quotes_by_id_keeps_requested_order_and_skips_missing():
    db = _db(_quote(1, 1, "我準備好了", "上班前"), _quote(2, 2, "又是星期一", "週一"))
    ids = {text: qid for qid, text, _, _ in quotedb.quote_docs(db)}
    got = quotedb.quotes_by_id(db, [ids["又是星期一"], 999, ids["我準備好了"]])
    assert [h["text"] for h in got] == ["又是星期一", "我準備好了"]
    assert got[0]["title"] == META.title and got[0]["distance"] is None
    assert quotedb.quotes_by_id(db, []) == []
