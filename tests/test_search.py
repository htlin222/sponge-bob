from psr import quotedb, search
from tests.test_quotedb import _db, _quote


def _three():
    db = _db(_quote(1, 1, "我準備好了", "上班前"), _quote(2, 2, "又是星期一", "週一"),
             _quote(3, 3, "100%完美", "追求完美"))
    ids = {text: qid for qid, text, _, _ in quotedb.quote_docs(db)}
    quotedb.save_vectors(db, [(ids["我準備好了"], "a", [1.0, 0.0]), (ids["又是星期一"], "b", [0.0, 1.0]),
                              (ids["100%完美"], "c", [0.7, 0.7])])
    return db, ids


def test_verbatim_line_wins_even_when_the_vector_prefers_another_quote():
    db, ids = _three()
    # 向量排名：準備 > 完美 > 星期一；字面只有「星期一」命中
    # RRF(k=10)：星期一 1/13 + 1/11 > 準備 1/11 > 完美 1/12
    got = search.rank(db, ["又是星期一"], limit=3, embed=lambda q: [1.0, 0.0])
    assert got == [ids["又是星期一"], ids["我準備好了"], ids["100%完美"]]


def test_every_query_contributes_and_limit_applies():
    db, ids = _three()
    vectors = {"準備好": [1.0, 0.0], "星期幾": [0.0, 1.0]}
    got = search.rank(db, ["準備好", "星期幾"], limit=2, embed=vectors.__getitem__)
    assert set(got) == {ids["我準備好了"], ids["又是星期一"]}


def test_search_returns_rows_in_rank_order():
    db, ids = _three()
    hits = search.search(db, ["又是星期一"], limit=1, embed=lambda q: [1.0, 0.0])
    assert [h["text"] for h in hits] == ["又是星期一"]
