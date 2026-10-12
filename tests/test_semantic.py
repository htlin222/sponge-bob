from psr import semantic


def test_doc_is_the_situation_not_the_line():
    assert semantic.doc("我準備好了", "上班前打氣") == "上班前打氣"


def test_stale_lists_missing_and_changed_vectors_only():
    fresh = semantic.digest(semantic.doc("又是星期一", "週一"))
    rows = [(1, "我準備好了", "上班前", None),
            (2, "又是星期一", "週一", fresh),
            (3, "100%完美", "炫耀", "old-digest")]
    assert [qid for qid, _, _ in semantic.stale(rows)] == [1, 3]


def test_digest_changes_with_text():
    assert semantic.digest("a") != semantic.digest("b")
