from psr import lexical


def test_bigrams_skip_punctuation_and_whitespace():
    assert lexical.bigrams("就像開，玉米罐！") == [
        "就像",
        "像開",
        "開玉",
        "玉米",
        "米罐",
    ]
    assert lexical.bigrams("好") == ["好"]
    assert lexical.bigrams("。") == []


def test_bm25_ranks_documents_sharing_query_bigrams():
    index = lexical.Bm25({1: "今天天氣很好", 2: "我想逃離那一切", 3: "逃離這個廚房"})
    got = index.search("好想逃離那一切", limit=10)
    assert got[0] == 2  # 共用「逃離」「離那」「那一」「一切」
    assert 3 in got  # 只共用「逃離」
    assert 1 not in got  # 沒有共同雙字詞


def test_bm25_rare_bigram_outweighs_common_one():
    docs = {i: "朋友朋友" for i in range(1, 9)}
    docs[9] = "朋友考試"
    docs[10] = "大巨蛋"
    index = lexical.Bm25(docs)
    assert index.search("大巨蛋朋友", limit=1) == [10]


def test_rrf_fuses_rankings_by_reciprocal_rank():
    # 1：1/61 + 1/62；3：1/63 + 1/61；2：1/62
    assert lexical.rrf([[1, 2, 3], [3, 1]], k=60) == [1, 3, 2]
    assert lexical.rrf([]) == []
