from psr import evalset


def test_rank_is_one_based_position_or_none():
    assert evalset.rank_of(7, [3, 7, 9]) == 2
    assert evalset.rank_of(3, [3, 7, 9]) == 1
    assert evalset.rank_of(4, [3, 7, 9]) is None


def test_metrics_hit_rate_and_mrr():
    # 四題：第 1 名、第 3 名、第 12 名、沒找到
    m = evalset.metrics([1, 3, 12, None], ks=(1, 5, 10, 30))
    assert m["n"] == 4
    assert m["hit@1"] == 0.25
    assert m["hit@5"] == 0.5
    assert m["hit@10"] == 0.5
    assert m["hit@30"] == 0.75
    assert abs(m["mrr"] - (1 + 1 / 3 + 1 / 12 + 0) / 4) < 1e-9


def test_metrics_empty():
    assert evalset.metrics([], ks=(1, 5)) == {
        "n": 0,
        "hit@1": 0.0,
        "hit@5": 0.0,
        "mrr": 0.0,
    }


def test_sample_one_quote_per_episode_and_deterministic():
    # (quote_id, episode_id, quotability)
    rows = [(i, i // 3, 4) for i in range(30)]  # 10 集、每集 3 段
    a = evalset.sample(rows, n=5, seed=1)
    assert a == evalset.sample(rows, n=5, seed=1)
    assert len(a) == 5
    assert len({i // 3 for i in a}) == 5  # 不同集


def test_sample_skips_low_quotability():
    rows = [(1, 1, 2), (2, 2, 3), (3, 3, 5)]
    assert sorted(evalset.sample(rows, n=10, seed=0)) == [2, 3]


def test_leaks_when_query_copies_four_chars_of_the_line():
    line = "就像開玉米罐一樣"
    assert evalset.leaks("朋友要面對開玉米罐這種小事", line)
    assert not evalset.leaks("朋友覺得眼前的事很難", line)
    # 標點與換行不算字
    assert not evalset.leaks("像開，玉", "像開，\n玉米")


def test_accept_keeps_only_batch_ids_and_drops_leaky_or_empty_queries():
    lines = {10: "就像開玉米罐一樣", 11: "我準備好了"}
    raw = {
        "items": [
            {
                "id": 10,
                "colloquial": "朋友怕自己做不到",
                "abstract": "鼓勵",
                "topic": "開玉米罐比賽",
            },
            {"id": 11, "colloquial": "", "abstract": "迎接挑戰", "topic": "考試"},
            {"id": 99, "colloquial": "不在這批", "abstract": "x", "topic": "y"},
        ]
    }
    got = evalset.accept(raw, lines)
    assert got == [
        (10, "colloquial", "朋友怕自己做不到"),
        (10, "abstract", "鼓勵"),
        (11, "abstract", "迎接挑戰"),
        (11, "topic", "考試"),
    ]
