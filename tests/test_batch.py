import pytest

from psr.batch import BatchError, plan, validate_folder_id


def _video(name, id_):
    return {"id": id_, "name": name, "mimeType": "video/mp4", "md5Checksum": f"md5-{id_}"}


def _file(name, id_):
    return {"id": id_, "name": name, "mimeType": "text/plain"}


def test_skips_episodes_that_already_have_srt():
    children = [
        _video("海綿寶寶_S01_ep001_急徵店員.mp4", "a"),
        _file("海綿寶寶_S01_ep001_急徵店員.zh-Hant.srt", "a-srt"),
        _video("海綿寶寶_S01_ep002_吹泡泡.mp4", "b"),
    ]
    assert [e.id for e in plan(children, [])] == ["b"]


def test_orders_by_name_regardless_of_listing_order():
    children = [_video("S02_ep010.mp4", "c"), _video("S01_ep002.mp4", "b"), _video("S01_ep001.mp4", "a")]
    assert [e.id for e in plan(children, [])] == ["a", "b", "c"]


def test_ignores_non_video_files_and_work_folder():
    children = [
        {"id": "w", "name": "_psr", "mimeType": "application/vnd.google-apps.folder"},
        _file("notes.txt", "n"),
        _video("ep1.mp4", "a"),
    ]
    assert [e.id for e in plan(children, [])] == ["a"]


def test_attaches_cached_transcript_from_work_folder():
    children = [_video("ep1.mp4", "a"), _video("ep2.mp4", "b")]
    work = [_file("ep1.words.json", "w1"), _file("ep1.manifest.json", "m1")]
    eps = {e.id: e for e in plan(children, work)}
    assert (eps["a"].cached_words_id, eps["a"].cached_manifest_id) == ("w1", "m1")
    assert (eps["b"].cached_words_id, eps["b"].cached_manifest_id) == (None, None)


def test_stem_keeps_parentheses_and_inner_dots():
    eps = plan([_video("海綿寶寶_S16_ep323_時空逆轉(上).v2.mp4", "a")], [])
    assert eps[0].stem == "海綿寶寶_S16_ep323_時空逆轉(上).v2"


def test_limit_takes_first_n_pending_and_zero_means_all():
    children = [_video(f"ep{i}.mp4", str(i)) for i in range(5)]
    assert [e.id for e in plan(children, [], limit=2)] == ["0", "1"]
    assert len(plan(children, [], limit=0)) == 5


@pytest.mark.parametrize("bad", ["", "abc", "x' or '1'='1", "a" * 101, "10RAAX/ly"])
def test_rejects_malformed_folder_id(bad):
    with pytest.raises(BatchError):
        validate_folder_id(bad)


def test_accepts_real_folder_id_and_strips_whitespace():
    assert validate_folder_id("  1AbCdEfGh_ijklMNOPqrstUVwx-yz0123\n") == "1AbCdEfGh_ijklMNOPqrstUVwx-yz0123"
