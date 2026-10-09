import pathlib
import pytest

from psr.asr import groq as g


def test_single_chunk_when_audio_is_small(tmp_path, monkeypatch):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * 1024)
    monkeypatch.setattr(g, "probe_duration", lambda p: 600.0)
    assert g.plan_chunks(f) == [(0.0, 600.0)]


def test_splits_when_over_limit_and_snaps_to_silence(tmp_path, monkeypatch):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * (g.MAX_CHUNK_BYTES * 2 + 10))   # 需要 3 塊
    monkeypatch.setattr(g, "probe_duration", lambda p: 900.0)
    # 靜音點刻意偏離等分位置，驗證切點會吸附過去
    monkeypatch.setattr(g, "silence_points", lambda p, n: [290.0, 610.0])
    chunks = g.plan_chunks(f)
    assert len(chunks) == 3
    assert chunks[0] == (0.0, 290.0)
    assert chunks[1] == (290.0, 610.0)
    assert chunks[-1][1] == 900.0


def test_falls_back_to_even_cuts_when_no_silence_found(tmp_path, monkeypatch):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * (g.MAX_CHUNK_BYTES + 10))
    monkeypatch.setattr(g, "probe_duration", lambda p: 100.0)
    monkeypatch.setattr(g, "silence_points", lambda p, n: [])
    chunks = g.plan_chunks(f)
    assert len(chunks) == 2
    assert chunks[0][1] == pytest.approx(50.0)


def test_chunks_are_contiguous_and_cover_the_whole_audio(tmp_path, monkeypatch):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * (g.MAX_CHUNK_BYTES * 3))
    monkeypatch.setattr(g, "probe_duration", lambda p: 1200.0)
    monkeypatch.setattr(g, "silence_points", lambda p, n: [100.0, 500.0, 900.0])
    chunks = g.plan_chunks(f)
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 1200.0
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0], "切塊之間不可有空隙或重疊"
