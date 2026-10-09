from types import SimpleNamespace

from psr import cli
from psr import punctuate as punct_mod


def test_failed_chunks_fall_back_to_traditional_text(monkeypatch):
    # S01E009 實測：加標點失敗的塊直接沿用 Whisper 的簡體原文（「醒醒孩子你在烧我的钱」）。
    monkeypatch.setattr(punct_mod, "make_chunks", lambda words: ["醒醒孩子你在烧我的钱"])
    monkeypatch.setattr(punct_mod, "punctuate_chunk", lambda chunk, client: (None, 0, 0, "fail"))
    text, failed, n, _, _ = cli._punctuate([SimpleNamespace()], client=None, executor_workers=1)
    assert text == "醒醒孩子你在燒我的錢"
    assert failed == [0] and n == 1
