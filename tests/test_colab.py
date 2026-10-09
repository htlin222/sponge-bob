from types import SimpleNamespace

import pytest

from psr.asr import colab
from psr.asr.colab import ColabSession, ColabUnavailable, RemoteJobFailed


def _fake_colab(exec_stdout, exec_stderr=""):
    def fake(*args, timeout=None):
        if args[0] == "exec":
            return SimpleNamespace(returncode=0, stdout=exec_stdout, stderr=exec_stderr)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake


def _session():
    s = ColabSession()
    s.opened = True
    return s


def test_python_exception_in_remote_job_is_a_code_error(monkeypatch, tmp_path):
    out = ("---> 46     with av.open(input_file) as container:\n"
           "TypeError: open() got an unexpected keyword argument 'metadata_errors'")
    monkeypatch.setattr(colab, "_colab", _fake_colab(out))
    with pytest.raises(RemoteJobFailed, match="metadata_errors"):
        _session().transcribe("drive", "id", "tok", "", tmp_path)


def test_missing_ok_marker_without_exception_is_infrastructure(monkeypatch, tmp_path):
    monkeypatch.setattr(colab, "_colab", _fake_colab("", "kernel disconnected"))
    with pytest.raises(ColabUnavailable):
        _session().transcribe("drive", "id", "tok", "", tmp_path)


def test_remote_job_failed_is_not_a_colab_unavailable():
    # batch 只對 ColabUnavailable 重開 session、只對它判定「配額用完」。
    assert not issubclass(RemoteJobFailed, ColabUnavailable)
