import pytest

from psr.drive import FileInfo, PreflightError, find_or_create, preflight


# ---------------------------------------------------------------------------
# 零網路 fakes：模仿 service.files().get()/.list()/.update()/.create().execute()
# 的巢狀呼叫鏈，不碰真正的 Google API。
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFiles:
    def __init__(self, get_result=None, list_result=None, update_result=None, create_result=None):
        self._get_result = get_result
        self._list_result = list_result if list_result is not None else {"files": []}
        self._update_result = update_result
        self._create_result = create_result
        self.get_calls: list[dict] = []
        self.list_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return _FakeRequest(self._get_result)

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return _FakeRequest(self._list_result)

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        return _FakeRequest(self._update_result)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeRequest(self._create_result)


class _FakeService:
    def __init__(self, files: _FakeFiles):
        self._files = files

    def files(self):
        return self._files


def _video_meta(**overrides):
    meta = {
        "id": "file123",
        "name": "tutorial.mp4",
        "mimeType": "video/mp4",
        "size": str(1 * 1024**3),  # 1 GB, as a string like the real API returns
        "md5Checksum": "dc59346753cd8fb79b361c87f22fa46a",
        "parents": ["folder456"],
    }
    meta.update(overrides)
    return meta


class TestPreflight:
    def test_rejects_non_video_mime_type(self):
        files = _FakeFiles(get_result=_video_meta(mimeType="application/pdf"))
        service = _FakeService(files)
        with pytest.raises(PreflightError, match="不是影片檔"):
            preflight(service, "file123")

    def test_rejects_oversized_file(self):
        oversized = 9 * 1024**3  # 9 GB > 8 GB cap
        files = _FakeFiles(get_result=_video_meta(size=str(oversized)))
        service = _FakeService(files)
        with pytest.raises(PreflightError, match="超過 8 GB"):
            preflight(service, "file123")

    def test_surfaces_md5_checksum_without_downloading(self):
        files = _FakeFiles(get_result=_video_meta(md5Checksum="dc59346753cd8fb79b361c87f22fa46a"))
        service = _FakeService(files)
        info = preflight(service, "file123")
        assert isinstance(info, FileInfo)
        assert info.md5_checksum == "dc59346753cd8fb79b361c87f22fa46a"
        assert info.mime_type == "video/mp4"
        assert info.size == 1 * 1024**3
        assert info.parents == ["folder456"]

    def test_accepts_file_at_exactly_the_size_cap(self):
        files = _FakeFiles(get_result=_video_meta(size=str(8 * 1024**3)))
        service = _FakeService(files)
        info = preflight(service, "file123")
        assert info.size == 8 * 1024**3

    def test_only_requests_the_needed_fields(self):
        files = _FakeFiles(get_result=_video_meta())
        service = _FakeService(files)
        preflight(service, "file123")
        assert files.get_calls == [
            {"fileId": "file123", "fields": "id,name,mimeType,size,md5Checksum,parents"}
        ]


class TestFindOrCreate:
    def test_creates_when_no_same_named_file_exists(self, tmp_path):
        local_file = tmp_path / "tutorial.zh-Hant.srt"
        local_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

        files = _FakeFiles(
            list_result={"files": []},
            create_result={"id": "new1", "name": "tutorial.zh-Hant.srt", "webViewLink": "https://drive/new1"},
        )
        service = _FakeService(files)

        meta = find_or_create(service, "tutorial.zh-Hant.srt", "folder456", str(local_file), "text/plain")

        assert meta.id == "new1"
        assert meta.web_view_link == "https://drive/new1"
        assert len(files.create_calls) == 1
        assert files.update_calls == []
        assert files.create_calls[0]["body"] == {"name": "tutorial.zh-Hant.srt", "parents": ["folder456"]}

    def test_updates_when_same_named_file_already_exists(self, tmp_path):
        local_file = tmp_path / "tutorial.zh-Hant.srt"
        local_file.write_text("second run\n", encoding="utf-8")

        files = _FakeFiles(
            list_result={"files": [{"id": "existing1", "name": "tutorial.zh-Hant.srt"}]},
            update_result={"id": "existing1", "name": "tutorial.zh-Hant.srt", "webViewLink": "https://drive/existing1"},
        )
        service = _FakeService(files)

        meta = find_or_create(service, "tutorial.zh-Hant.srt", "folder456", str(local_file), "text/plain")

        assert meta.id == "existing1"
        assert meta.web_view_link == "https://drive/existing1"
        assert len(files.update_calls) == 1
        assert files.update_calls[0]["fileId"] == "existing1"
        assert files.create_calls == []

    def test_repeated_runs_stay_idempotent_not_triplicated(self, tmp_path):
        # 模擬三次重跑：第一次沒有同名檔案（create），之後兩次都找得到剛
        # 建立的那個檔案（update）。同名檔案應該永遠只有一個。
        local_file = tmp_path / "tutorial.zh-Hant.srt"
        local_file.write_text("run\n", encoding="utf-8")

        files = _FakeFiles(
            list_result={"files": []},
            create_result={"id": "only1", "name": "tutorial.zh-Hant.srt", "webViewLink": "https://drive/only1"},
        )
        service = _FakeService(files)
        first = find_or_create(service, "tutorial.zh-Hant.srt", "folder456", str(local_file), "text/plain")

        files._list_result = {"files": [{"id": "only1", "name": "tutorial.zh-Hant.srt"}]}
        files._update_result = {"id": "only1", "name": "tutorial.zh-Hant.srt", "webViewLink": "https://drive/only1"}
        second = find_or_create(service, "tutorial.zh-Hant.srt", "folder456", str(local_file), "text/plain")
        third = find_or_create(service, "tutorial.zh-Hant.srt", "folder456", str(local_file), "text/plain")

        assert first.id == second.id == third.id == "only1"
        assert len(files.create_calls) == 1
        assert len(files.update_calls) == 2

    def test_query_string_escapes_apostrophe_in_name(self, tmp_path):
        local_file = tmp_path / "weird.srt"
        local_file.write_text("x\n", encoding="utf-8")

        files = _FakeFiles(
            list_result={"files": []},
            create_result={"id": "id1", "name": "it's a test.srt"},
        )
        service = _FakeService(files)

        find_or_create(service, "it's a test.srt", "folder456", str(local_file), "text/plain")

        assert len(files.list_calls) == 1
        query = files.list_calls[0]["q"]
        assert "name = 'it\\'s a test.srt'" in query
        assert "'folder456' in parents" in query
        assert "trashed = false" in query

    def test_query_string_escapes_backslash_in_name(self, tmp_path):
        local_file = tmp_path / "weird2.srt"
        local_file.write_text("x\n", encoding="utf-8")

        files = _FakeFiles(list_result={"files": []}, create_result={"id": "id2", "name": "a\\b.srt"})
        service = _FakeService(files)

        find_or_create(service, "a\\b.srt", "folder456", str(local_file), "text/plain")

        query = files.list_calls[0]["q"]
        assert "name = 'a\\\\b.srt'" in query


def test_ensure_folder_reuses_an_existing_folder():
    from psr.drive import ensure_folder

    class Files:
        def list(self, **kw):
            self.q = kw["q"]
            return _Exec({"files": [{"id": "existing"}]})
        def create(self, **kw):
            raise AssertionError("已存在同名資料夾時不該再建一個")

    class _Exec:
        def __init__(self, payload): self.payload = payload
        def execute(self): return self.payload

    class Svc:
        def __init__(self): self._f = Files()
        def files(self): return self._f

    svc = Svc()
    assert ensure_folder(svc, "videos", None) == "existing"
    assert "mimeType = 'application/vnd.google-apps.folder'" in svc._f.q


def test_ensure_folder_creates_when_missing_and_escapes_quotes():
    from psr.drive import ensure_folder

    class _Exec:
        def __init__(self, payload): self.payload = payload
        def execute(self): return self.payload

    class Files:
        def list(self, **kw):
            self.q = kw["q"]
            return _Exec({"files": []})
        def create(self, **kw):
            self.body = kw["body"]
            return _Exec({"id": "new"})

    class Svc:
        def __init__(self): self._f = Files()
        def files(self): return self._f

    svc = Svc()
    assert ensure_folder(svc, "It's a folder", "parent1") == "new"
    assert "It\\'s a folder" in svc._f.q
    assert svc._f.body["parents"] == ["parent1"]


def test_with_retry_recovers_from_a_transient_broken_pipe():
    # 實測踩過：Colab T4 上花了 11 分鐘轉錄，接著在一個 metadata 查詢上
    # 撞到 BrokenPipeError，整趟 GPU 成果全部丟失。
    from psr import drive as d
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise BrokenPipeError(32, "Broken pipe")
        return {"ok": True}

    d._RETRY_DELAYS = (0, 0, 0, 0)          # 測試不真的等
    assert d.with_retry(flaky, "測試") == {"ok": True}
    assert calls["n"] == 3


def test_with_retry_gives_up_and_reports_which_call_failed():
    from psr import drive as d
    d._RETRY_DELAYS = (0, 0)

    def always_broken():
        raise BrokenPipeError(32, "Broken pipe")

    with pytest.raises(d.DriveError, match="查詢同名檔案"):
        d.with_retry(always_broken, "查詢同名檔案 x.srt")
