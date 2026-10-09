import pytest

from psr.youtube import ParseError, drive_paths, parse_youtube_id, slugify


class TestParseYoutubeId:
    def test_standard_watch_url(self):
        assert parse_youtube_id("影片在這 https://www.youtube.com/watch?v=dQw4w9WgXcQ 謝謝") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert parse_youtube_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"

    def test_shorts_and_embed_and_live(self):
        assert parse_youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert parse_youtube_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert parse_youtube_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url_with_leading_params(self):
        # v= 不一定是第一個參數
        assert parse_youtube_id("https://www.youtube.com/watch?app=desktop&v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_same_id_twice_is_still_one_video(self):
        text = "https://youtu.be/dQw4w9WgXcQ 跟 https://www.youtube.com/watch?v=dQw4w9WgXcQ 是同一支"
        assert parse_youtube_id(text) == "dQw4w9WgXcQ"

    def test_no_link_fails_loudly(self):
        with pytest.raises(ParseError, match="找不到"):
            parse_youtube_id("這個 issue 忘記貼連結了")

    def test_two_different_links_fail_rather_than_guess(self):
        # 猜錯的代價是花十幾分鐘轉錄了錯的影片，而且要看完才會發現
        text = "https://youtu.be/dQw4w9WgXcQ 和 https://youtu.be/aaaaaaaaaaa"
        with pytest.raises(ParseError, match="2 個"):
            parse_youtube_id(text)


class TestSlugify:
    def test_keeps_chinese_strips_symbols(self):
        assert slugify("提示詞（Prompt）設計與實作技巧") == "提示詞-Prompt-設計與實作技巧"

    def test_normalises_fullwidth_alphanumerics(self):
        assert slugify("ＡＩ　工作流") == "AI-工作流"

    def test_collapses_whitespace_and_trims_hyphens(self):
        assert slugify("  多餘   空白  ") == "多餘-空白"

    def test_strips_path_unsafe_characters(self):
        assert "/" not in slugify("a/b:c*d?e")
        assert slugify("a/b") == "a-b"

    def test_strips_emoji(self):
        assert slugify("🔥 深度學習 🚀") == "深度學習"

    def test_truncates_long_titles(self):
        assert len(slugify("字" * 200)) <= 80

    def test_never_returns_empty(self):
        # 純符號的標題不能產生空資料夾名
        assert slugify("！！！") == "untitled"
        assert slugify("") == "untitled"


def test_drive_paths_share_one_slug_prefix():
    paths = drive_paths("我的影片")
    assert paths["folder"] == "videos/我的影片"
    for key in ("video", "audio", "words", "srt", "manifest"):
        assert paths[key].startswith("我的影片.")
