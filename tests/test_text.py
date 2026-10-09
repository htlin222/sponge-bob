from hypothesis import given, settings, strategies as st

from psr.text import display_width, normalize


def test_normalize_fullwidth_digits():
    normalized, orig_index = normalize("１２３")
    assert normalized == "123"
    assert orig_index == [0, 1, 2]


def test_normalize_traditional_simplified_fold_to_same_string():
    # 資料（繁體）與 资料（簡體）是同一個詞的繁簡字形差異，
    # 正規化後應該疊成同一個字串，這樣潤稿把繁簡混用也不會被誤判成內容不同。
    assert normalize("資料")[0] == normalize("资料")[0] == "资料"


def test_normalize_strips_punctuation():
    normalized, orig_index = normalize("你好，世界！")
    assert normalized == "你好世界"


def test_normalize_casefolds_latin():
    assert normalize("Python")[0] == normalize("python")[0] == "python"


@given(st.text(min_size=0, max_size=50))
@settings(max_examples=300)
def test_normalize_index_map_is_valid(s):
    normalized, orig_index = normalize(s)
    assert len(normalized) == len(orig_index)
    for idx in orig_index:
        assert 0 <= idx < len(s)
    for i in range(len(orig_index) - 1):
        assert orig_index[i] <= orig_index[i + 1]


def test_display_width_pure_chinese():
    assert display_width("你好嗎最近過得如何呀") == 10.0


def test_display_width_pure_latin():
    assert display_width("Python") == 3.0


def test_display_width_mixed():
    assert display_width("Python你好") == 5.0


def test_to_traditional_converts_glyphs_only():
    from psr.text import to_traditional
    # 只轉字形，不轉詞彙——詞彙層的台灣化是 glossary.yml 的職責
    assert to_traditional("软件工程师写程序") == "軟件工程師寫程序"
    assert to_traditional("网络和数据库") == "網絡和數據庫"
    assert to_traditional("这个视频的内存优化") == "這個視頻的內存優化"


def test_to_traditional_is_identity_on_traditional_text():
    from psr.text import to_traditional
    s = "這一句本來就是繁體，不該被動到"
    assert to_traditional(s) == s


def test_to_traditional_is_idempotent():
    # 確定性的最低要求：轉兩次跟轉一次一樣
    from psr.text import to_traditional
    s = "混合：我们說的软件跟我們講的軟體"
    once = to_traditional(s)
    assert to_traditional(once) == once


def test_to_traditional_leaves_english_alone():
    from psr.text import to_traditional
    assert to_traditional("用 Python 寫 prompt") == "用 Python 寫 prompt"


def test_to_traditional_does_not_rewrite_natural_taiwanese_wording():
    # s2twp 會把「打開」改成「開啟」、「權限」改成「許可權」。逐字稿是轉錄，
    # 不是在地化——改寫講者實際說出口的詞是越權，所以刻意只用 s2tw。
    from psr.text import to_traditional
    assert to_traditional("首先打開終端機") == "首先打開終端機"
    assert to_traditional("你沒有權限") == "你沒有權限"
