"""psr.polish 的測試。**零網路**——一律注入假 client，絕不打真實 DeepSeek API。"""

import json

from psr.glossary import Glossary, GlossaryEntry
from psr.models import Word
from psr.polish import (
    MAX_ATTEMPTS,
    _dedupe_exact_lines,
    _strip_context_echo,
    edit_budget,
    make_windows,
    polish_words,
)


# ---------------------------------------------------------------------------
# 假 client：模擬 OpenAI SDK 的 chat.completions.create() 回傳形狀，
# 讓 _polish_one_window / polish_words 不用改一行程式碼就能被測試。
# ---------------------------------------------------------------------------


class _Usage:
    def __init__(self, prompt_tokens=10, completion_tokens=10):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, lines, prompt_tokens=10, completion_tokens=10):
        content = json.dumps({"lines": lines}, ensure_ascii=False)
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _RawResponse:
    """直接指定 message.content 字串，用來製造格式錯誤的 JSON。"""

    def __init__(self, content, prompt_tokens=10, completion_tokens=10):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _Completions:
    def __init__(self, respond):
        """respond(call_index, kwargs) -> response 物件。call_index 從 0 起算，
        跨所有窗口共用同一個計數器（單執行緒／max_workers=1 時具決定性）。"""
        self._respond = respond
        self.calls = []

    def create(self, **kwargs):
        idx = len(self.calls)
        self.calls.append(kwargs)
        return self._respond(idx, kwargs)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, respond):
        self.completions = _Completions(respond)
        self.chat = _Chat(self.completions)


def fake_client_returning(*responses):
    """依序回傳固定的 response 清單，多打幾次就重複最後一個。"""
    def respond(idx, kwargs):
        return responses[min(idx, len(responses) - 1)]
    return FakeClient(respond)


def fake_client_from_fn(fn):
    return FakeClient(fn)


def _words(text: str, start: float = 0.0, step: float = 0.1) -> list[Word]:
    out = []
    t = start
    for ch in text:
        out.append(Word(ch, t, t + step))
        t += step
    return out


_EMPTY_GLOSSARY = Glossary(entries=(), raw_bytes=b"")


# ---------------------------------------------------------------------------
# make_windows: 切窗邊界
# ---------------------------------------------------------------------------


def test_make_windows_splits_once_length_reaches_threshold():
    words = _words("abcde")  # 5 words，每個 1 字元
    windows = make_windows(words, window_chars=3)
    # 累積到 >=3 就收：前 3 字一窗，剩下 2 字一窗
    assert [len(w) for w in windows] == [3, 2]
    assert "".join(w.text for w in windows[0]) == "abc"
    assert "".join(w.text for w in windows[1]) == "de"


def test_make_windows_exact_boundary_closes_window():
    words = _words("abc")
    windows = make_windows(words, window_chars=3)
    assert len(windows) == 1
    assert len(windows[0]) == 3


def test_make_windows_empty_input():
    assert make_windows([], window_chars=1800) == []


def test_make_windows_no_overlap():
    words = _words("abcdefghij")
    windows = make_windows(words, window_chars=4)
    flat = [w for win in windows for w in win]
    assert flat == words  # 串起來剛好是原始清單，沒有重複也沒有缺漏


def test_make_windows_single_long_word_forms_its_own_window():
    words = [Word("word_longer_than_threshold", 0.0, 1.0), Word("x", 1.0, 1.1)]
    windows = make_windows(words, window_chars=5)
    assert len(windows) == 2
    assert windows[0][0].text == "word_longer_than_threshold"


# ---------------------------------------------------------------------------
# edit_budget: 不對稱的刪除/插入比例
# ---------------------------------------------------------------------------


def test_edit_budget_identity_is_zero():
    assert edit_budget("大家好我是林醫師", "大家好我是林醫師") == (0.0, 0.0)


def test_edit_budget_pure_insertion_legitimate_bpe_repair():
    # Whisper 把 prompt 截斷成 Promp，模型補回遺漏的字母 t——
    # 這是合法的插入，插入比例應該遠低於 15% 的上限。
    original = "這個 Promp 很好用"
    polished = "這個 prompt 很好用"
    deletion, insertion = edit_budget(original, polished)
    assert deletion <= 0.12
    assert insertion <= 0.15


def test_edit_budget_deletion_heavy_exceeds_budget():
    original = "大家好，今天我們要來介紹一下這個新工具的用法跟安裝步驟"
    polished = "介紹"  # 幾乎整段被刪掉
    deletion, insertion = edit_budget(original, polished)
    assert deletion > 0.12


def test_edit_budget_replace_of_equal_length_nets_to_zero():
    # replace 區塊只計「長度差」那一段：等長替換（同音字修正這類）在刪除/
    # 插入比例上都是 0，因為它既沒有淨掉內容也沒有淨增內容。
    original = "今天天氣很好我們去公園散步"
    same_length_replace = "今天天氣很好我們去山上露營"  # 「公園散步」與「山上露營」等長
    deletion, insertion = edit_budget(original, same_length_replace)
    assert (deletion, insertion) == (0.0, 0.0)


def test_edit_budget_replace_of_unequal_length_splits_into_both_ratios():
    # 對稱 ratio 會把「刪一段、補另一段不同長度的內容」混成同一個數字；
    # 分開量測後，長度差的部分該落在刪除還是插入要能分辨出來。
    original = "今天天氣很好我們去公園散步"
    shorter_fabrication = "今天天氣很好我們去爬山"  # 「公園散步」(4字) 換成「爬山」(2字)：淨刪除 2 字
    deletion, insertion = edit_budget(original, shorter_fabrication)
    assert deletion > 0.0
    assert insertion == 0.0

    longer_fabrication = "今天天氣很好我們去爬山露營遠足健行"  # 換成 8 字：淨插入 4 字
    deletion2, insertion2 = edit_budget(original, longer_fabrication)
    assert deletion2 == 0.0
    assert insertion2 > 0.0


# ---------------------------------------------------------------------------
# _strip_context_echo
# ---------------------------------------------------------------------------


def test_strip_context_echo_removes_leading_line_matching_before():
    before = "上一段的結尾內容在這裡"
    lines = ["上一段的結尾內容在這裡", "這才是這個窗口真正的第一行"]
    result = _strip_context_echo(lines, before, "")
    assert result == ["這才是這個窗口真正的第一行"]


def test_strip_context_echo_removes_trailing_line_matching_after():
    after = "下一段的開頭內容在這裡"
    lines = ["這是本窗口真正的最後一行", "下一段的開頭內容在這裡"]
    result = _strip_context_echo(lines, "", after)
    assert result == ["這是本窗口真正的最後一行"]


def test_strip_context_echo_no_context_is_noop():
    lines = ["第一行", "第二行"]
    assert _strip_context_echo(lines, "", "") == lines


def test_strip_context_echo_does_not_touch_unrelated_lines():
    lines = ["完全無關的內容", "另一行也無關"]
    result = _strip_context_echo(lines, "前一窗的文字", "後一窗的文字")
    assert result == lines


# ---------------------------------------------------------------------------
# _dedupe_exact_lines
# ---------------------------------------------------------------------------


def test_dedupe_keeps_repeated_short_lines():
    # 「好的」這種短行講者真的可能連續講兩次，不該被去重規則吃掉。
    lines = ["好的", "這是一句夠長會被判定要去重的示範句子", "好的"]
    assert _dedupe_exact_lines(lines) == lines


def test_dedupe_removes_exact_duplicate_long_lines():
    long_line = "這是一句長度超過門檻會被去重規則處理的示範句子內容"
    lines = [long_line, "中間插一句別的話", long_line]
    result = _dedupe_exact_lines(lines)
    assert result == [long_line, "中間插一句別的話"]


def test_dedupe_does_not_remove_substring_of_a_longer_line():
    # 只刪「完全相同」的行，不能因為短行是長行的子字串就刪掉。
    lines = ["這是一句長度超過門檻會被去重規則處理的示範句子", "示範句子"]
    assert _dedupe_exact_lines(lines) == lines


# ---------------------------------------------------------------------------
# polish_words：端到端（假 client，max_workers=1 保證決定性順序）
# ---------------------------------------------------------------------------


def test_polish_words_single_window_accepted():
    text = "大家好我是林醫師今天我們來介紹一下這個新工具的用法"
    words = _words(text)
    client = fake_client_returning(_Response(["大家好，我是林醫師。", "今天我們來介紹一下這個新工具的用法。"]))

    windows, result = polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    assert len(windows) == 1
    assert windows[0] is not None
    assert result.degraded == []
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 10


def test_polish_words_retries_then_degrades_to_none(monkeypatch):
    import psr.polish as polish_mod

    slept = []
    monkeypatch.setattr(polish_mod.time, "sleep", lambda s: slept.append(s))

    text = "大家好我是林醫師今天我們來介紹一下這個新工具的用法跟安裝步驟"
    words = _words(text)
    # 每次都只回一個字，遠遠超過刪除比例上限，三次重試都不會過關。
    client = fake_client_returning(_Response(["介"]))

    windows, result = polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    assert windows == [None]
    assert result.degraded_count == 1
    idx, err = result.degraded[0]
    assert idx == 0
    assert "刪除" in err
    assert len(client.completions.calls) == MAX_ATTEMPTS
    # 三次重試中，前兩次失敗後應該有退避（第三次失敗後不需要，因為沒有下一次了）
    assert slept == [3, 10]


def test_polish_words_retries_then_succeeds_on_second_attempt(monkeypatch):
    import psr.polish as polish_mod

    monkeypatch.setattr(polish_mod.time, "sleep", lambda s: None)

    text = "大家好我是林醫師今天我們來介紹一下這個新工具的用法跟安裝步驟"
    words = _words(text)
    bad = _Response(["介"])  # 刪除過多，第一次失敗
    good = _Response(["大家好，我是林醫師。", "今天我們來介紹一下這個新工具的用法跟安裝步驟。"])
    client = fake_client_returning(bad, good)

    windows, result = polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    assert windows[0] is not None
    assert result.degraded == []
    # 兩次呼叫的 token 用量都該計入總花費，即使第一次失敗了。
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 20


def test_polish_words_malformed_json_retries_then_degrades(monkeypatch):
    import psr.polish as polish_mod

    monkeypatch.setattr(polish_mod.time, "sleep", lambda s: None)

    text = "大家好我是林醫師今天我們來介紹一下這個新工具"
    words = _words(text)
    client = fake_client_returning(_RawResponse("這不是合法的 JSON"))

    windows, result = polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    assert windows == [None]
    assert result.degraded_count == 1
    assert len(client.completions.calls) == MAX_ATTEMPTS


def test_polish_words_uses_seed_42_plus_attempt(monkeypatch):
    import psr.polish as polish_mod

    monkeypatch.setattr(polish_mod.time, "sleep", lambda s: None)

    text = "大家好我是林醫師今天我們來介紹一下這個新工具"
    words = _words(text)
    client = fake_client_returning(_Response(["介"]))  # 一路失敗到底

    polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    seeds = [call["seed"] for call in client.completions.calls]
    assert seeds == [42, 43, 44]


def test_polish_words_request_has_no_context_fields():
    # 硬規則 #2：送給模型的訊息裡不可以出現前後文——只能有 system + user
    # 兩則訊息，user 就是這個窗口自己的文字。
    text = "第一窗的內容在這裡" * 3
    words = _words(text, step=0.05)
    client = fake_client_returning(_Response([text]))

    polish_words(words, _EMPTY_GLOSSARY, client=client, max_workers=1)

    call = client.completions.calls[0]
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1]["content"] == text
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0
    assert call["top_p"] == 1
    assert call["max_tokens"] == 8000


def test_polish_words_empty_input_returns_empty():
    windows, result = polish_words([], _EMPTY_GLOSSARY, client=fake_client_returning(_Response([])))
    assert windows == []
    assert result.degraded == []
    assert result.prompt_tokens == 0


def test_polish_words_includes_glossary_hint_in_system_prompt():
    glossary = Glossary(
        entries=(
            GlossaryEntry(correct="Claude Code", wrong=("Cloud Code",), whisper_hint=True, note=None),
        ),
        raw_bytes=b"",
    )
    text = "我們來用 Cloud Code 寫程式"
    words = _words(text, step=0.05)
    client = fake_client_returning(_Response([text]))

    polish_words(words, glossary, client=client, max_workers=1)

    system_content = client.completions.calls[0]["messages"][0]["content"]
    assert "Claude Code←Cloud Code" in system_content
