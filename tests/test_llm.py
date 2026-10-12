import pytest

from psr import claude_cli, llm


class _Client:
    """假 DeepSeek client：依序回傳預先給的 content，並記下請求。"""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.requests = []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.requests.append(kw)
        content = self.contents.pop(0)
        msg = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


SCHEMA = {"type": "object", "properties": {"fixes": {"type": "array"}}}


def test_deepseek_returns_the_json_object():
    client = _Client('{"fixes": [1]}')
    assert llm.ask("p", system="s", schema=SCHEMA, model="deepseek-v4-flash", client=client) == {"fixes": [1]}


def test_deepseek_gets_json_mode_and_the_schema_in_the_system_prompt():
    client = _Client("{}")
    llm.ask("字幕", system="你是校對", schema=SCHEMA, model="deepseek-v4-flash", client=client)
    req = client.requests[0]
    assert req["model"] == "deepseek-v4-flash"
    assert req["response_format"] == {"type": "json_object"}
    system, user = req["messages"]
    assert system["role"] == "system" and system["content"].startswith("你是校對")
    assert '"fixes"' in system["content"]
    assert user == {"role": "user", "content": "字幕"}


def test_deepseek_retries_once_on_broken_json():
    client = _Client("not json", '{"fixes": []}')
    assert llm.ask("p", system="s", schema=SCHEMA, model="deepseek-v4-flash", client=client) == {"fixes": []}


def test_deepseek_gives_up_after_two_broken_answers():
    client = _Client("not json", "[1, 2]")
    with pytest.raises(llm.LlmError):
        llm.ask("p", system="s", schema=SCHEMA, model="deepseek-v4-flash", client=client)


def test_other_models_go_through_claude_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(claude_cli, "ask", lambda prompt, **kw: calls.append((prompt, kw)) or {"ok": 1})
    assert llm.ask("p", system="s", schema=SCHEMA, model="haiku") == {"ok": 1}
    assert calls == [("p", {"system": "s", "schema": SCHEMA, "model": "haiku"})]
