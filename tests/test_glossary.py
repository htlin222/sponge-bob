import hashlib
from pathlib import Path

from psr.glossary import (
    WHISPER_PROMPT_TOKEN_LIMIT,
    Glossary,
    GlossaryEntry,
    _estimate_tokens,
    load,
)

# 這些測試驗證的是載入邏輯，所以釘在一份固定的 fixture 上，而不是會隨
# 卡通集數持續擴充的正式 glossary.yml。
LECTURE = str(Path(__file__).parent / "fixtures" / "glossary_lecture.yml")


def _write(tmp_path, content, name="glossary.yml"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_parses_terms_in_file_order():
    g = load(LECTURE)
    assert g.entries[0].correct == "林協霆"
    assert g.entries[0].whisper_hint is True
    assert g.entries[0].wrong[:2] == ("林協廷", "林協庭")
    # 台灣用詞區塊在檔案尾端，順序應保留
    assert g.entries[-1].correct == "型別"


def test_load_defaults_missing_whisper_hint_to_false():
    g = load(LECTURE)
    chatgpt = next(e for e in g.entries if e.correct == "ChatGPT")
    assert chatgpt.whisper_hint is False


def test_content_hash_matches_sha256_of_file_bytes():
    raw = open(LECTURE, "rb").read()
    g = load(LECTURE)
    assert g.content_hash() == hashlib.sha256(raw).hexdigest()


def test_content_hash_changes_when_file_changes(tmp_path):
    p1 = _write(tmp_path, "version: 1\nterms:\n  - correct: \"A\"\n    wrong: [\"a\"]\n", "a.yml")
    p2 = _write(tmp_path, "version: 1\nterms:\n  - correct: \"B\"\n    wrong: [\"b\"]\n", "b.yml")
    assert load(p1).content_hash() != load(p2).content_hash()


def test_polish_hint_format_and_wrong_limited_to_two(tmp_path):
    content = (
        "version: 1\n"
        "terms:\n"
        "  - correct: \"Claude Code\"\n"
        "    wrong: [\"Cloud Code\", \"Cloud code\", \"third one ignored\"]\n"
        "  - correct: \"agent\"\n"
        "    wrong: [\"Agent\"]\n"
    )
    p = _write(tmp_path, content)
    g = load(p)
    assert g.polish_hint() == "Claude Code←Cloud Code/Cloud code；agent←Agent"


def test_polish_hint_skips_entries_without_wrong(tmp_path):
    content = (
        "version: 1\n"
        "terms:\n"
        "  - correct: \"no wrong here\"\n"
        "  - correct: \"has wrong\"\n"
        "    wrong: [\"typo\"]\n"
    )
    p = _write(tmp_path, content)
    g = load(p)
    assert g.polish_hint() == "has wrong←typo"


def test_polish_hint_on_real_glossary_is_full_table():
    g = load(LECTURE)
    hint = g.polish_hint()
    # 41 個條目、每條都有 wrong，應該全部進榜（實測約 720 字元）。
    assert hint.count("；") == 40
    assert 650 <= len(hint) <= 800


def test_whisper_prompt_only_includes_whisper_hint_entries():
    g = load(LECTURE)
    prompt = g.whisper_prompt()
    included = prompt.split("，")
    hinted = [e.correct for e in g.entries if e.whisper_hint]
    # 全部應該都塞得下（真實術語表遠低於 224 tokens），且順序與檔案順序一致。
    assert included == hinted


def test_whisper_prompt_stays_under_token_budget(tmp_path):
    # 用合成的大術語表逼出貪婪填充在上限前停止的行為。
    lines = ["version: 1", "terms:"]
    for i in range(60):
        term = f"專有名詞測試字串編號{i:03d}長長長長長長長長"
        lines.append(f'  - correct: "{term}"')
        lines.append("    whisper_hint: true")
    p = _write(tmp_path, "\n".join(lines) + "\n")
    g = load(p)

    prompt = g.whisper_prompt()
    assert _estimate_tokens(prompt) <= WHISPER_PROMPT_TOKEN_LIMIT

    # 貪婪、依檔案順序：拿到的一定是最前面連續 N 個詞，不是任意子集。
    all_terms = [e.correct for e in g.entries]
    picked = prompt.split("，")
    assert picked == all_terms[: len(picked)]

    # 而且沒有全部塞進去——這份合成表刻意設計得超過 224 tokens。
    assert len(picked) < len(all_terms)


def test_whisper_prompt_greedy_fill_boundary_does_not_overshoot(tmp_path):
    # 手動建構一個「加入下一項就會超過上限」的邊界情境，確認停在邊界前
    # 而不是硬塞進去超標。
    entries = [
        GlossaryEntry(correct="寬" * 80, wrong=(), whisper_hint=True, note=None),
        GlossaryEntry(correct="再寬一點" * 20, wrong=(), whisper_hint=True, note=None),
    ]
    g = Glossary(entries=tuple(entries), raw_bytes=b"")
    prompt = g.whisper_prompt()
    assert _estimate_tokens(prompt) <= WHISPER_PROMPT_TOKEN_LIMIT
    # 第一項本身（80 個全形字 = 160 tokens）放得下，第二項會超過，應該只留第一項。
    assert prompt == "寬" * 80


def test_whisper_prompt_empty_when_no_hints(tmp_path):
    content = 'version: 1\nterms:\n  - correct: "x"\n    wrong: ["y"]\n'
    p = _write(tmp_path, content)
    g = load(p)
    assert g.whisper_prompt() == ""


def test_production_glossary_fits_whisper_limit_without_truncation():
    # 正式術語表的每個 hint 都該送進 Whisper：一旦超過上限被貪婪截斷，
    # 尾端的角色名會默默失效，這裡要先擋下來。
    g = load(str(Path(__file__).parent.parent / "glossary.yml"))
    hints = [e.correct for e in g.entries if e.whisper_hint]
    assert g.whisper_prompt() == "，".join(hints)
    assert _estimate_tokens(g.whisper_prompt()) <= WHISPER_PROMPT_TOKEN_LIMIT
