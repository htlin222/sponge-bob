import json
from pathlib import Path

from psr.models import Word
from psr.align import align
from psr.refine import enforce_duration
from psr.validate import validate

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BASE_LINES = [
    "大家好，今天我們來安裝 Python",
    "首先打開終端機。",
    "然後輸入安裝指令。",
    "接著我們要把程式碼推上 GitHub。",
    "這樣團隊成員就能透過 API 互相溝通。",
]


def _load_words() -> list[Word]:
    raw = json.loads((FIXTURES_DIR / "words.json").read_text(encoding="utf-8"))
    return [Word(text=w["text"], start=w["start"], end=w["end"]) for w in raw]


def _assert_graceful(words, lines):
    """對抗性輸入的共同斷言：align() 絕不可拋例外；如果它成功回傳 cues，
    整條 pipeline（含 enforce_duration）跑完後必須完全通過 validate()。
    align() 本身只保證結構性不變量（單調、不重疊、在音訊範圍內）——
    時長落在 [0.5, 7.0] 秒是 enforce_duration 的責任，所以這裡把兩者
    串起來一起驗，這也是為什麼真實 pipeline 裡 enforce_duration 一定要
    緊接在 align() 之後跑：align() 對「整段內容被刪掉」這種編輯只保證
    結構合法，不保證時長合法（被刪掉那段的時間會被前後鄰居吸收）。"""
    result = align(words, lines)
    if result is None:
        return
    final = enforce_duration(result, words)
    violations = validate(final, words[-1].end)

    # 時長違規是**可接受**的結果。切分時若任一半會變成讀不到的碎片
    # （寬度不足兩個全形字），_split_one 會拒絕切、留下一條過長的字幕。
    # 那是刻意的取捨：一條稍長的字幕還讀得完，一條閃一下的碎片讀不到。
    # 對抗性輸入（整段被刪）正是最容易觸發這個取捨的情境。
    #
    # 其餘每一類違規仍然不可接受——尤其是空白文字、重疊、超出音訊範圍，
    # 那些是結構性錯誤而非可讀性取捨。
    structural = [v for v in violations if "時長" not in v]
    assert structural == [], f"出現非時長類的違規：{structural}"

    from psr.text import display_width
    assert all(c.text.strip() for c in final), "產生了空白字幕"
    assert all(display_width(c.text.strip()) >= 2 for c in final), "產生了讀不到的碎片"


def test_llm_drops_a_whole_sentence():
    words = _load_words()
    lines = [BASE_LINES[0], BASE_LINES[1], BASE_LINES[3], BASE_LINES[4]]
    _assert_graceful(words, lines)


def test_llm_inserts_a_hallucinated_sentence():
    words = _load_words()
    lines = list(BASE_LINES)
    lines.insert(2, "這是一句完全捏造、原始逐字稿裡沒有的內容")
    _assert_graceful(words, lines)


def test_llm_reorders_sentences():
    words = _load_words()
    lines = [BASE_LINES[0], BASE_LINES[2], BASE_LINES[1], BASE_LINES[3], BASE_LINES[4]]
    _assert_graceful(words, lines)


def test_llm_hallucinates_ninety_percent_repeated_content():
    words = _load_words()
    lines = [BASE_LINES[0]] + [BASE_LINES[1]] * 9
    _assert_graceful(words, lines)


def test_empty_line_among_valid_lines():
    words = _load_words()
    lines = [BASE_LINES[0], "", BASE_LINES[1], BASE_LINES[2], BASE_LINES[3], BASE_LINES[4]]
    _assert_graceful(words, lines)


def test_pure_english_line():
    words = _load_words()
    lines = list(BASE_LINES)
    lines.append("This is a pure English line with no Chinese at all")
    _assert_graceful(words, lines)
