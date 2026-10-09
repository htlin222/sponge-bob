from hypothesis import given, settings, strategies as st

from psr.align import align
from psr.models import Word
from psr.segment import raw_segment
from psr.validate import validate

CHAR_POOL = list("我們先來後安裝這是的一二三四五六七八九十你好嗎最近過得如何")


@st.composite
def word_lists(draw):
    """產生一個「單一窗口」的 word 清單：字數、每字時長、字間 gap 都控制在
    小範圍內，讓產生出來的視窗天生就落在驗證器的合法範圍內（寬度 <=20、
    時長 0.5-7.0 秒、閱讀速度 <=9）。"""
    n = draw(st.integers(min_value=5, max_value=10))
    words = []
    t = 0.0
    for _ in range(n):
        char = draw(st.sampled_from(CHAR_POOL))
        duration = draw(st.floats(min_value=0.3, max_value=0.5))
        gap = draw(st.floats(min_value=0.0, max_value=0.05))
        start = round(t + gap, 3)
        end = round(start + duration, 3)
        words.append(Word(text=char, start=start, end=end))
        t = end
    return words


@st.composite
def edited_lines(draw, words):
    """對 words 串接後的文字，在 8% 編輯預算內做隨機插入/刪除/替換，模擬
    潤稿（設計文件 §5 的硬不變量：正規化後編輯距離必須 <=8%）。回傳單一行
    組成的清單（模擬單行窗口）。"""
    original = "".join(w.text for w in words)
    budget = max(1, int(len(original) * 0.08))
    n_edits = draw(st.integers(min_value=0, max_value=budget))
    chars = list(original)
    for _ in range(n_edits):
        if not chars:
            break
        op = draw(st.sampled_from(["insert", "delete", "substitute"]))
        pos = draw(st.integers(min_value=0, max_value=len(chars) - 1))
        if op == "insert":
            chars.insert(pos, draw(st.sampled_from(CHAR_POOL)))
        elif op == "delete":
            del chars[pos]
        else:
            chars[pos] = draw(st.sampled_from(CHAR_POOL))
    return ["".join(chars)]


@given(st.data())
@settings(max_examples=300)
def test_align_result_always_satisfies_validator_invariants(data):
    """不管 align() 吐出什麼，只要它沒有降級成 None，結果就必須是一份
    結構完全合法的字幕：單調不重疊、在音訊範圍內、時長合法。"""
    words = data.draw(word_lists())
    lines = data.draw(edited_lines(words))

    result = align(words, lines)

    if result is None:
        return  # 降級是允許的結果，不是失敗

    audio_duration = words[-1].end
    violations = validate(result, audio_duration)
    assert violations == []


@given(word_lists())
@settings(max_examples=200)
def test_identity_polish_matches_raw_segment(words):
    """Metamorphic test：把 raw_segment 切出來的 cue 文字原封不動地當成
    「潤稿結果」餵回 align()（等於沒有潤稿、純粹重新斷句），產生的時間碼
    應該跟 raw_segment 本身在 1ms 內完全一致。

    這裡刻意重用 word_lists()：它產生的字間 gap 上限（0.05s）遠低於
    raw_segment 預設的 max_gap（0.6s），累積寬度上限也遠低於預設的
    max_width（20.0），所以 raw_segment 在這個策略下必定回傳單一 cue，
    讓這個 metamorphic test 不需要額外處理「多 cue 對齊」的邊界情況。
    """
    raw_cues = raw_segment(words)
    assert len(raw_cues) == 1  # 驗證上面這段推理沒有被之後的策略調整破壞

    lines = [cue.text for cue in raw_cues]
    aligned_cues = align(words, lines)

    assert aligned_cues is not None
    assert len(aligned_cues) == len(raw_cues)
    for aligned, raw in zip(aligned_cues, raw_cues):
        assert abs(aligned.start - raw.start) <= 0.001
        assert abs(aligned.end - raw.end) <= 0.001
