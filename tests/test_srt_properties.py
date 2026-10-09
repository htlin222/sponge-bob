from hypothesis import given, settings, strategies as st

from psr.models import Cue
from psr.srt import parse, render


@st.composite
def cue_lists(draw):
    """產生毫秒精度的合法 Cue 清單：時間單調遞增、不重疊、文字不含換行。

    刻意用「整數毫秒 / 1000」建構秒數，而不是直接產生任意 float——
    這樣 format_timestamp 內部的 round(seconds * 1000) 才能精確復原成同一個
    整數毫秒，round-trip 才有機會做到位元級相等，而不是「差不多相等」。
    """
    n = draw(st.integers(min_value=1, max_value=5))
    cues = []
    prev_end_ms = 0
    for i in range(n):
        start_ms = prev_end_ms + draw(st.integers(min_value=0, max_value=5000))
        duration_ms = draw(st.integers(min_value=100, max_value=7000))
        end_ms = start_ms + duration_ms
        text = draw(
            st.text(
                alphabet=st.characters(blacklist_categories=("Cc", "Cs")),
                min_size=1,
                max_size=20,
            ).filter(lambda s: "\n" not in s and s.strip() != "")
        )
        cues.append(Cue(index=i + 1, start=start_ms / 1000, end=end_ms / 1000, text=text))
        prev_end_ms = end_ms
    return cues


@given(cue_lists())
@settings(max_examples=200)
def test_parse_render_round_trip(cues):
    assert parse(render(cues)) == cues
