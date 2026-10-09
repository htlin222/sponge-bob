from psr.models import Cue


def format_timestamp(seconds: float) -> str:
    """把秒數轉成 SRT 時間碼格式 HH:MM:SS,mmm。"""
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render(cues: list[Cue]) -> str:
    """把 Cue 清單渲染成 SRT 檔案內容。每個區塊之間用一個空行分隔，
    檔案結尾恰好一個換行符（沒有多餘的結尾空白行）。"""
    if not cues:
        return ""
    blocks = [
        f"{cue.index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
        for cue in cues
    ]
    return "\n\n".join(blocks) + "\n"


def _parse_timestamp(ts: str) -> float:
    """把 "HH:MM:SS,mmm" 轉回秒數。先算出總毫秒數（整數運算），
    最後只除一次 1000——如果分開算 h*3600 + m*60 + s + ms/1000，
    浮點加法的結合順序不同會產生極小的誤差（例如 2.369 vs
    2.3689999999999998），足以讓 round-trip 測試在位元級比對時失敗。
    這跟 format_timestamp 用整數毫秒運算是同一個原則的兩面。
    """
    hms, _, ms = ts.partition(",")
    h, m, s = hms.split(":")
    total_ms = int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)
    return total_ms / 1000


def parse(text: str) -> list[Cue]:
    """把 SRT 檔案內容解析回 Cue 清單，是 render() 的逆運算。"""
    blocks = text.strip("\n").split("\n\n")
    cues = []
    for block in blocks:
        if not block.strip():
            continue
        lines = block.split("\n")
        index = int(lines[0].strip())
        start_str, _, end_str = lines[1].partition(" --> ")
        start = _parse_timestamp(start_str.strip())
        end = _parse_timestamp(end_str.strip())
        cue_text = "\n".join(lines[2:])
        cues.append(Cue(index=index, start=start, end=end, text=cue_text))
    return cues
