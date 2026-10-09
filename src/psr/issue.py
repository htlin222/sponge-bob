"""嚴格解析 issue 內文，取出唯一的影片來源（設計文件 §1.5, §12）。

issue 內文來自公開 repo，視為 hostile input：只用正則表達式解析，絕不
eval、絕不把內文內插進 shell 指令、絕不用它未經檢查地組出路徑。

**Fail loudly, never guess.** 內文必須恰好指向一支影片。0 個或 2 個以上
*不同*的來源——包含「一個 Drive 連結 + 一個 YouTube 連結」這種混合情況——
都直接失敗。同一支影片用兩種形式表達（例如 youtu.be 連結與 watch?v= 連結
指向同一個 ID）視為一個，不算衝突。猜錯的代價是花十幾分鐘轉錄了錯的影片，
而且要看完才會發現。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from psr.youtube import ParseError, parse_youtube_id

__all__ = ["Source", "parse_issue", "ParseError"]

# Drive 檔案 ID 沒有固定長度，但官方分享連結裡的 ID 一律由這個字元集組成，
# 以已知的分隔字元（/ ? & 空白 結尾）為界。
_DRIVE_ID_CHARS = r"[A-Za-z0-9_-]"

_DRIVE_URL_PATTERNS = [
    # https://drive.google.com/file/d/{ID}/view...
    re.compile(rf"drive\.google\.com/file/d/({_DRIVE_ID_CHARS}+)"),
    # https://drive.google.com/open?id={ID}
    re.compile(rf"drive\.google\.com/open\?(?:[^&\s]*&)*id=({_DRIVE_ID_CHARS}+)"),
    # https://drive.google.com/uc?id={ID}
    re.compile(rf"drive\.google\.com/uc\?(?:[^&\s]*&)*id=({_DRIVE_ID_CHARS}+)"),
]

# 裸 ID：至少 25 碼的 [A-Za-z0-9_-]，前後不能緊接同一字元集（避免截出更長
# token 的一部分）。
_BARE_DRIVE_ID = re.compile(rf"(?<!{_DRIVE_ID_CHARS}){_DRIVE_ID_CHARS}{{25,}}(?!{_DRIVE_ID_CHARS})")


@dataclass(frozen=True)
class Source:
    """issue 內文解析出的唯一影片來源。"""

    kind: Literal["drive", "youtube"]
    id: str


def _find_drive_ids(text: str) -> list[str]:
    """依序找出內文裡所有不重複的 Drive 檔案 ID（URL 形式與裸 ID 都算）。"""
    found: list[str] = []
    for pattern in [*_DRIVE_URL_PATTERNS, _BARE_DRIVE_ID]:
        for match in pattern.finditer(text):
            drive_id = match.group(1) if pattern.groups else match.group(0)
            if drive_id not in found:
                found.append(drive_id)
    return found


def _find_youtube_id(text: str) -> str | None:
    """借用 `psr.youtube.parse_youtube_id` 取出唯一的 YouTube 影片 ID。

    內文裡完全沒有 YouTube 連結時回傳 None（讓上層跟 Drive 的結果合併判
    斷）；內文裡有 2 個以上*不同*的 YouTube 連結時，直接讓底層的
    ParseError 往上炸——那本身已經是「多支不同影片」的正確失敗。
    """
    try:
        return parse_youtube_id(text)
    except ParseError as exc:
        if "找不到" in str(exc):
            return None
        raise


def parse_issue(body: str) -> Source:
    """從 issue 內文解析出唯一的影片來源，找不到剛好一個就直接失敗。"""
    drive_ids = _find_drive_ids(body)
    youtube_id = _find_youtube_id(body)

    candidates: list[Source] = [Source(kind="drive", id=i) for i in drive_ids]
    if youtube_id is not None:
        candidates.append(Source(kind="youtube", id=youtube_id))

    if not candidates:
        raise ParseError(
            "內文裡找不到影片來源。請貼上 Drive 分享連結、YouTube 網址，"
            "或 25 碼以上的 Drive 檔案 ID。"
        )
    if len(candidates) > 1:
        listing = ", ".join(f"{c.kind}:{c.id}" for c in candidates)
        raise ParseError(
            f"內文裡引用了 {len(candidates)} 支不同的影片（{listing}）。"
            "一個 issue 只處理一支影片，請分開開。"
        )
    return candidates[0]
