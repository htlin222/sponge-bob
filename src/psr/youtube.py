import re
import unicodedata

# YouTube 影片 ID 固定 11 碼，字元集是 base64url。
_ID = r"[A-Za-z0-9_-]{11}"

_URL_PATTERNS = [
    re.compile(rf"youtube\.com/watch\?(?:[^&\s]*&)*v=({_ID})"),
    re.compile(rf"youtu\.be/({_ID})"),
    re.compile(rf"youtube\.com/embed/({_ID})"),
    re.compile(rf"youtube\.com/shorts/({_ID})"),
    re.compile(rf"youtube\.com/live/({_ID})"),
]


class ParseError(ValueError):
    """issue 內文解析失敗。訊息會原樣貼回 issue 留言，所以要寫得像給人看的。"""


def parse_youtube_id(text: str) -> str:
    """從一段文字裡取出唯一的 YouTube 影片 ID。

    找到 0 個或 2 個以上都直接失敗，**絕不「取第一個」**——猜錯的代價是花
    十幾分鐘轉錄了錯的影片，而且要看完才會發現。這與 Drive 檔案 ID 的解析
    採同一個原則（設計文件 §12）。
    """
    found: list[str] = []
    for pattern in _URL_PATTERNS:
        for match in pattern.finditer(text):
            if match.group(1) not in found:
                found.append(match.group(1))

    if not found:
        raise ParseError("內文裡找不到 YouTube 連結。請貼上完整網址，例如 https://www.youtube.com/watch?v=xxxxxxxxxxx")
    if len(found) > 1:
        raise ParseError(f"內文裡有 {len(found)} 個不同的 YouTube 連結（{', '.join(found)}）。一個 issue 只處理一支影片，請分開開。")
    return found[0]


# 這些字元在 Drive 或路徑處理上會惹麻煩，一律剔除。
_UNSAFE = set('/\\:*?"<>|\n\r\t')


def slugify(title: str, max_len: int = 80) -> str:
    """把影片標題轉成資料夾名。**保留中文**，只去掉符號。

    不轉拼音的理由：這個資料夾會出現在使用者自己的 Drive 裡，看得懂比
    ASCII 安全重要。「提示詞（Prompt）設計與實作技巧」轉成
    tishici-prompt-... 之後，在 Drive 列表裡根本認不出是哪一支。

    規則：NFKC 正規化（全形英數轉半形）→ 剔除標點、符號與路徑不安全字元
    → 空白收斂成單一連字號 → 去除頭尾連字號 → 截斷至 max_len。
    """
    text = unicodedata.normalize("NFKC", title)

    kept: list[str] = []
    for ch in text:
        if ch in _UNSAFE:
            kept.append(" ")
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S") or category.startswith("C"):
            kept.append(" ")            # 標點、符號、控制字元 → 視為分隔
        elif category.startswith("Z") or ch.isspace():
            kept.append(" ")
        else:
            kept.append(ch)

    slug = re.sub(r"\s+", "-", "".join(kept).strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "untitled"


def drive_paths(slug: str) -> dict[str, str]:
    """一支影片在 Drive 裡的完整檔名組合。

    資料夾 videos/<slug>/ 底下所有產物共用同一個 slug 前綴，這樣在 Drive
    列表裡一眼就看得出誰跟誰是一組，也讓 §11 的 find-or-update 查詢有穩定
    的檔名可以比對。
    """
    return {
        "folder": f"videos/{slug}",
        "video": f"{slug}.raw.mp4",
        "audio": f"{slug}.16k.mp3",
        "words": f"{slug}.words.json",
        "raw_srt": f"{slug}.raw.srt",
        "srt": f"{slug}.zh-Hant.srt",
        "manifest": f"{slug}.manifest.json",
    }
