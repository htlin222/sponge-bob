"""從 SpongeBob Fandom wiki 抓英文原版台詞，給只看中文判斷不了的誤聽當劇情參考。

Whisper 把「逃獄時刻」聽成「桃雨時刻」，只讀中文前後文猜不出來；英文原版寫著
"jailbreak time"，就知道在講什麼。我們的集數編號（ep266）就是 wiki 的製作編號
（Prod），檔名裡的單元順序也跟 a／b 一致，所以對照不必靠模型。

英文台詞跟中文字幕一樣有版權：只存在 Turso，不進 repo。
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request

API = "https://spongebob.fandom.com/api.php"
LIST_PAGE = "List_of_episodes"
_UA = "sponge-bob subtitle proofreader (personal use)"

_ROW = re.compile(
    r'<b>(\d+[a-z]?(?:-\d+)?)</b>\s*</td>\s*<td[^>]*>"<a href="[^"]*" title="([^"]+)"'
)
_LINE = re.compile(r"^\{\{L\|(.*)\}\}\s*$")
_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")


def parse_episode_list(page_html: str) -> list[tuple[str, str]]:
    """「List of episodes」渲染後的 HTML → [(製作編號, 英文集名)]，依頁面順序。"""
    return [(prod, html.unescape(name)) for prod, name in _ROW.findall(page_html)]


def episode_names(number: int, listing: list[tuple[str, str]]) -> list[str]:
    """製作編號 266 → 266a、266b… 的集名；254-255 這種上下集兩個編號都對得到。"""
    out = []
    for prod, name in listing:
        lo, _, hi = prod.partition("-")
        lo_n, hi_n = (
            int(re.match(r"\d+", lo).group()),
            int(hi or re.match(r"\d+", lo).group()),
        )
        if lo_n <= number <= hi_n and name not in out:
            out.append(name)
    return out


def parse_transcript(wikitext: str) -> str:
    """transcript 頁的 {{L|說話者|台詞}}／{{L|[動作]}} → 「說話者: 台詞」純文字。"""
    lines = []
    for raw in wikitext.splitlines():
        m = _LINE.match(raw.strip())
        if not m:
            continue
        body = _LINK.sub(r"\1", m.group(1)).replace("''", "")
        speaker, sep, said = body.partition("|")
        lines.append(f"{speaker}: {said}" if sep else speaker)
    return "\n".join(lines)


def _api(**params) -> dict:
    url = (
        API
        + "?"
        + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_listing() -> list[tuple[str, str]]:
    return parse_episode_list(
        _api(action="parse", page=LIST_PAGE, prop="text")["parse"]["text"]
    )


def fetch_transcript(name: str) -> str | None:
    data = _api(action="parse", page=f"{name}/transcript", prop="wikitext", redirects=1)
    if "error" in data:
        return None
    return parse_transcript(data["parse"]["wikitext"]) or None
