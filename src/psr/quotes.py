"""金句索引的純邏輯：分類表、給 Haiku 的 prompt 與 schema、驗證它的回答。

設計原則跟字幕 pipeline 的「LLM 只插標點」一樣：**Haiku 只回字幕編號，
不回台詞**。金句文字一律從 SRT 原文按編號切出來，Haiku 無論怎麼幻覺，
資料庫裡的台詞都是字幕上真的有的字。

本模組不碰網路、不碰資料庫，全部可單元測試。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from psr.models import Cue

# 改 prompt 或 schema 時遞增；連同分類表內容一起決定 index 版本。
PROMPT_VERSION = "2"  # 2：說話者寫真名（不再出現字面的「A／B」）、段落不跨到不相干的下一句
# 低於這個分數的段落不存（使用者決定：2 分以下不存）。
MIN_QUOTABILITY = 3
# 一段金句最多跨幾條字幕。再長就不是「一句話」，是一場戲。
MAX_SPAN = 8
FACETS = ("situations", "emotions", "functions", "characters")
ISSUE_TYPES = ("錯字", "人名", "幻覺", "斷句", "歌詞", "其他")

_EPISODE_NAME = re.compile(r"_S(\d+)_ep(\d+)_(.+?)(?:\.zh-Hant\.srt)?$")


class TaxonomyError(ValueError):
    """分類表格式錯誤。"""


@dataclass(frozen=True)
class Taxonomy:
    facets: dict[str, tuple[str, ...]]

    def content_hash(self) -> str:
        blob = json.dumps(self.facets, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_taxonomy(path: str | Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    facets = {}
    for facet in FACETS:
        names = data.get(facet)
        if not names or not all(isinstance(n, str) and n for n in names):
            raise TaxonomyError(f"分類表缺少 {facet}，或其中有空白項目。")
        if len(set(names)) != len(names):
            raise TaxonomyError(f"分類表 {facet} 有重複的標籤。")
        facets[facet] = tuple(names)
    return Taxonomy(facets)


def index_version(taxonomy: Taxonomy) -> str:
    return f"p{PROMPT_VERSION}-t{taxonomy.content_hash()}"


@dataclass(frozen=True)
class EpisodeMeta:
    stem: str
    season: int
    episode: int
    title: str


def parse_episode_name(name: str) -> EpisodeMeta:
    """`海綿寶寶_S07_ep142_集名.zh-Hant.srt` → 季、集、集名。"""
    stem = name.removesuffix(".zh-Hant.srt")
    m = _EPISODE_NAME.search(stem)
    if not m:
        raise ValueError(f"看不懂的檔名：{name}")
    return EpisodeMeta(stem, int(m.group(1)), int(m.group(2)), m.group(3).replace("_", "／"))


SYSTEM_PROMPT = """你是《海綿寶寶》台灣配音版的台詞編輯，替「生活大小事都能引用海綿寶寶」的金句庫做標記。

輸入是一集的字幕，每行格式為「編號|時間|台詞」。字幕是語音辨識產生的，可能有錯字。

任務一：挑出值得引用的金句段落。
- 一段 = 連續的 from..to 編號（含頭尾，最多 8 條），可以是一句話或一來一往的短對話。
- 只回編號，不要抄寫或改寫台詞。
- 只挑脫離劇情也看得懂、能套用到現實生活的段落；片頭曲、歌詞、純劇情交代不要挑。
- 段落要剛好包住那句話或那段對話：從句子開頭開始，說完就結束，不要把前後不相干的台詞一起包進來。
- 寧多勿漏：一集 11 分鐘的故事通常有 10–20 段能用，22 分鐘的有 20–40 段。好引用程度交給 quotability 判斷，不要因為字幕有錯字就跳過。
- quotability：5=經典名台詞、4=很好用、3=特定場合能用、2 以下=勉強。
- use_when：一句話描述「現實生活中什麼時候可以引用」，例如「朋友考砸了需要打氣時」。
- speaker：說話者，寫角色的台灣譯名（海綿寶寶、派大星、章魚哥、蟹老闆、珊迪、皮老闆、泡芙阿姨、珍珍、凱倫，或劇中其他角色的名字）。字幕沒有標示說話者，請從語氣、稱呼、口頭禪與劇情推斷：被叫「章魚哥」的通常是對方在說話；「小子」「錢錢」多半是蟹老闆；喊「我準備好了」的是海綿寶寶。一來一往的對話依發言順序寫成「海綿寶寶／章魚哥」這種格式，絕對不要寫字面上的「A／B」。大多數段落都推斷得出來，真的完全無法判斷才填「不明」。
- 標籤只能從 schema 列出的選項挑，每類 0–3 個。

任務二：審計字幕品質，列出可疑的條目。
- 錯字（同音字誤聽）、人名（角色名聽錯）、幻覺（不像台詞的浮水印、廣告詞）、斷句、歌詞（片頭曲被當成台詞）。
- suggestion 填你認為正確的寫法；沒把握就留空。

任務三：如果現有標籤明顯不夠用，在 new_tags 提出建議（不要為了提而提）。"""


def output_schema(taxonomy: Taxonomy) -> dict:
    tag_lists = {
        facet: {"type": "array", "items": {"enum": list(names)}, "maxItems": 3}
        for facet, names in taxonomy.facets.items()
    }
    quote = {
        "type": "object",
        "properties": {
            "from": {"type": "integer"},
            "to": {"type": "integer"},
            "speaker": {"type": "string"},
            "quotability": {"type": "integer", "minimum": 1, "maximum": 5},
            "use_when": {"type": "string"},
            **tag_lists,
        },
        "required": ["from", "to", "speaker", "quotability", "use_when", *FACETS],
    }
    issue = {
        "type": "object",
        "properties": {
            "cue": {"type": "integer"},
            "type": {"enum": list(ISSUE_TYPES)},
            "detail": {"type": "string"},
            "suggestion": {"type": "string"},
        },
        "required": ["cue", "type", "detail", "suggestion"],
    }
    new_tag = {
        "type": "object",
        "properties": {
            "facet": {"enum": list(FACETS)},
            "name": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["facet", "name", "reason"],
    }
    return {
        "type": "object",
        "properties": {
            "quotes": {"type": "array", "items": quote},
            "issues": {"type": "array", "items": issue},
            "new_tags": {"type": "array", "items": new_tag},
        },
        "required": ["quotes", "issues", "new_tags"],
    }


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def user_prompt(meta: EpisodeMeta, cues: list[Cue]) -> str:
    lines = [f"集名：{meta.title}（第 {meta.season} 季第 {meta.episode} 集）", ""]
    lines += [f"{c.index}|{_clock(c.start)}|{c.text.replace(chr(10), ' ')}" for c in cues]
    return "\n".join(lines)


@dataclass(frozen=True)
class Quote:
    cue_from: int
    cue_to: int
    text: str
    start: float
    end: float
    speaker: str
    quotability: int
    use_when: str
    tags: tuple[tuple[str, str], ...]  # (facet, name)


@dataclass(frozen=True)
class Issue:
    cue: int
    type: str
    detail: str
    suggestion: str


@dataclass(frozen=True)
class Indexed:
    quotes: tuple[Quote, ...]
    issues: tuple[Issue, ...]
    new_tags: tuple[tuple[str, str, str], ...]  # (facet, name, reason)
    rejected: int  # 編號不合法、太長或標籤不在分類表內而被丟掉的段落數


def accept(raw: dict, cues: list[Cue], taxonomy: Taxonomy) -> Indexed:
    """驗證 Haiku 的回答，台詞從字幕原文切出。

    schema 已經限制了型別與 enum，但模型輸出不保證完全遵守，這裡再擋一次：
    編號超出範圍、from > to、跨太多條、出現分類表外的標籤，整段丟掉並計數。
    """
    by_index = {c.index: c for c in cues}
    quotes: list[Quote] = []
    rejected = 0
    seen: set[tuple[int, int]] = set()
    for q in raw.get("quotes", []):
        a, b = q.get("from"), q.get("to")
        valid = (isinstance(a, int) and isinstance(b, int) and a <= b
                 and b - a < MAX_SPAN and all(i in by_index for i in range(a, b + 1)))
        tags = tuple((facet, name) for facet in FACETS for name in dict.fromkeys(q.get(facet, [])))
        if not valid or any(name not in taxonomy.facets[facet] for facet, name in tags):
            rejected += 1
            continue
        score = q.get("quotability")
        if not isinstance(score, int) or score < MIN_QUOTABILITY or (a, b) in seen:
            continue
        seen.add((a, b))
        span = [by_index[i] for i in range(a, b + 1)]
        quotes.append(Quote(
            cue_from=a, cue_to=b,
            text="\n".join(c.text for c in span),
            start=span[0].start, end=span[-1].end,
            speaker=(q.get("speaker") or "不明").strip(),
            quotability=score,
            use_when=(q.get("use_when") or "").strip(),
            tags=tags,
        ))
    issues = tuple(
        Issue(i["cue"], i["type"], i.get("detail", ""), i.get("suggestion", ""))
        for i in raw.get("issues", [])
        if i.get("cue") in by_index and i.get("type") in ISSUE_TYPES
    )
    new_tags = tuple(
        (t["facet"], t["name"].strip(), t.get("reason", ""))
        for t in raw.get("new_tags", [])
        if t.get("facet") in FACETS and (t.get("name") or "").strip()
        and t["name"].strip() not in taxonomy.facets[t["facet"]]
    )
    return Indexed(tuple(quotes), issues, new_tags, rejected)
