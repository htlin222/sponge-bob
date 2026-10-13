"""`bob ask` 的檢索評測集：給一段金句出「使用者會怎麼問」，看檢索能不能把它排上來。

出題的模型只看台詞與前後文，不看「使用情境」欄——向量比對的正是那一欄，給它看等於拿答案出題。
同一段台詞出三種問法，因為實際使用者三種都會用：

- colloquial：自己的處境，口語（「朋友說暫時還不敢」）
- abstract：抽象主題詞（「陪伴」「告別」）——已知是語意搜尋的弱點
- topic：時事或具體名詞（「AI」「選舉」）——海底沒有的東西，要靠劇情對應

這是 known-item 評測：只認出題的那一段為正解，別段同樣合適也算沒中，所以分數偏低，
適合比較兩種檢索方式誰好，不是絕對品質。題目由台詞衍生，只存 Turso。
"""

from __future__ import annotations

import random
import re

STYLES = ("colloquial", "abstract", "topic")
KS = (1, 5, 10, 30)
MIN_QUOTABILITY = 3
LEAK_CHARS = 4  # 題目照抄台詞連續四個字以上，就變成字面比對

GEN_SYSTEM = """你在幫《海綿寶寶》台詞搜尋工具出測試題。每段台詞附有前後文，請想像一位使用者現實生活中
遇到某個處境、想找這句台詞來回應或轉貼，他會在搜尋框打什麼。每段寫三種問法：

- colloquial：用第一人稱或「朋友……」描述自己的具體處境，口語、一句話，10–30 字。
- abstract：2–8 字的抽象主題或情緒（例如「自我懷疑」「離別」），不要提劇情。
- topic：現實世界的時事、職場、科技或社會話題，3–15 字（例如「AI 取代工作」「選舉造勢」）；
  台詞若和任何現實話題都扯不上，寫最接近的生活話題。

規則：
- 不要抄台詞裡的字詞，也不要提角色名、集名、海底的專有名詞——使用者不知道答案長什麼樣子。
- 依台詞在劇中的真正語氣出題（反諷就是反諷），不要望文生義。"""

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    **{s: {"type": "string"} for s in STYLES},
                },
                "required": ["id", *STYLES],
            },
        }
    },
    "required": ["items"],
}


def sample(rows: list[tuple[int, int, int]], n: int, seed: int) -> list[int]:
    """(quote_id, episode_id, quotability) → 每集至多一段、★3 以上，隨機但可重現的 n 個 quote_id。"""
    rng = random.Random(seed)
    by_episode: dict[int, list[int]] = {}
    for qid, episode_id, quotability in sorted(rows):
        if quotability >= MIN_QUOTABILITY:
            by_episode.setdefault(episode_id, []).append(qid)
    picks = [rng.choice(ids) for _, ids in sorted(by_episode.items())]
    rng.shuffle(picks)
    return picks[:n]


def gen_prompt(items: list[tuple[int, str, list[str]]]) -> str:
    """[(quote_id, 台詞, 前後文)] → 出題用的 user prompt。"""
    blocks = []
    for qid, text, context in items:
        blocks.append(f"[{qid}]\n前後文：{' '.join(context)}\n台詞：{_flat(text)}")
    return "\n\n".join(blocks)


def _flat(text: str) -> str:
    return text.replace("\n", " ")


def _chars(text: str) -> str:
    return re.sub(r"[\W_]", "", text)


def leaks(query: str, line: str) -> bool:
    q, s = _chars(query), _chars(line)
    return any(q[i : i + LEAK_CHARS] in s for i in range(len(q) - LEAK_CHARS + 1))


def accept(raw: dict, lines: dict[int, str]) -> list[tuple[int, str, str]]:
    """模型輸出 → [(quote_id, style, query)]：只收這批的編號，丟掉空的與抄台詞的題目。"""
    out = []
    for item in raw.get("items", []):
        qid = item.get("id")
        if qid not in lines:
            continue
        for style in STYLES:
            query = (item.get(style) or "").strip()
            if query and not leaks(query, lines[qid]):
                out.append((qid, style, query))
    return out


def rank_of(gold: int, ids: list[int]) -> int | None:
    return ids.index(gold) + 1 if gold in ids else None


def metrics(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict[str, float]:
    n = len(ranks)
    out: dict[str, float] = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = (
            sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
        )
    out["mrr"] = sum(1 / r for r in ranks if r) / n if n else 0.0
    return out
