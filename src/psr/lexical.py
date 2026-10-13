"""字面比對：中文雙字詞 BM25，和語意向量用 RRF 合併排名。

向量只嵌入「使用情境」欄，使用者直接打出台詞原文（「有的時候我就是好想逃離那一切」）
或具體名詞（「考試」「飯店」）時反而排不上來；字面比對正好補這一塊。
中文不斷詞，用相鄰兩字當詞：不需要字典，錯字也只壞掉兩個詞。
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.2
B = 0.75
RRF_K = 60


def bigrams(text: str) -> list[str]:
    chars = re.sub(r"[\W_]", "", text)
    if len(chars) < 2:
        return [chars] if chars else []
    return [chars[i : i + 2] for i in range(len(chars) - 1)]


class Bm25:
    def __init__(self, docs: dict[int, str]):
        self.tf = {doc_id: Counter(bigrams(text)) for doc_id, text in docs.items()}
        self.length = {doc_id: sum(c.values()) for doc_id, c in self.tf.items()}
        self.avg = sum(self.length.values()) / len(docs) if docs else 0.0
        df = Counter(term for c in self.tf.values() for term in c)
        n = len(docs)
        self.idf = {
            term: math.log((n - f + 0.5) / (f + 0.5) + 1) for term, f in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for doc_id, c in self.tf.items():
            for term in c:
                self.postings.setdefault(term, []).append(doc_id)

    def search(self, query: str, limit: int) -> list[int]:
        scores: Counter = Counter()
        for term in set(bigrams(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                f = self.tf[doc_id][term]
                norm = K1 * (1 - B + B * self.length[doc_id] / self.avg)
                scores[doc_id] += idf * f * (K1 + 1) / (f + norm)
        return [doc_id for doc_id, _ in scores.most_common(limit)]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion：每份排名的第 r 名得 1/(k+r) 分，加總後排序。"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + r)
    return sorted(scores, key=lambda d: -scores[d])
