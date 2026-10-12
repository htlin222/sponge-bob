"""語意搜尋：把金句的「使用情境」嵌入成向量，存進 Turso 的 quote_vecs，用處境直接找最近的金句。

模型在本機跑（fastembed／ONNX），免費、離線、不吃 `claude -p` 額度。
"""

from __future__ import annotations

import hashlib
from functools import cache

MODEL = "BAAI/bge-small-zh-v1.5"
# bge 的短查詢對長段落要加這句指令前綴
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def doc(text: str, use_when: str) -> str:
    # 只嵌入使用情境：台詞本身的字面（玉米、罐頭）會把向量拉離處境，實測命中率較差
    return use_when


def digest(document: str) -> str:
    return hashlib.md5(f"{MODEL}\n{document}".encode("utf-8")).hexdigest()


def stale(rows: list[tuple[int, str, str, str | None]]) -> list[tuple[int, str, str]]:
    """`quotedb.quote_docs` 的結果中，還沒有向量或文字已變的 (id, 嵌入文字, digest)。"""
    out = []
    for qid, text, use_when, stored in rows:
        document = doc(text, use_when)
        fresh = digest(document)
        if stored != fresh:
            out.append((qid, document, fresh))
    return out


@cache
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(MODEL)


def embed_docs(documents: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(documents, batch_size=64)]


def embed_query(situation: str) -> list[float]:
    return next(iter(_model().embed([QUERY_PREFIX + situation]))).tolist()
