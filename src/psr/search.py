"""`bob ask` 的混合檢索：原話加改寫的情境句，每句都跑語意向量和字面比對，RRF 合併排名。

`bob eval`（593 題 known-item）實測，前 10 名命中率：
只用向量 16.0% → 加字面比對 20.6% → 再加三句改寫（向量＋字面）27.3%；抽象主題 11.5% → 25.5%。
改寫在對話裡由 Claude 自己寫（`bob ask 原話 情境1 情境2`），命令列可用 `--rewrite` 交給模型。
"""

from __future__ import annotations

from psr import lexical, llm, quotedb, semantic

CANDIDATES = 100  # 每份排名取前幾名參與合併
FUSE_K = 10  # RRF 常數；實測 10 比常見的 60 好，排名前段的權重較重

REWRITE_SYSTEM = """你在幫《海綿寶寶》台詞搜尋工具改寫查詢。台詞庫每段台詞都附一句「使用情境」，描述現實生活中什麼時候適合引用它，
句型像「朋友怕得不敢嘗試時，用來鼓勵他」「工作忙到累壞、想放假時」「被人誤會、想解釋清楚時」。
搜尋是拿查詢和使用情境比語意，所以請把使用者的查詢改寫成 3 句不同角度的使用情境（各 12–35 字）：
- 抽象詞要落到具體生活場景（「離別」→「朋友要搬走了，捨不得跟他道別時」）。
- 時事、科技、職場名詞要換成劇中可能發生的對應處境（「AI 取代工作」→「擔心自己被機器取代、丟了工作時」）。
- 三句盡量涵蓋不同的可能解讀；不要提海綿寶寶角色名。"""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"situations": {"type": "array", "items": {"type": "string"}}},
    "required": ["situations"],
}


def rewrite(query: str, model: str) -> list[str]:
    raw = llm.ask(query, system=REWRITE_SYSTEM, schema=REWRITE_SCHEMA, model=model)
    return [s.strip() for s in raw.get("situations", []) if s.strip()][:3]


REWRITE_MANY_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "situations": {"type": "array", "items": {"type": "string"}}},
        "required": ["i", "situations"],
    }}},
    "required": ["items"],
}


def rewrite_many(queries: list[str], model: str) -> dict[int, list[str]]:
    """一次改寫多句（評測用，省呼叫次數）：輸入每行「編號|查詢」，回傳 {編號: 情境句}。"""
    raw = llm.ask("\n".join(f"{i}|{q}" for i, q in enumerate(queries)),
                  system=REWRITE_SYSTEM + "\n\n輸入每行「編號|查詢」，每個編號各回 3 句。",
                  schema=REWRITE_MANY_SCHEMA, model=model)
    out = {}
    for item in raw.get("items", []):
        i = item.get("i")
        if isinstance(i, int) and 0 <= i < len(queries):
            out[i] = [s.strip() for s in item.get("situations", []) if s.strip()][:3]
    return out


def lexical_index(db) -> lexical.Bm25:
    return lexical.Bm25(
        {qid: f"{text} {use_when}" for qid, text, use_when, _ in quotedb.quote_docs(db)}
    )


def rank(
    db,
    queries: list[str],
    limit: int,
    *,
    index: lexical.Bm25 | None = None,
    embed=semantic.embed_query,
) -> list[int]:
    """每句查詢各產生一份向量排名與一份字面排名，合併後回傳前 limit 個 quote id。"""
    index = index or lexical_index(db)
    rankings = []
    for q in queries:
        rankings.append(
            [h["id"] for h in quotedb.nearest(db, embed(q), limit=CANDIDATES)]
        )
        rankings.append(index.search(q, CANDIDATES))
    return lexical.rrf(rankings, k=FUSE_K)[:limit]


def search(db, queries: list[str], limit: int, **kw) -> list[dict]:
    return quotedb.quotes_by_id(db, rank(db, queries, limit, **kw))
