"""金句資料庫。正式資料在 Turso（libSQL），測試與離線用本機 SQLite，兩者同一份 SQL。

Turso 走 HTTP pipeline API（`/v2/pipeline`），只用標準函式庫，不多一個依賴。
一集的寫入包成一個交易：先刪掉這集的舊資料再整批插入，中途失敗就 rollback，
不會留下半集。

搜尋刻意不用 FTS5：SQLite 預設的斷詞器不會切中文，trigram 又不吃兩個字的詞
（「緊張」「加班」），而全部金句只有一萬筆上下，LIKE 掃一遍就夠快。
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from psr.models import Cue
from psr.quotes import EpisodeMeta, Indexed, Taxonomy

CONFIG_PATH = Path.home() / ".config/sponge-bob/turso.json"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY,
        stem TEXT NOT NULL UNIQUE,
        season INTEGER NOT NULL,
        episode INTEGER NOT NULL,
        title TEXT NOT NULL,
        drive_id TEXT,
        srt_md5 TEXT NOT NULL,
        index_version TEXT NOT NULL,
        indexed_at TEXT NOT NULL,
        rejected INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS cues (
        episode_id INTEGER NOT NULL REFERENCES episodes(id),
        idx INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        PRIMARY KEY (episode_id, idx))""",
    """CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY,
        episode_id INTEGER NOT NULL REFERENCES episodes(id),
        cue_from INTEGER NOT NULL,
        cue_to INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        speaker TEXT NOT NULL,
        quotability INTEGER NOT NULL,
        use_when TEXT NOT NULL,
        UNIQUE (episode_id, cue_from, cue_to))""",
    """CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY,
        facet TEXT NOT NULL,
        name TEXT NOT NULL,
        UNIQUE (facet, name))""",
    """CREATE TABLE IF NOT EXISTS quote_tags (
        quote_id INTEGER NOT NULL REFERENCES quotes(id),
        tag_id INTEGER NOT NULL REFERENCES tags(id),
        PRIMARY KEY (quote_id, tag_id))""",
    """CREATE TABLE IF NOT EXISTS audit_issues (
        id INTEGER PRIMARY KEY,
        episode_id INTEGER NOT NULL REFERENCES episodes(id),
        cue_idx INTEGER NOT NULL,
        type TEXT NOT NULL,
        detail TEXT NOT NULL,
        suggestion TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open')""",
    """CREATE TABLE IF NOT EXISTS tag_suggestions (
        id INTEGER PRIMARY KEY,
        episode_id INTEGER NOT NULL REFERENCES episodes(id),
        facet TEXT NOT NULL,
        name TEXT NOT NULL,
        reason TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS srt_fixes (
        stem TEXT PRIMARY KEY,
        md5_before TEXT NOT NULL,
        md5_after TEXT NOT NULL,
        fix_version TEXT NOT NULL,
        applied INTEGER NOT NULL,
        skipped INTEGER NOT NULL,
        fixed_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS punctuations (
        stem TEXT PRIMARY KEY,
        version TEXT NOT NULL,
        text TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS resegments (
        stem TEXT PRIMARY KEY,
        version TEXT NOT NULL,
        md5 TEXT NOT NULL,
        cues INTEGER NOT NULL,
        failed_chunks INTEGER NOT NULL,
        violations INTEGER NOT NULL,
        done_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS fix_log (
        stem TEXT NOT NULL,
        cue_idx INTEGER NOT NULL,
        heard TEXT NOT NULL,
        fixed TEXT NOT NULL,
        applied INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS fix_log_stem ON fix_log(stem)",
    "CREATE INDEX IF NOT EXISTS quotes_episode ON quotes(episode_id)",
    "CREATE INDEX IF NOT EXISTS quote_tags_tag ON quote_tags(tag_id)",
    "CREATE INDEX IF NOT EXISTS audit_episode ON audit_issues(episode_id)",
]

_EPISODE_ID = "(SELECT id FROM episodes WHERE stem = ?)"


class SqliteDb:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)

    def execute(self, sql: str, args=()) -> list[tuple]:
        return self.conn.execute(sql, tuple(args)).fetchall()

    def transaction(self, stmts: list[tuple[str, tuple]]) -> None:
        with self.conn:
            for sql, args in stmts:
                self.conn.execute(sql, args)


def _encode(v) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        raise TypeError("布林值請先轉成 0/1")
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _decode(v: dict):
    kind = v.get("type")
    if kind == "null":
        return None
    if kind == "integer":
        return int(v["value"])
    if kind == "float":
        return float(v["value"])
    if kind == "blob":
        return base64.b64decode(v["base64"])
    return v["value"]


def _stmt(sql: str, args=()) -> dict:
    return {"sql": sql, "args": [_encode(a) for a in args]}


class TursoError(RuntimeError):
    """Turso 回傳錯誤。"""


class TursoDb:
    def __init__(self, url: str, token: str, timeout: int = 60):
        self.endpoint = url.replace("libsql://", "https://", 1).rstrip("/") + "/v2/pipeline"
        self.token = token
        self.timeout = timeout

    def _pipeline(self, requests: list[dict]) -> list[dict]:
        body = json.dumps({"requests": [*requests, {"type": "close"}]}).encode()
        req = urllib.request.Request(self.endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            results = json.load(resp)["results"][:-1]
        for r in results:
            if r["type"] == "error":
                raise TursoError(r["error"].get("message", r["error"]))
        return results

    def execute(self, sql: str, args=()) -> list[tuple]:
        (res,) = self._pipeline([{"type": "execute", "stmt": _stmt(sql, args)}])
        rows = res["response"]["result"]["rows"]
        return [tuple(_decode(v) for v in row) for row in rows]

    def transaction(self, stmts: list[tuple[str, tuple]]) -> None:
        # Hrana 的 batch：每一步都以前一步成功為條件，最後 COMMIT；
        # 任何一步失敗，COMMIT 不會執行，改跑 ROLLBACK。
        steps = [{"stmt": _stmt("BEGIN")}]
        for sql, args in stmts:
            steps.append({"stmt": _stmt(sql, args), "condition": {"type": "ok", "step": len(steps) - 1}})
        commit = len(steps)
        steps.append({"stmt": _stmt("COMMIT"), "condition": {"type": "ok", "step": commit - 1}})
        steps.append({"stmt": _stmt("ROLLBACK"),
                      "condition": {"type": "not", "cond": {"type": "ok", "step": commit}}})
        (res,) = self._pipeline([{"type": "batch", "batch": {"steps": steps}}])
        errors = [e for e in res["response"]["result"]["step_errors"] if e]
        if errors:
            raise TursoError(errors[0].get("message", errors[0]))


def connect(local_path: str | None = None):
    """`local_path` 給了就用本機 SQLite，否則連 Turso（環境變數優先，其次設定檔）。"""
    if local_path:
        return SqliteDb(local_path)
    url, token = os.environ.get("TURSO_DATABASE_URL"), os.environ.get("TURSO_AUTH_TOKEN")
    if not (url and token) and CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        url, token = cfg["url"], cfg["token"]
    if not (url and token):
        raise TursoError(f"找不到 Turso 連線資訊：設定 TURSO_DATABASE_URL/TURSO_AUTH_TOKEN 或 {CONFIG_PATH}")
    return TursoDb(url, token)


def init(db, taxonomy: Taxonomy) -> None:
    db.transaction([(sql, ()) for sql in SCHEMA] + [
        ("INSERT OR IGNORE INTO tags (facet, name) VALUES (?, ?)", (facet, name))
        for facet, names in taxonomy.facets.items() for name in names
    ])


def indexed_versions(db) -> dict[str, tuple[str, str]]:
    """stem → (srt_md5, index_version)，用來判斷哪些集數要（重新）標記。"""
    return {stem: (md5, ver) for stem, md5, ver in
            db.execute("SELECT stem, srt_md5, index_version FROM episodes")}


def write_episode(db, meta: EpisodeMeta, *, drive_id: str, srt_md5: str, version: str,
                  cues: list[Cue], result: Indexed) -> None:
    stem = meta.stem
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stmts: list[tuple[str, tuple]] = [
        ("""INSERT INTO episodes (stem, season, episode, title, drive_id, srt_md5,
                                  index_version, indexed_at, rejected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stem) DO UPDATE SET season = excluded.season,
                episode = excluded.episode, title = excluded.title,
                drive_id = excluded.drive_id, srt_md5 = excluded.srt_md5,
                index_version = excluded.index_version,
                indexed_at = excluded.indexed_at, rejected = excluded.rejected""",
         (stem, meta.season, meta.episode, meta.title, drive_id, srt_md5, version, now,
          result.rejected)),
        (f"DELETE FROM quote_tags WHERE quote_id IN "
         f"(SELECT id FROM quotes WHERE episode_id = {_EPISODE_ID})", (stem,)),
    ]
    stmts += [(f"DELETE FROM {table} WHERE episode_id = {_EPISODE_ID}", (stem,))
              for table in ("quotes", "cues", "audit_issues", "tag_suggestions")]
    stmts += [(f"INSERT INTO cues (episode_id, idx, start_s, end_s, text) VALUES ({_EPISODE_ID}, ?, ?, ?, ?)",
               (stem, c.index, c.start, c.end, c.text)) for c in cues]
    for q in result.quotes:
        stmts.append((
            f"""INSERT INTO quotes (episode_id, cue_from, cue_to, start_s, end_s, text, speaker,
                                    quotability, use_when)
                VALUES ({_EPISODE_ID}, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (stem, q.cue_from, q.cue_to, q.start, q.end, q.text, q.speaker, q.quotability,
             q.use_when)))
        stmts += [(
            f"""INSERT INTO quote_tags (quote_id, tag_id)
                SELECT q.id, t.id FROM quotes q, tags t
                WHERE q.episode_id = {_EPISODE_ID} AND q.cue_from = ? AND q.cue_to = ?
                  AND t.facet = ? AND t.name = ?""",
            (stem, q.cue_from, q.cue_to, facet, name)) for facet, name in q.tags]
    stmts += [(f"""INSERT INTO audit_issues (episode_id, cue_idx, type, detail, suggestion)
                   VALUES ({_EPISODE_ID}, ?, ?, ?, ?)""",
               (stem, i.cue, i.type, i.detail, i.suggestion)) for i in result.issues]
    stmts += [(f"INSERT INTO tag_suggestions (episode_id, facet, name, reason) VALUES ({_EPISODE_ID}, ?, ?, ?)",
               (stem, facet, name, reason)) for facet, name, reason in result.new_tags]
    db.transaction(stmts)


def cue_timings(db, stem: str) -> list[tuple[int, float, float]]:
    """該集已入庫字幕的 (編號, 起, 訖)，用來判斷新字幕是否只改了字。"""
    return [tuple(r) for r in db.execute(
        f"SELECT idx, start_s, end_s FROM cues WHERE episode_id = {_EPISODE_ID} ORDER BY idx", (stem,))]


def refresh_text(db, stem: str, *, srt_md5: str, cues: list[Cue]) -> None:
    """字幕只改了字（編號與時間都沒變）時，就地更新台詞文字，保留標記結果。

    金句只記 cue 編號範圍，文字依編號重新切出即可，不必再花一次模型額度重標。
    """
    text = {c.index: c.text for c in cues}
    spans = db.execute(f"SELECT id, cue_from, cue_to FROM quotes WHERE episode_id = {_EPISODE_ID}", (stem,))
    stmts = [(f"UPDATE cues SET text = ? WHERE episode_id = {_EPISODE_ID} AND idx = ?", (c.text, stem, c.index))
             for c in cues]
    stmts += [("UPDATE quotes SET text = ? WHERE id = ?",
               ("\n".join(text[i] for i in range(a, b + 1)), qid)) for qid, a, b in spans]
    stmts.append(("UPDATE episodes SET srt_md5 = ? WHERE stem = ?", (srt_md5, stem)))
    db.transaction(stmts)


def fix_state(db) -> dict[str, tuple[str, str]]:
    """stem → (修正後 md5, fix 版本)。Drive 上的 md5 等於修正後 md5 代表檔案仍是我們的輸出。"""
    return {stem: (md5, ver) for stem, md5, ver in
            db.execute("SELECT stem, md5_after, fix_version FROM srt_fixes")}


def cached_punctuation(db, stem: str, version: str) -> str | None:
    """模型加好標點的全文。斷句規則改了可以直接重建，不必再問一次模型。"""
    rows = db.execute("SELECT text FROM punctuations WHERE stem = ? AND version = ?", (stem, version))
    return rows[0][0] if rows else None


def cache_punctuation(db, stem: str, version: str, text: str) -> None:
    db.transaction([("""INSERT INTO punctuations (stem, version, text) VALUES (?, ?, ?)
        ON CONFLICT (stem) DO UPDATE SET version = excluded.version, text = excluded.text""",
                     (stem, version, text))])


def reseg_state(db) -> dict[str, str]:
    """stem → 重新斷句的版本。"""
    return dict(db.execute("SELECT stem, version FROM resegments"))


def record_reseg(db, stem: str, *, version: str, md5: str, cues: int, failed_chunks: int,
                 violations: int) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.transaction([("""INSERT INTO resegments (stem, version, md5, cues, failed_chunks, violations, done_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (stem) DO UPDATE SET version = excluded.version, md5 = excluded.md5,
            cues = excluded.cues, failed_chunks = excluded.failed_chunks,
            violations = excluded.violations, done_at = excluded.done_at""",
                     (stem, version, md5, cues, failed_chunks, violations, now))])


def record_fix(db, stem: str, *, md5_before: str, md5_after: str, version: str,
               applied: list, skipped: list) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stmts = [
        ("""INSERT INTO srt_fixes (stem, md5_before, md5_after, fix_version, applied, skipped, fixed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stem) DO UPDATE SET md5_before = excluded.md5_before,
                md5_after = excluded.md5_after, fix_version = excluded.fix_version,
                applied = excluded.applied, skipped = excluded.skipped, fixed_at = excluded.fixed_at""",
         (stem, md5_before, md5_after, version, len(applied), len(skipped), now)),
        ("DELETE FROM fix_log WHERE stem = ?", (stem,)),
    ]
    stmts += [("INSERT INTO fix_log (stem, cue_idx, heard, fixed, applied) VALUES (?, ?, ?, ?, ?)",
               (stem, f.cue if isinstance(f.cue, int) else -1, f.wrong, f.right, flag))
              for flag, fixes in ((1, applied), (0, skipped)) for f in fixes]
    db.transaction(stmts)


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


SEARCH_COLUMNS = ("id", "season", "episode", "title", "start", "end", "text", "speaker",
                  "quotability", "use_when", "tags", "score")


def search(db, *, tags: list[tuple[str, str]], keywords: list[str], limit: int = 30) -> list[dict]:
    """依標籤命中數（每個 2 分）與關鍵字命中數（台詞 3 分、使用情境 2 分）排序。

    同分時好引用程度高的優先。完全沒命中的不回傳。
    """
    tag_keys = [f"{facet}:{name}" for facet, name in tags]
    keywords = [k.strip() for k in keywords if k.strip()]
    parts, args = ["0"], []
    if tag_keys:
        marks = ", ".join("?" * len(tag_keys))
        parts.append(f"""2 * (SELECT count(*) FROM quote_tags qt JOIN tags t ON t.id = qt.tag_id
                              WHERE qt.quote_id = q.id AND t.facet || ':' || t.name IN ({marks}))""")
        args += tag_keys
    for k in keywords:
        parts.append("(CASE WHEN q.text LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END)"
                     " + (CASE WHEN q.use_when LIKE ? ESCAPE '\\' THEN 2 ELSE 0 END)")
        args += [_like(k), _like(k)]
    sql = f"""
        SELECT * FROM (
            SELECT q.id, e.season, e.episode, e.title, q.start_s, q.end_s, q.text, q.speaker,
                   q.quotability, q.use_when,
                   (SELECT group_concat(t.name, '、') FROM quote_tags qt
                    JOIN tags t ON t.id = qt.tag_id WHERE qt.quote_id = q.id) AS tags,
                   {' + '.join(parts)} AS score
            FROM quotes q JOIN episodes e ON e.id = q.episode_id)
        WHERE score > 0
        ORDER BY score DESC, quotability DESC, id
        LIMIT ?"""
    rows = db.execute(sql, [*args, limit])
    return [dict(zip(SEARCH_COLUMNS, row)) for row in rows]
