"""`bob`：海綿寶寶金句庫。

    bob index [--limit N] [--workers 4]   用 Haiku 標記 Drive 上已完成的字幕
    bob ask "明天要上台報告好緊張"          找出最適合引用的台詞
    bob audit                             彙整 Haiku 審計出的字幕問題與標籤提案

全部在本機執行：模型走 `claude -p`（訂閱額度），資料寫進 Turso。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from googleapiclient.discovery import build

from psr import claude_cli, drive, quotedb, quotes
from psr.batch import validate_folder_id
from psr.cli import TOKEN_PATH
from psr.srt import parse

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "taxonomy.yml"
SRT_SUFFIX = ".zh-Hant.srt"


def _drive_srts(folder_id: str):
    creds = drive.load_credentials(TOKEN_PATH.read_text())
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    files = [f for f in drive.list_children(service, folder_id) if f["name"].endswith(SRT_SUFFIX)]
    return service, sorted(files, key=lambda f: f["name"])


def _label(ep: quotes.EpisodeMeta) -> str:
    return f"S{ep.season:02d}E{ep.episode:03d}"


def _tag_episode(text: str, meta: quotes.EpisodeMeta, taxonomy, model: str):
    cues = parse(text)
    raw = claude_cli.ask(quotes.user_prompt(meta, cues), system=quotes.SYSTEM_PROMPT,
                         schema=quotes.output_schema(taxonomy), model=model)
    return cues, quotes.accept(raw, cues, taxonomy)


def cmd_index(args) -> int:
    taxonomy = quotes.load_taxonomy(TAXONOMY_PATH)
    version = quotes.index_version(taxonomy)
    db = quotedb.connect(args.db)
    quotedb.init(db, taxonomy)
    done = quotedb.indexed_versions(db)
    service, files = _drive_srts(args.folder)

    pending = []
    for f in files:
        meta = quotes.parse_episode_name(f["name"])
        if done.get(meta.stem) != (f.get("md5Checksum"), version):
            pending.append((f, meta))
    print(f"字幕 {len(files)} 集，已標記 {len(files) - len(pending)}，待標記 {len(pending)}（版本 {version}）")
    if args.limit:
        pending = pending[:args.limit]

    # Drive client 不是 thread-safe：先在主執行緒下載完，平行的只有 claude -p。
    texts = {f["id"]: drive.read_text(service, f["id"]) for f, _ in pending}
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_tag_episode, texts[f["id"]], meta, taxonomy, args.model): (f, meta)
                   for f, meta in pending}
        for n, fut in enumerate(as_completed(futures), 1):
            f, meta = futures[fut]
            try:
                cues, result = fut.result()
                md5 = f.get("md5Checksum") or hashlib.md5(texts[f["id"]].encode()).hexdigest()
                quotedb.write_episode(db, meta, drive_id=f["id"], srt_md5=md5, version=version,
                                      cues=cues, result=result)
            except Exception as e:  # 單集失敗不中斷整批，下次重跑會補
                failures += 1
                print(f"[{n}/{len(pending)}] {_label(meta)} 失敗：{e}", file=sys.stderr)
                continue
            print(f"[{n}/{len(pending)}] {_label(meta)} {meta.title}：金句 {len(result.quotes)}、"
                  f"審計 {len(result.issues)}、丟棄 {result.rejected}、新標籤提案 {len(result.new_tags)}")
    return 1 if failures else 0


PLAN_SYSTEM = """你幫使用者在《海綿寶寶》金句庫裡找台詞。把使用者描述的生活處境轉成搜尋條件：
- 標籤只能從 schema 選項挑，挑最貼近的，每類 0–3 個。
- characters 只在使用者點名角色時才填，否則留空（角色標籤會讓該角色的台詞全部加分）。
- keywords：台詞或情境描述裡可能出現的 2–4 字詞，台灣用語，最多 8 個。"""

PICK_SYSTEM = """你是海綿寶寶通。從候選台詞中挑出最適合使用者此刻引用的 1–5 段，依推薦程度排序。
why 用一句話說明為什麼適合（可以帶點幽默）。不適合的寧可不挑。"""


def cmd_ask(args) -> int:
    taxonomy = quotes.load_taxonomy(TAXONOMY_PATH)
    db = quotedb.connect(args.db)
    tag_props = {facet: {"type": "array", "items": {"enum": list(names)}, "maxItems": 3}
                 for facet, names in taxonomy.facets.items()}
    plan = claude_cli.ask(args.situation, system=PLAN_SYSTEM, model=args.model, schema={
        "type": "object",
        "properties": {**tag_props, "keywords": {"type": "array", "items": {"type": "string"},
                                                  "maxItems": 8}},
        "required": [*quotes.FACETS, "keywords"],
    })
    tags = [(facet, name) for facet in quotes.FACETS for name in plan.get(facet, [])]
    candidates = quotedb.search(db, tags=tags, keywords=plan.get("keywords", []), limit=30)
    if not candidates:
        print("找不到相關台詞。")
        return 1
    listing = "\n\n".join(f"[{c['id']}] {c['speaker']}：{c['text']}\n情境：{c['use_when']}"
                          for c in candidates)
    picks = claude_cli.ask(f"使用者的處境：{args.situation}\n\n候選台詞：\n{listing}",
                           system=PICK_SYSTEM, model=args.model, schema={
        "type": "object",
        "properties": {"picks": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "why": {"type": "string"}},
            "required": ["id", "why"]}}},
        "required": ["picks"],
    })
    by_id = {c["id"]: c for c in candidates}
    shown = 0
    for p in picks.get("picks", []):
        c = by_id.get(p.get("id"))
        if not c:
            continue
        shown += 1
        start = int(c["start"])
        print(f"\n{shown}. 「{c['text'].replace(chr(10), ' ')}」—— {c['speaker']}")
        print(f"   S{c['season']:02d}E{c['episode']:03d}《{c['title']}》 {start // 60:02d}:{start % 60:02d}")
        print(f"   {p.get('why', '')}")
    if args.verbose:
        print(f"\n搜尋條件：標籤 {tags}，關鍵字 {plan.get('keywords')}，候選 {len(candidates)} 筆")
    return 0 if shown else 1


def cmd_audit(args) -> int:
    db = quotedb.connect(args.db)
    print("## 審計問題（依類型）")
    for kind, n in db.execute("SELECT type, count(*) FROM audit_issues GROUP BY type ORDER BY 2 DESC"):
        print(f"- {kind}：{n}")
    print("\n## 最常見的修正建議（可考慮加進 glossary.yml）")
    rows = db.execute("""
        SELECT a.type, c.text, a.suggestion, count(*) AS n
        FROM audit_issues a JOIN cues c ON c.episode_id = a.episode_id AND c.idx = a.cue_idx
        WHERE a.suggestion != '' AND a.type IN ('人名', '錯字', '幻覺')
        GROUP BY a.type, c.text, a.suggestion ORDER BY n DESC LIMIT ?""", [args.top])
    for kind, text, suggestion, n in rows:
        print(f"- [{kind}] ×{n}「{text}」→「{suggestion}」")
    print("\n## 新標籤提案")
    for facet, name, n in db.execute("""
            SELECT facet, name, count(*) FROM tag_suggestions
            GROUP BY facet, name ORDER BY 3 DESC LIMIT ?""", [args.top]):
        print(f"- {facet}/{name} ×{n}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bob", description="海綿寶寶金句庫")
    parser.add_argument("--db", help="改用本機 SQLite 檔（預設連 Turso）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    idx = sub.add_parser("index", help="用 Haiku 標記尚未標記或已更新的字幕")
    idx.add_argument("--folder", default=os.environ.get("DRIVE_FOLDER_ID", ""))
    idx.add_argument("--limit", type=int, default=0, help="本輪最多幾集，0 = 全部")
    idx.add_argument("--workers", type=int, default=4)
    idx.add_argument("--model", default="haiku")
    ask = sub.add_parser("ask", help="描述你的處境，找海綿寶寶台詞")
    ask.add_argument("situation")
    ask.add_argument("--model", default="haiku")
    ask.add_argument("-v", "--verbose", action="store_true")
    audit = sub.add_parser("audit", help="彙整字幕審計結果與標籤提案")
    audit.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cmd == "index":
        args.folder = validate_folder_id(args.folder)
    return {"index": cmd_index, "ask": cmd_ask, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
