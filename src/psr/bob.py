"""`bob`：海綿寶寶金句庫。

    bob fix [--limit N] [--workers 4]     照語音修正誤聽（同音字），改寫 Drive 上的字幕
    bob index [--limit N] [--workers 4]   用 Haiku 標記已修正的字幕
    bob ask "明天要上台報告好緊張"          找出最適合引用的台詞
    bob audit                             彙整 Haiku 審計出的字幕問題與標籤提案

全部在本機執行：模型走 `claude -p`（訂閱額度），資料寫進 Turso。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from googleapiclient.discovery import build

from psr import claude_cli, drive, fix, glossary, quotedb, quotes
from psr.batch import WORK_SUBFOLDER, validate_folder_id
from psr.cli import TOKEN_PATH
from psr.srt import parse, render

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "taxonomy.yml"
GLOSSARY_PATH = ROOT / "glossary.yml"
SRT_SUFFIX = ".zh-Hant.srt"
BACKUP_SUFFIX = ".asr.srt"  # 修正前的原始辨識結果，放在 _psr/


def _drive_srts(folder_id: str):
    creds = drive.load_credentials(TOKEN_PATH.read_text())
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    children = drive.list_children(service, folder_id)
    files = [f for f in children if f["name"].endswith(SRT_SUFFIX)]
    work_id = next((f["id"] for f in children if f["name"] == WORK_SUBFOLDER), None)
    return service, sorted(files, key=lambda f: f["name"]), work_id


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _write(service, name: str, folder_id: str, text: str) -> str:
    """把文字寫成 Drive 檔案（同名就更新），回傳內容的 md5（等於 Drive 的 md5Checksum）。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.srt"
        path.write_bytes(text.encode("utf-8"))
        drive.find_or_create(service, name, folder_id, str(path), "text/plain")
    return _md5(text)


def _fix_episode(text: str, terms: list[str], model: str):
    cues = fix.traditionalize(parse(text))
    raw = claude_cli.ask(fix.user_prompt(cues), system=fix.system_prompt(terms),
                         schema=fix.SCHEMA, model=model)
    return fix.apply_fixes(cues, raw.get("fixes", []))


def cmd_fix(args) -> int:
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    state = quotedb.fix_state(db)
    terms = [t.correct for t in glossary.load(GLOSSARY_PATH).entries]
    service, files, work_id = _drive_srts(args.folder)
    if not work_id:
        print(f"找不到 {WORK_SUBFOLDER}/ 子資料夾，無處備份原始字幕。", file=sys.stderr)
        return 1
    backups = {f["name"]: f["id"] for f in drive.list_children(service, work_id)}

    pending = [f for f in files
               if state.get(f["name"].removesuffix(SRT_SUFFIX)) != (f.get("md5Checksum"), fix.FIX_VERSION)]
    print(f"字幕 {len(files)} 集，已修正 {len(files) - len(pending)}，待修正 {len(pending)}（版本 {fix.FIX_VERSION}）")
    if args.limit:
        pending = pending[:args.limit]

    # 每集的修正一律從原始辨識結果出發：Drive 上的檔案若仍是上次修正的輸出，
    # 就改讀 _psr/ 的備份；若字幕被 pipeline 重做過（md5 對不上），新檔就是新的原始版本。
    jobs = []
    for f in pending:
        stem = f["name"].removesuffix(SRT_SUFFIX)
        ours = stem in state and state[stem][0] == f.get("md5Checksum")
        backup_id = backups.get(stem + BACKUP_SUFFIX)
        if ours and not backup_id:
            print(f"{stem}：找不到原始備份，略過", file=sys.stderr)
            continue
        current = drive.read_text(service, f["id"])
        source = drive.read_text(service, backup_id) if ours else current
        jobs.append((f, stem, current, source, not ours))

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fix_episode, source, terms, args.model): (f, stem, current, source, fresh)
                   for f, stem, current, source, fresh in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            f, stem, current, source, fresh = futures[fut]
            label = _label(quotes.parse_episode_name(f["name"]))
            try:
                cues, applied, skipped = fut.result()
                if fresh:
                    _write(service, stem + BACKUP_SUFFIX, work_id, source)
                new_text = render(cues)
                md5_after = (_write(service, f["name"], args.folder, new_text)
                             if new_text != current else f.get("md5Checksum") or _md5(current))
                quotedb.record_fix(db, stem, md5_before=_md5(source), md5_after=md5_after,
                                   version=fix.FIX_VERSION, applied=applied, skipped=skipped)
            except Exception as e:  # 單集失敗不中斷整批，下次重跑會補
                failures += 1
                print(f"[{n}/{len(jobs)}] {label} 失敗：{e}", file=sys.stderr, flush=True)
                continue
            sample = "、".join(f"{x.wrong}→{x.right}" for x in applied[:4])
            print(f"[{n}/{len(jobs)}] {label}：修正 {len(applied)}、擋下 {len(skipped)}  {sample}", flush=True)
    return 1 if failures else 0


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
    fixed = {stem: md5 for stem, (md5, _) in quotedb.fix_state(db).items()}
    service, files, _ = _drive_srts(args.folder)

    # 只標記修正過的字幕：沒修正就標，修正後 md5 改變又得整集重標。
    ready = [f for f in files if fixed.get(f["name"].removesuffix(SRT_SUFFIX)) == f.get("md5Checksum")]
    pending = []
    for f in ready:
        meta = quotes.parse_episode_name(f["name"])
        if done.get(meta.stem) != (f.get("md5Checksum"), version):
            pending.append((f, meta))
    print(f"字幕 {len(files)} 集，已修正 {len(ready)}，已標記 {len(ready) - len(pending)}，"
          f"待標記 {len(pending)}（版本 {version}）")
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
                print(f"[{n}/{len(pending)}] {_label(meta)} 失敗：{e}", file=sys.stderr, flush=True)
                continue
            print(f"[{n}/{len(pending)}] {_label(meta)} {meta.title}：金句 {len(result.quotes)}、"
                  f"審計 {len(result.issues)}、丟棄 {result.rejected}、新標籤提案 {len(result.new_tags)}", flush=True)
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
    print("\n## 最常見的誤聽修正（重複出現的可加進 glossary.yml 的 wrong）")
    for heard, fixed_text, n in db.execute("""
            SELECT heard, fixed, count(*) FROM fix_log WHERE applied = 1
            GROUP BY heard, fixed ORDER BY 3 DESC LIMIT ?""", [args.top]):
        print(f"- ×{n}「{heard}」→「{fixed_text}」")
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
    for name, help_text in (("fix", "照語音修正誤聽，改寫 Drive 上的字幕（原檔備份到 _psr/）"),
                            ("index", "用 Haiku 標記已修正、尚未標記或內容已變的字幕")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--folder", default=os.environ.get("DRIVE_FOLDER_ID", ""))
        p.add_argument("--limit", type=int, default=0, help="本輪最多幾集，0 = 全部")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--model", default="haiku")
    ask = sub.add_parser("ask", help="描述你的處境，找海綿寶寶台詞")
    ask.add_argument("situation")
    ask.add_argument("--model", default="haiku")
    ask.add_argument("-v", "--verbose", action="store_true")
    audit = sub.add_parser("audit", help="彙整字幕審計結果與標籤提案")
    audit.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cmd in ("fix", "index"):
        args.folder = validate_folder_id(args.folder)
    return {"fix": cmd_fix, "index": cmd_index, "ask": cmd_ask, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
