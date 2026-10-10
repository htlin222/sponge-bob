"""`bob`：海綿寶寶金句庫。

    bob reseg [--limit N] [--only 集名片段] 用 Haiku 重新加標點與換人記號，重新斷句字幕
    bob fix [--limit N] [--workers 4]     照語音修正誤聽（同音字），改寫 Drive 上的字幕
    bob index [--limit N] [--workers 4]   用 Haiku 標記已修正的字幕
    bob ask "明天要上台報告好緊張"          找出最適合引用的台詞
    bob audit                             彙整 Haiku 審計出的字幕問題與標籤提案

全部在本機執行：模型走 `claude -p`（訂閱額度），資料寫進 Turso。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from googleapiclient.discovery import build

from psr import claude_cli, cleanup, drive, fix, glossary, quotedb, quotes
from psr import punctuate as punct_mod
from psr.batch import WORK_SUBFOLDER, validate_folder_id
from psr.cli import TOKEN_PATH, _build_cues
from psr.models import Word
from psr.srt import parse, render
from psr.text import to_traditional
from psr.validate import validate
from psr.youtube import drive_paths

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


def _fix_episode(text: str, gloss: glossary.Glossary, model: str):
    cues, known = fix.apply_glossary(fix.traditionalize(parse(text)), gloss.corrections())
    raw = claude_cli.ask(fix.user_prompt(cues), system=fix.system_prompt([e.correct for e in gloss.entries]),
                         schema=fix.SCHEMA, model=model)
    cues, applied, skipped = fix.apply_fixes(cues, raw.get("fixes", []))
    return cues, known + applied, skipped


def cmd_fix(args) -> int:
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    state = quotedb.fix_state(db)
    gloss = glossary.load(GLOSSARY_PATH)
    # 對照表一改就要重修：版本帶上 glossary 的 hash。重修只改字，index 會就地更新文字。
    version = f"{fix.FIX_VERSION}-g{gloss.content_hash()[:8]}"
    service, files, work_id = _drive_srts(args.folder)
    if not work_id:
        print(f"找不到 {WORK_SUBFOLDER}/ 子資料夾，無處備份原始字幕。", file=sys.stderr)
        return 1
    backups = {f["name"]: f["id"] for f in drive.list_children(service, work_id)}

    pending = [f for f in files
               if state.get(f["name"].removesuffix(SRT_SUFFIX)) != (f.get("md5Checksum"), version)]
    print(f"字幕 {len(files)} 集，已修正 {len(files) - len(pending)}，待修正 {len(pending)}（版本 {version}）")
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
        futures = {pool.submit(_fix_episode, source, gloss, args.model): (f, stem, current, source, fresh)
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
                                   version=version, applied=applied, skipped=skipped)
            except Exception as e:  # 單集失敗不中斷整批，下次重跑會補
                failures += 1
                print(f"[{n}/{len(jobs)}] {label} 失敗：{e}", file=sys.stderr, flush=True)
                continue
            sample = "、".join(f"{x.wrong}→{x.right}" for x in applied[:4])
            print(f"[{n}/{len(jobs)}] {label}：修正 {len(applied)}、擋下 {len(skipped)}  {sample}", flush=True)
    return 1 if failures else 0


# 標點（模型輸出，有快取）與斷句規則（純函式，便宜）分開版本：只改規則時
# 直接用快取的標點重建，不必再花一次模型額度。
PUNCT_VERSION = "1"
SEG_VERSION = "2"  # 2：過短字幕併給鄰居、折行左半也限寬
RESEG_VERSION = f"p{PUNCT_VERSION}-s{SEG_VERSION}"
PUNCT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _haiku_punctuate(chunk: str, model: str, attempts: int = 2) -> str | None:
    """跟雲端 DeepSeek 同一份「只插標點」契約與驗證。驗證不過就從中間的停頓
    切成兩半各自重試——塊越小模型越守規矩；切到不能再切仍失敗才回 None，退回原文。"""
    for _ in range(attempts):
        try:
            out = claude_cli.ask(chunk, system=punct_mod.SYSTEM, schema=PUNCT_SCHEMA, model=model)["text"]
        except claude_cli.ClaudeError:
            continue
        out = punct_mod.fullwidth(to_traditional(out.strip()))
        if punct_mod.similarity(chunk, out) >= punct_mod.MIN_SIMILARITY:
            return out
    halves = punct_mod.split_at_pause(chunk)
    if halves is None:
        return None
    parts = [_haiku_punctuate(h, model, attempts) for h in halves]
    return None if None in parts else "".join(parts)


def _reseg_episode(words: list[Word], gloss: glossary.Glossary, audio_duration: float, model: str,
                   cached: str | None):
    clean, _ = cleanup.drop_hallucinations(words, list(gloss.hallucinations))
    clean, _ = cleanup.apply_corrections(clean, gloss.corrections())
    clean, _ = cleanup.collapse_repeats(clean)
    failed = 0
    punctuated = cached
    if punctuated is None:
        chunks = punct_mod.make_chunks(clean)
        with ThreadPoolExecutor(max_workers=3) as pool:
            outs = list(pool.map(lambda c: _haiku_punctuate(c, model), chunks))
        failed = sum(o is None for o in outs)
        punctuated = "".join(o if o else to_traditional(c) for o, c in zip(outs, chunks))
    cues = _build_cues(clean, punctuated, audio_duration)
    return cues, failed, punctuated


def cmd_reseg(args) -> int:
    """從 _psr/ 的原始轉錄重新加標點、重新斷句，覆寫 Drive 上的字幕。

    之後 `bob fix` 會把新字幕視為新的原始版本（md5 對不上）重新備份與修正，
    `bob index` 也會因為時間軸改變而整集重標。
    """
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    state = quotedb.reseg_state(db)
    gloss = glossary.load(GLOSSARY_PATH)
    service, files, work_id = _drive_srts(args.folder)
    work = {f["name"]: f["id"] for f in drive.list_children(service, work_id)} if work_id else {}
    pending = [f for f in files
               if args.redo or state.get(f["name"].removesuffix(SRT_SUFFIX)) != RESEG_VERSION]
    print(f"字幕 {len(files)} 集，已重新斷句 {len(files) - len(pending)}，待處理 {len(pending)}（版本 {RESEG_VERSION}）",
          flush=True)
    pending = [f for f in pending if args.only in f["name"]]
    if args.limit:
        pending = pending[:args.limit]

    jobs = []
    for f in pending:
        stem = f["name"].removesuffix(SRT_SUFFIX)
        names = drive_paths(stem)
        if names["words"] not in work:
            print(f"{stem}：找不到 words.json，略過", file=sys.stderr)
            continue
        raw = json.loads(drive.read_text(service, work[names["words"]]))
        words = [Word(w["text"], w["start"], w["end"]) for w in raw]
        duration = words[-1].end if words else 0.0
        if names["manifest"] in work:
            timings = json.loads(drive.read_text(service, work[names["manifest"]])).get("timings", {})
            duration = float(timings.get("audio_duration") or duration)
        jobs.append((f, stem, words, duration, quotedb.cached_punctuation(db, stem, PUNCT_VERSION)))

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_reseg_episode, words, gloss, duration, args.model, cached): (f, stem, duration, cached)
                   for f, stem, words, duration, cached in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            f, stem, duration, cached = futures[fut]
            label = _label(quotes.parse_episode_name(f["name"]))
            try:
                cues, failed, punctuated = fut.result()
                if cached is None and not failed:
                    quotedb.cache_punctuation(db, stem, PUNCT_VERSION, punctuated)
                violations = validate(cues, duration)
                md5 = _write(service, f["name"], args.folder, render(cues))
                quotedb.record_reseg(db, stem, version=RESEG_VERSION, md5=md5, cues=len(cues),
                                     failed_chunks=failed, violations=len(violations))
            except Exception as e:  # 單集失敗不中斷整批，下次重跑會補
                failures += 1
                print(f"[{n}/{len(jobs)}] {label} 失敗：{e}", file=sys.stderr, flush=True)
                continue
            print(f"[{n}/{len(jobs)}] {label}：字幕 {len(cues)}、換人 {punctuated.count(punct_mod.TURN)}、"
                  f"標點失敗塊 {failed}、違規 {len(violations)}{'（快取標點）' if cached else ''}", flush=True)
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
    refreshed = 0
    for f in ready:
        meta = quotes.parse_episode_name(f["name"])
        md5 = f.get("md5Checksum")
        if done.get(meta.stem) == (md5, version):
            continue
        # 同版本、只是字幕又修了字：時間軸沒變就只更新文字，不重標。
        if meta.stem in done and done[meta.stem][1] == version:
            cues = parse(drive.read_text(service, f["id"]))
            if [(c.index, c.start, c.end) for c in cues] == quotedb.cue_timings(db, meta.stem):
                quotedb.refresh_text(db, meta.stem, srt_md5=md5, cues=cues)
                refreshed += 1
                continue
        pending.append((f, meta))
    print(f"字幕 {len(files)} 集，已修正 {len(ready)}，已標記 {len(ready) - len(pending)}"
          f"（其中只更新文字 {refreshed}），待標記 {len(pending)}（版本 {version}）")
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
    for name, help_text in (("reseg", "用 Haiku 重新加標點與換人記號，重新斷句 Drive 上的字幕"),
                            ("fix", "照語音修正誤聽，改寫 Drive 上的字幕（原檔備份到 _psr/）"),
                            ("index", "用 Haiku 標記已修正、尚未標記或內容已變的字幕")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--folder", default=os.environ.get("DRIVE_FOLDER_ID", ""))
        p.add_argument("--limit", type=int, default=0, help="本輪最多幾集，0 = 全部")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--model", default="haiku")
        p.add_argument("--only", default="", help="只處理檔名含此字串的集數")
        if name == "reseg":
            p.add_argument("--redo", action="store_true", help="已重新斷句過的也重做")
    ask = sub.add_parser("ask", help="描述你的處境，找海綿寶寶台詞")
    ask.add_argument("situation")
    ask.add_argument("--model", default="haiku")
    ask.add_argument("-v", "--verbose", action="store_true")
    audit = sub.add_parser("audit", help="彙整字幕審計結果與標籤提案")
    audit.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cmd in ("reseg", "fix", "index"):
        args.folder = validate_folder_id(args.folder)
    return {"reseg": cmd_reseg, "fix": cmd_fix, "index": cmd_index, "ask": cmd_ask, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
