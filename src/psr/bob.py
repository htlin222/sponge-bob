"""`bob`：海綿寶寶金句庫。

    bob reseg [--limit N] [--only 集名片段] 用 Haiku 重新加標點與換人記號，重新斷句字幕
    bob fix [--limit N] [--workers 4]     照語音修正誤聽（同音字），改寫 Drive 上的字幕
    bob index [--limit N] [--workers 4]   用 Haiku 標記已修正的字幕
    bob ask "明天要上台報告好緊張" ["朋友很怕上台報告時"]  混合檢索最貼切的台詞（--rewrite 讓模型改寫、--pick 再讓模型挑）
    bob embed                             補齊／更新金句的語意向量（bob index 結束時也會做）
    bob ctx ep161 6:36                    列出某集某時間前後的字幕，看這段在演什麼
    bob grep 大巨蛋 市長                   字幕關鍵字搜尋（含任一詞）
    bob evalset [-n 200]                  抽樣金句出檢索評測題（只存 Turso）
    bob eval [--label baseline]           用評測題衡量 bob ask 的命中率
    bob backup [--keep 8]                 整庫備份到 Drive 的 _psr/backups/
    bob health                            檢查 Drive token、Turso、DeepSeek
    bob audit                             彙整 Haiku 審計出的字幕問題與標籤提案
    bob refs refs.json                    載入台灣配音版台詞參考庫（只存 Turso）
    bob scan [--dry-run]                  拼音比對找誤聽，只把嫌疑交給模型確認
    bob still ep161 --text "開玉米罐"       抓台詞畫面加字幕與出處，一次出 6 張候選

模型預設走本機 `claude -p`（訂閱額度）；`--model deepseek-v4-flash` 改走 DeepSeek API，
GitHub Actions 上的 library workflow 用這個。資料寫進 Turso。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from googleapiclient.discovery import build

from psr import backup, cleanup, drive, evalset, fandom, fix, glossary, llm, quotedb, quotes, refaudit, search, semantic, still
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


def _fix_episode(text: str, gloss: glossary.Glossary, model: str, local: list[tuple[str, str]] = ()):
    cues, known = fix.apply_glossary(fix.traditionalize(parse(text)), gloss.corrections())
    if local:  # `bob scan` 確認過的單集修正
        cues, scanned = fix.apply_glossary(cues, list(local))
        known += scanned
    raw = llm.ask(fix.user_prompt(cues), system=fix.system_prompt([e.correct for e in gloss.entries]),
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
    local = quotedb.episode_fixes(db)

    pending = [f for f in files
               if state.get(f["name"].removesuffix(SRT_SUFFIX)) != (f.get("md5Checksum"), version)]
    print(f"字幕 {len(files)} 集，已修正 {len(files) - len(pending)}，待修正 {len(pending)}（版本 {version}）")
    pending = [f for f in pending if args.only in f["name"]]
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
        futures = {pool.submit(_fix_episode, source, gloss, args.model, local.get(stem, ())):
                   (f, stem, current, source, fresh)
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
            # 公開 repo 的 Actions log 人人看得到：--quiet 不印字幕片段
            sample = "" if args.quiet else "、".join(f"{x.wrong}→{x.right}" for x in applied[:4])
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
            out = llm.ask(chunk, system=punct_mod.SYSTEM, schema=PUNCT_SCHEMA, model=model)["text"]
        except llm.ERRORS:
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
    raw = llm.ask(quotes.user_prompt(meta, cues), system=quotes.system_prompt(model),
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
    pending = [(f, meta) for f, meta in pending if args.only in f["name"]]
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
    _sync_vectors(db)
    return 1 if failures else 0


def _sync_vectors(db) -> int:
    """替還沒有向量、或台詞／情境改過的金句重算向量。"""
    todo = semantic.stale(quotedb.quote_docs(db))
    for i in range(0, len(todo), 256):
        batch = todo[i:i + 256]
        vecs = semantic.embed_docs([document for _, document, _ in batch])
        quotedb.save_vectors(db, [(qid, d, v) for (qid, _, d), v in zip(batch, vecs)])
        print(f"語意向量 {i + len(batch)}/{len(todo)}", flush=True)
    return len(todo)


def cmd_embed(args) -> int:
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    print(f"更新 {_sync_vectors(db)} 段金句的語意向量")
    return 0


def cmd_evalset(args) -> int:
    """抽樣金句，讓模型出「使用者會怎麼問」的評測題；已出過題的金句跳過，中斷後重跑會接著做。"""
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    rows = [tuple(r) for r in db.execute("SELECT id, episode_id, quotability FROM quotes")]
    done = {qid for qid, *_ in quotedb.eval_queries(db)}
    todo = [qid for qid in evalset.sample(rows, args.n, args.seed) if qid not in done]
    print(f"抽樣 {args.n} 段，已出題 {len(done)}，待出題 {len(todo)}")
    items = []
    for qid in todo:
        ep_id, cue_from, cue_to, text = db.execute(
            "SELECT episode_id, cue_from, cue_to, text FROM quotes WHERE id = ?", (qid,))[0]
        context = [r[0].replace("\n", " ") for r in db.execute(
            "SELECT text FROM cues WHERE episode_id = ? AND idx BETWEEN ? AND ? ORDER BY idx",
            (ep_id, cue_from - 4, cue_to + 3))]
        items.append((qid, text, context))
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]

    def run(batch):
        raw = llm.ask(evalset.gen_prompt(batch), system=evalset.GEN_SYSTEM,
                      schema=evalset.GEN_SCHEMA, model=args.model)
        return evalset.accept(raw, {qid: text for qid, text, _ in batch})

    made = failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(run, b) for b in batches]):
            try:
                got = fut.result()
            except Exception as exc:  # 單批失敗不影響其他批，重跑會補
                failures += 1
                print(f"  一批出題失敗：{type(exc).__name__}", file=sys.stderr)
                continue
            quotedb.save_eval_queries(db, got)
            made += len(got)
    print(f"新增評測題 {made} 題（失敗 {failures} 批）")
    return 1 if failures else 0


def cmd_eval(args) -> int:
    """用評測題跑一次 `bob ask` 的檢索，印各問法的命中率（只印統計，不印題目）。"""
    db = quotedb.connect(args.db)
    queries = quotedb.eval_queries(db)
    if not queries:
        print("還沒有評測題，先跑 `bob evalset`。")
        return 1
    k = max(evalset.KS)
    extra: dict[str, list[str]] = {}
    if args.rewrite:
        quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
        extra = quotedb.eval_rewrites(db, args.rewrite)
        todo = sorted({q for _, _, q in queries} - set(extra))
        batches = [todo[i:i + 25] for i in range(0, len(todo), 25)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for batch, got in zip(batches, pool.map(lambda b: search.rewrite_many(b, args.rewrite), batches)):
                rows = {batch[i]: situations for i, situations in got.items()}
                quotedb.save_eval_rewrites(db, args.rewrite, rows)
                extra.update(rows)
        print(f"改寫（{args.rewrite}）：{sum(1 for _, _, q in queries if q in extra)}/{len(queries)} 題")
    index = search.lexical_index(db) if args.mode == "hybrid" else None

    def rank(q):
        qid, _, query = q
        if index is None:
            ids = [h["id"] for h in quotedb.nearest(db, semantic.embed_query(query), limit=k)]
        else:
            ids = search.rank(db, [query, *extra.get(query, [])], limit=k, index=index)
        return evalset.rank_of(qid, ids)

    semantic.embed_query("")  # 先在主執行緒載入模型
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ranks = list(pool.map(rank, queries))
    by_style = {style: evalset.metrics([r for (_, s, _), r in zip(queries, ranks) if s == style])
                for style in sorted({s for _, s, _ in queries})}
    by_style["all"] = evalset.metrics(ranks)
    cols = [f"hit@{n}" for n in evalset.KS] + ["mrr"]
    print(f"{'問法':<12}{'題數':>6}" + "".join(f"{c:>9}" for c in cols))
    for style, m in by_style.items():
        print(f"{style:<12}{m['n']:>6}" + "".join(f"{m[c]:>9.3f}" for c in cols))
    if args.label:
        quotedb.save_eval_run(db, args.label, by_style)
        print(f"已記錄為「{args.label}」")
    return 0


PICK_SYSTEM = """你是海綿寶寶通。從候選台詞中挑出最適合使用者此刻引用的 1–5 段，依推薦程度排序。
why 用一句話說明為什麼適合（可以帶點幽默）。不適合的寧可不挑。"""


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def cmd_ask(args) -> int:
    db = quotedb.connect(args.db)
    queries = list(args.situation)
    if args.rewrite:
        extra = search.rewrite(queries[0], args.model)
        print("改寫：" + "／".join(extra))
        queries += extra
    hits = search.search(db, queries, limit=30 if args.pick else args.top)
    if not hits:
        print("金句還沒有語意向量，先跑 `bob embed`。")
        return 1
    if args.pick:
        listing = "\n\n".join(f"[{c['id']}] {c['speaker']}：{c['text']}\n情境：{c['use_when']}" for c in hits)
        picks = llm.ask(f"使用者的處境：{'／'.join(args.situation)}\n\n候選台詞：\n{listing}",
                               system=PICK_SYSTEM, model=args.model, schema={
            "type": "object",
            "properties": {"picks": {"type": "array", "maxItems": 5, "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "why": {"type": "string"}},
                "required": ["id", "why"]}}},
            "required": ["picks"],
        })
        by_id = {c["id"]: c for c in hits}
        hits = [{**by_id[p["id"]], "why": p.get("why", "")} for p in picks.get("picks", []) if p.get("id") in by_id]
    for n, c in enumerate(hits, 1):
        print(f"{n}. 「{c['text'].replace(chr(10), ' / ')}」—— {c['speaker']}")
        print(f"   S{c['season']:02d}E{c['episode']:03d}《{c['title']}》{_stamp(c['start'])}　"
              f"★{c['quotability']}　{c.get('why') or c['use_when']}")
    return 0 if hits else 1


def cmd_ctx(args) -> int:
    db = quotedb.connect(args.db)
    eps = db.execute("SELECT id, stem, title FROM episodes WHERE stem LIKE ?", (f"%{args.episode}%",))
    if len(eps) != 1:
        print(f"「{args.episode}」對到 {len(eps)} 集：{[e[1] for e in eps][:5]}，請寫得更精確")
        return 1
    ep_id, stem, _ = eps[0]
    at = _clock(args.at)
    rows = db.execute("SELECT start_s, text FROM cues WHERE episode_id = ? AND start_s BETWEEN ? AND ? "
                      "ORDER BY idx", (ep_id, at - args.span, at + args.span))
    print(stem)
    current = max((r[0] for r in rows if r[0] < at + 1), default=None)  # m:ss 涵蓋整秒
    for start, text in rows:
        print(f"{'▶' if start == current else ' '}{_stamp(start):>6} {text.replace(chr(10), ' ')}")
    return 0


def cmd_grep(args) -> int:
    rows = quotedb.grep_cues(quotedb.connect(args.db), args.terms, args.limit)
    for stem, start, text in rows:
        print(f"{stem.removeprefix('海綿寶寶_')}  {_stamp(start):>6}  {text.replace(chr(10), ' ')}")
    if len(rows) == args.limit:
        print(f"（只列前 {args.limit} 句，用 --limit 調高）")
    return 0


def cmd_backup(args) -> int:
    from datetime import date

    service = build("drive", "v3", credentials=drive.load_credentials(TOKEN_PATH.read_text()),
                    cache_discovery=False)
    folder = drive.ensure_folder(service, backup.FOLDER, drive.ensure_folder(service, WORK_SUBFOLDER, args.folder))
    name = backup.name_for(date.today())
    with tempfile.TemporaryDirectory() as tmp:
        raw, packed = Path(tmp) / "bob.sqlite", Path(tmp) / name
        counts = quotedb.dump(quotedb.connect(args.db), raw)
        with raw.open("rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        drive.find_or_create(service, name, folder, str(packed), "application/gzip")
        size = packed.stat().st_size
    print(f"備份 {name}（{size / 1e6:.1f} MB）：" + "、".join(f"{t} {n}" for t, n in sorted(counts.items())))
    names = {f["name"]: f["id"] for f in drive.list_children(service, folder)}
    for old in backup.to_trash(list(names), args.keep):
        service.files().update(fileId=names[old], body={"trashed": True}).execute()
        print(f"移到垃圾桶：{old}")
    return 0


def cmd_health(args) -> int:
    """Drive token、Turso、DeepSeek 都還能用嗎？任一項失敗就回傳 1（排程用來開 issue）。"""
    def drive_ok():
        service = build("drive", "v3", credentials=drive.load_credentials(TOKEN_PATH.read_text()),
                        cache_discovery=False)
        service.files().get(fileId=args.folder, fields="id", supportsAllDrives=True).execute()
        return "token 有效、資料夾可讀"

    def turso_ok():
        (n,), = quotedb.connect(args.db).execute("SELECT count(*) FROM episodes")
        return f"{n} 集"

    def deepseek_ok():
        if not os.environ.get("DEEPSEEK_API_KEY"):
            return "未設定 DEEPSEEK_API_KEY，略過"
        from openai import OpenAI

        models = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=llm.DEEPSEEK_URL).models.list()
        return f"可用模型 {len(models.data)} 個"

    failed = 0
    for label, check in (("Drive", drive_ok), ("Turso", turso_ok), ("DeepSeek", deepseek_ok)):
        try:
            print(f"✓ {label}：{check()}")
        except Exception as e:  # 健康檢查要把每一項都跑完
            failed += 1
            print(f"✗ {label}：{type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


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


def cmd_refs(args) -> int:
    refs = json.loads(Path(args.file).read_text())
    refs = [r for r in refs if (r.get("phrase") or "").strip()]
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    quotedb.load_refs(db, refs)
    print(f"參考庫載入 {len(refs)} 句，目前共 {len(quotedb.ref_phrases(db))} 句")
    return 0


SCAN_BATCH = 25  # 每題帶前後文，批次放小一點


def _confirm(batch, model, around=None):
    raw = llm.ask(refaudit.confirm_prompt(batch, around), system=refaudit.CONFIRM_SYSTEM,
                         schema=refaudit.CONFIRM_SCHEMA, model=model)
    return {v["id"]: v for v in raw.get("verdicts", []) if isinstance(v.get("id"), int)}


def _puncts(text: str) -> list[str]:
    return sorted(ch for ch in text if not ch.isalnum() and not ch.isspace())


def _accept(s: refaudit.Suspect, v: dict | None) -> tuple[bool, str, str]:
    """模型說是誤聽，還要過跟 `bob fix` 同一道拼音門檻，才算數。"""
    if not v or not v.get("ok"):
        return False, "", ""
    wrong, right = (v.get("wrong") or "").strip(), (v.get("right") or "").strip()
    if wrong not in s.context and wrong.replace(" ", "\n") in s.context:
        # 送給模型時換行攤平成空白，它回傳的片段也是空白
        wrong, right = wrong.replace(" ", "\n"), right.replace(" ", "\n")
    ok = (wrong and right and wrong != right and wrong in s.context
          and abs(len(wrong) - len(right)) <= fix.MAX_LENGTH_CHANGE
          and _puncts(wrong) == _puncts(right)  # 只修字，不准順手加減標點
          and fix.sound_similarity(wrong, right) >= fix.SOUND_THRESHOLD)
    return bool(ok), wrong, right


FANDOM_SYSTEM = refaudit.CONFIRM_SYSTEM + """

另外附上本集的英文原版台詞（Fandom wiki 的 transcript），用來弄清楚這句在劇情裡說的是什麼。
中文配音是意譯，不會逐字對應英文：你仍然只修「聽錯的字」，修正前後唸起來必須相近，
不要照英文翻譯改寫，也不要因為英文沒有對應的句子就判定沒錯。"""


def _episode_transcript(db, stem: str, listing, stored: dict[str, str]) -> str:
    m = re.search(r"_ep(\d+)", stem)
    parts = []
    for name in fandom.episode_names(int(m.group(1)), listing) if m else []:
        if name not in stored:
            text = fandom.fetch_transcript(name)
            if not text:
                continue
            quotedb.save_transcript(db, name, text)
            stored[name] = text
        parts.append(f"## {name}\n{stored[name]}")
    return "\n\n".join(parts)


def _fandom_recheck(db, suspects, cache, around, args) -> None:
    """只看中文判斷不了的審計嫌疑，按集附上英文原版台詞再問一次。"""
    done = quotedb.fandom_checked(db)
    todo = [s for s in suspects if s.reason == "audit"
            and not cache.get((s.key[0], s.heard, s.proposed), (False,))[0]
            and (s.key[0], s.heard, s.proposed) not in done]
    by_stem: dict[str, list] = {}
    for s in todo:
        by_stem.setdefault(s.key[0], []).append(s)
    listing, stored = fandom.fetch_listing(), quotedb.transcripts(db)
    jobs = []
    for stem, items in sorted(by_stem.items()):
        transcript = _episode_transcript(db, stem, listing, stored)
        if not transcript:
            print(f"{stem}：Fandom 找不到英文台詞，略過", file=sys.stderr)
            continue
        jobs += [(transcript, list(enumerate(items[i:i + SCAN_BATCH])))
                 for i in range(0, len(items), SCAN_BATCH)]
    print(f"附英文原版台詞重新確認：{len(todo)} 個嫌疑、{len(by_stem)} 集、{len(jobs)} 批", flush=True)

    def ask(transcript, batch):
        prompt = f"英文原版台詞：\n{transcript}\n\n=== 待確認 ===\n\n" + refaudit.confirm_prompt(batch, around)
        raw = llm.ask(prompt, system=FANDOM_SYSTEM, schema=refaudit.CONFIRM_SCHEMA, model=args.model)
        return {v["id"]: v for v in raw.get("verdicts", []) if isinstance(v.get("id"), int)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ask, t, b): b for t, b in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            try:
                verdicts = fut.result()
            except llm.ERRORS as e:
                print(f"[fandom {n}/{len(jobs)}] 失敗：{e}", file=sys.stderr, flush=True)
                continue
            rows = []
            for i, s in batch:
                if i not in verdicts:
                    continue
                ok, wrong, right = _accept(s, verdicts[i])
                rows.append((s.key[0], s.heard, s.proposed, ok, wrong, right, "fandom"))
                cache[(s.key[0], s.heard, s.proposed)] = (ok, wrong, right)
            quotedb.record_scan(db, rows)
            print(f"[fandom {n}/{len(jobs)}] 確認 {sum(r[3] for r in rows)}/{len(rows)}  "
                  + "、".join(f"{r[4]}→{r[5]}" for r in rows if r[3])[:120], flush=True)


def cmd_scan(args) -> int:
    db = quotedb.connect(args.db)
    quotedb.init(db, quotes.load_taxonomy(TAXONOMY_PATH))
    rows = db.execute("SELECT e.stem, c.idx, c.text FROM cues c JOIN episodes e ON e.id = c.episode_id")
    cues = [((stem, idx), text) for stem, idx, text in rows if args.only in stem]
    text_of = {key: text for key, text in cues}
    titles = dict(db.execute("SELECT stem, title FROM episodes"))

    def around(key):
        stem, idx = key
        n = refaudit.CONTEXT_CUES
        return (titles.get(stem, ""),
                [text_of[(stem, j)] for j in range(idx - n, idx) if (stem, j) in text_of],
                [text_of[(stem, j)] for j in range(idx + 1, idx + n + 1) if (stem, j) in text_of])
    refs = quotedb.ref_phrases(db)
    found = refaudit.ref_suspects(cues, refs) + refaudit.homophone_suspects(cues)
    if args.audit:  # 標記金句時模型順手回報的字幕問題，也走同一道確認與拼音門檻
        issues = db.execute("""SELECT e.stem, a.cue_idx, a.detail, a.suggestion
            FROM audit_issues a JOIN episodes e ON e.id = a.episode_id
            WHERE a.type IN ('錯字', '人名', '歌詞')""")
        found += refaudit.audit_suspects([((stem, idx), text_of[(stem, idx)], detail, suggestion)
                                          for stem, idx, detail, suggestion in issues
                                          if (stem, idx) in text_of])
    seen, suspects = set(), []
    for s in found:
        k = (s.key[0], s.heard, s.proposed)
        if k not in seen:
            seen.add(k)
            suspects.append(s)
    cache = quotedb.scan_verdicts(db)
    todo = [s for s in suspects if (s.key[0], s.heard, s.proposed) not in cache]
    print(f"字幕 {len(cues)} 條、參考句 {len(refs)}：嫌疑 {len(suspects)}"
          f"（參考庫 {sum(s.reason == 'ref' for s in suspects)}、同音異寫 {sum(s.reason == 'homophone' for s in suspects)}"
          f"、審計 {sum(s.reason == 'audit' for s in suspects)}），"
          f"待模型確認 {len(todo)}", flush=True)
    if args.limit:
        todo = todo[:args.limit]

    batches = [list(enumerate(todo[i:i + SCAN_BATCH])) for i in range(0, len(todo), SCAN_BATCH)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_confirm, b, args.model, around): b for b in batches}
        for n, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            try:
                verdicts = fut.result()
            except llm.ERRORS as e:
                print(f"[{n}/{len(batches)}] 失敗：{e}", file=sys.stderr, flush=True)
                continue
            rows = []
            for i, s in batch:
                if i not in verdicts:  # 模型漏答的題目不記，下次重問
                    continue
                ok, wrong, right = _accept(s, verdicts.get(i))
                rows.append((s.key[0], s.heard, s.proposed, ok, wrong, right, s.reason))
                cache[(s.key[0], s.heard, s.proposed)] = (ok, wrong, right)
            quotedb.record_scan(db, rows)
            print(f"[{n}/{len(batches)}] 確認 {sum(r[3] for r in rows)}/{len(rows)}  "
                  + "、".join(f"{r[4]}→{r[5]}" for r in rows if r[3])[:120], flush=True)

    if args.fandom:
        _fandom_recheck(db, suspects, cache, around, args)

    # 確認過的修正延伸成至少三個字的片段（整集替換才不會誤傷），存成單集對照表
    by_stem: dict[str, dict[str, str]] = {}
    episode_text: dict[str, list[str]] = {}
    for (stem, _), text in cues:
        episode_text.setdefault(stem, []).append(text)
    for s in suspects:
        ok, wrong, right = cache.get((s.key[0], s.heard, s.proposed), (False, "", ""))
        pair = refaudit.widen(s.context, s.heard, wrong, right) if ok else None
        if pair and not refaudit.ambiguous_insertion(*pair, "\n".join(episode_text[s.key[0]])):
            by_stem.setdefault(s.key[0], {})[pair[0]] = pair[1]
    known = quotedb.episode_fixes(db)
    new_rows = [(stem, w, r, "scan") for stem, pairs in by_stem.items() for w, r in pairs.items()
                if (w, r) not in known.get(stem, [])]
    print(f"新增單集修正 {len(new_rows)} 筆、影響 {len({stem for stem, *_ in new_rows})} 集")
    # 寫回所有有修正的集數，已經套用過的不會變：中途失敗重跑就能補完
    stems = sorted(by_stem)
    if args.dry_run:
        for row in new_rows[:40]:
            print(f"  {row[0][-24:]}  {row[1]}→{row[2]}")
        return 0
    if new_rows:
        quotedb.add_episode_fixes(db, new_rows)
    if not stems:
        return 0
    service, files, _ = _drive_srts(args.folder)
    by_name = {f["name"].removesuffix(SRT_SUFFIX): f for f in files}
    state = quotedb.fix_state(db)
    for stem in stems:
        f = by_name.get(stem)
        if not f:
            continue
        current = drive.read_text(service, f["id"])
        cues, applied = fix.apply_glossary(parse(current), list(by_stem[stem].items()))
        new_text = render(cues)
        if new_text == current:
            continue
        md5_after = _write(service, f["name"], args.folder, new_text)
        if state.get(stem, (None,))[0] == f.get("md5Checksum"):
            quotedb.set_fixed_md5(db, stem, md5_after)
        print(f"{stem}：改 {len(applied)} 處  " + "、".join(f"{x.wrong}→{x.right}" for x in applied[:5]))
    print("接著跑 `bob index` 會就地更新台詞庫的文字。")
    return 0


def _clock(text: str) -> float:
    m, _, sec = text.rpartition(":")
    return int(m or 0) * 60 + float(sec)


def cmd_still(args) -> int:
    db = quotedb.connect(args.db)
    eps = db.execute("SELECT id, stem, season, episode, title FROM episodes WHERE stem LIKE ?",
                     (f"%{args.episode}%",))
    if len(eps) != 1:
        print(f"「{args.episode}」對到 {len(eps)} 集：{[e[1] for e in eps][:5]}，請寫得更精確")
        return 1
    ep_id, stem, season, episode, title = eps[0]
    cues = [tuple(r) for r in db.execute(
        "SELECT idx, start_s, end_s, text FROM cues WHERE episode_id = ? ORDER BY idx", (ep_id,))]
    if args.text:
        hit = still.match_cues(cues, args.text)
    else:
        at = _clock(args.at)
        hit = [c for c in cues if c[1] <= at < c[2]] or [min(cues, key=lambda c: abs(c[1] - at))]
    if not hit:
        print(f"{stem} 找不到含「{args.text}」的字幕")
        return 1
    start, end = hit[0][1], hit[-1][2]
    text = args.caption.replace(" / ", "\n") if args.caption else "\n".join(c[3] for c in hit)
    unit = args.title or title  # 兩個單元名時不猜：順序常與片中相反，也可能是同一集的兩個譯名
    source = still.source_line(args.speaker, season, episode, unit, start)

    service = build("drive", "v3", credentials=drive.load_credentials(TOKEN_PATH.read_text()),
                    cache_discovery=False)
    name = drive._escape_query_value(f"{stem}.mp4")
    videos = service.files().list(q=f"name = '{name}' and trashed = false", fields="files(id)",
                                  supportsAllDrives=True, includeItemsFromAllDrives=True).execute()["files"]
    if not videos:
        print(f"Drive 上找不到 {stem}.mp4")
        return 1
    token = drive.mint_readonly_access_token(drive.load_credentials(TOKEN_PATH.read_text()))
    times = still.frame_times(start, end, args.count, args.step, args.window)
    s = int(start)
    paths = still.render_stills(
        url=f"https://www.googleapis.com/drive/v3/files/{videos[0]['id']}?alt=media", token=token,
        times=times, lines=still.caption_lines(text), source=source, out_dir=Path(args.out),
        prefix=f"S{season:02d}ep{episode:03d}_{s // 60}m{s % 60:02d}s", font=args.font)
    print(f"{source}\n字幕：{text.replace(chr(10), ' / ')}")
    for i, (p, t) in enumerate(zip(paths, times), 1):
        print(f"  {i}. {t:7.2f}s  {p}")
    print(f"  挑選總覽：{paths[0].with_name(paths[0].name.rsplit('_', 1)[0] + '_contact.png')}")
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
        if name == "fix":
            p.add_argument("--quiet", action="store_true", help="不印修正範例（公開的 Actions log 用）")
    ask = sub.add_parser("ask", help="描述你的處境，找海綿寶寶台詞")
    ask.add_argument("situation", nargs="+", help="處境；可多給幾句改寫過的情境，一起合併排名")
    ask.add_argument("-n", "--top", type=int, default=10, help="列出幾段（不加 --pick 時）")
    ask.add_argument("--pick", action="store_true", help="從前 30 段再讓模型挑 1–5 段並說明理由")
    ask.add_argument("--rewrite", action="store_true", help="先讓模型把第一句改寫成三句使用情境再搜")
    ask.add_argument("--model", default="haiku")
    sub.add_parser("embed", help="補齊／更新金句的語意向量")
    ctx = sub.add_parser("ctx", help="列出某集某時間前後的字幕")
    ctx.add_argument("episode", help="集名片段，例如 ep161 或 月球遊記")
    ctx.add_argument("at", help="時間點 m:ss")
    ctx.add_argument("--span", type=float, default=30, help="前後各幾秒")
    grep = sub.add_parser("grep", help="字幕關鍵字搜尋（含任一詞），語意搜尋找不到具體名詞時用")
    grep.add_argument("terms", nargs="+")
    grep.add_argument("--limit", type=int, default=200)
    es = sub.add_parser("evalset", help="抽樣金句，讓模型出檢索評測題（只存 Turso）")
    es.add_argument("-n", type=int, default=200, help="抽幾段金句（每集至多一段）")
    es.add_argument("--seed", type=int, default=1)
    es.add_argument("--batch", type=int, default=8, help="每次呼叫出幾段的題")
    es.add_argument("--workers", type=int, default=4)
    es.add_argument("--model", default="opus")
    ev = sub.add_parser("eval", help="用評測題衡量 `bob ask` 的檢索命中率")
    ev.add_argument("--label", default="", help="記錄這次結果的名稱（例如 baseline），不填就不記錄")
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--mode", choices=("hybrid", "vec"), default="hybrid", help="vec＝只用語意向量（舊版）")
    ev.add_argument("--rewrite", default="", metavar="MODEL", help="每題先用這個模型改寫成三句情境（結果快取在 Turso）")
    refs = sub.add_parser("refs", help="載入台灣配音版台詞參考庫（JSON：[{phrase, kind, source}]）")
    refs.add_argument("file")
    scan = sub.add_parser("scan", help="用拼音比對參考庫與全庫同音異寫找誤聽，只把嫌疑交給模型確認")
    scan.add_argument("--folder", default=os.environ.get("DRIVE_FOLDER_ID", ""))
    scan.add_argument("--limit", type=int, default=0, help="本輪最多確認幾個嫌疑，0 = 全部")
    scan.add_argument("--workers", type=int, default=4)
    scan.add_argument("--model", default="haiku")
    scan.add_argument("--only", default="", help="只處理檔名含此字串的集數")
    scan.add_argument("--dry-run", action="store_true", help="只確認、不改寫 Drive")
    scan.add_argument("--audit", action="store_true", help="也確認標記時回報的字幕問題（audit_issues）")
    scan.add_argument("--fandom", action="store_true",
                      help="只看中文判斷不了的審計嫌疑，附 Fandom 英文原版台詞再確認一次（需搭配 --audit）")
    st = sub.add_parser("still", help="抓台詞畫面，下方加字幕與出處，一次出多張候選")
    st.add_argument("episode", help="集名片段，例如 ep161 或 月球遊記")
    where = st.add_mutually_exclusive_group(required=True)
    where.add_argument("--text", help="台詞片段（忽略標點，可跨句）")
    where.add_argument("--at", help="時間點 m:ss，取該時間的字幕")
    st.add_argument("--speaker", default="", help="說話者；字幕沒標，從畫面與前後文判斷後填入")
    st.add_argument("--title", default="", help="單元名；預設印完整集名（兩個單元名時用這個指定）")
    st.add_argument("--caption", default="", help="改寫字幕文字，用「 / 」換行")
    st.add_argument("--count", type=int, default=6, help="候選張數")
    st.add_argument("--step", type=float, default=0.5, help="候選間隔秒數")
    st.add_argument("--window", type=float, default=3.0, help="只在台詞前後幾秒內取")
    st.add_argument("--out", default=str(ROOT / "scratch" / "stills"), help="輸出資料夾（預設不進 git）")
    st.add_argument("--font", default=None, help="中文字型檔路徑")
    for name, help_text in (("backup", "整個金句庫備份成 SQLite，gzip 後放 Drive 的 _psr/backups/"),
                            ("health", "檢查 Drive token、Turso、DeepSeek 是否可用")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--folder", default=os.environ.get("DRIVE_FOLDER_ID", ""))
        if name == "backup":
            p.add_argument("--keep", type=int, default=8, help="保留最近幾份")
    audit = sub.add_parser("audit", help="彙整字幕審計結果與標籤提案")
    audit.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cmd in ("reseg", "fix", "index", "backup", "health") or (args.cmd == "scan" and not args.dry_run):
        args.folder = validate_folder_id(args.folder)
    return {"reseg": cmd_reseg, "fix": cmd_fix, "index": cmd_index, "ask": cmd_ask, "audit": cmd_audit,
            "refs": cmd_refs, "scan": cmd_scan, "still": cmd_still,
            "embed": cmd_embed, "ctx": cmd_ctx, "evalset": cmd_evalset, "eval": cmd_eval, "grep": cmd_grep,
            "backup": cmd_backup, "health": cmd_health}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
