"""比較另一個模型標的金句與 Turso 裡現有的（Haiku 標的），決定能不能換模型。

只印統計數字、不印台詞：這支在 GitHub Actions 上跑（DeepSeek key 只在 secret 裡），
而公開 repo 的 log 人人看得到。不寫入 Turso。

    uv run python scripts/compare_index.py ep161 ep098 ep194 --model deepseek-v4-flash
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from googleapiclient.discovery import build

from psr import drive, llm, quotedb, quotes
from psr.bob import SRT_SUFFIX, TAXONOMY_PATH
from psr.cli import TOKEN_PATH
from psr.srt import parse


def overlaps(span, spans) -> bool:
    return any(a <= span[1] and span[0] <= b for a, b in spans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="集名片段，例如 ep161")
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    db = quotedb.connect()
    taxonomy = quotes.load_taxonomy(TAXONOMY_PATH)
    service = build(
        "drive",
        "v3",
        credentials=drive.load_credentials(TOKEN_PATH.read_text()),
        cache_discovery=False,
    )
    jobs = []
    for frag in args.episodes:
        rows = db.execute(
            "SELECT id, stem, drive_id FROM episodes WHERE stem LIKE ?", (f"%{frag}%",)
        )
        if len(rows) != 1:
            print(f"{frag}：對到 {len(rows)} 集，略過")
            continue
        ep_id, stem, drive_id = rows[0]
        old = db.execute(
            "SELECT cue_from, cue_to, quotability FROM quotes WHERE episode_id = ?",
            (ep_id,),
        )
        jobs.append((stem, parse(drive.read_text(service, drive_id)), old))

    def tag(job):
        stem, cues, _ = job
        meta = quotes.parse_episode_name(stem + SRT_SUFFIX)
        raw = llm.ask(
            quotes.user_prompt(meta, cues),
            system=quotes.system_prompt(args.model),
            schema=quotes.output_schema(taxonomy),
            model=args.model,
        )
        return quotes.accept(raw, cues, taxonomy)

    totals = {"old": 0, "new": 0, "old_hit": 0, "new_hit": 0, "rejected": 0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (stem, _, old), result in zip(jobs, pool.map(tag, jobs)):
            old_spans = [(a, b) for a, b, _ in old]
            new_spans = [(q.cue_from, q.cue_to) for q in result.quotes]
            old_hit = sum(overlaps(s, new_spans) for s in old_spans)
            new_hit = sum(overlaps(s, old_spans) for s in new_spans)
            print(
                f"{stem.split('_')[2]}  Haiku {len(old)} 段（★4+ {sum(q >= 4 for *_, q in old)}）"
                f"｜{args.model} {len(new_spans)} 段（★4+ {sum(q.quotability >= 4 for q in result.quotes)}）"
                f"｜Haiku 的有 {old_hit} 段被涵蓋、新模型的有 {new_hit} 段與 Haiku 重疊"
                f"｜丟棄 {result.rejected}｜情境平均 "
                f"{sum(len(q.use_when) for q in result.quotes) / max(len(result.quotes), 1):.0f} 字"
            )
            for k, v in (
                ("old", len(old)),
                ("new", len(new_spans)),
                ("old_hit", old_hit),
                ("new_hit", new_hit),
                ("rejected", result.rejected),
            ):
                totals[k] += v
    if totals["old"] and totals["new"]:
        print(
            f"合計：Haiku {totals['old']} 段、{args.model} {totals['new']} 段；"
            f"Haiku 的 {totals['old_hit'] / totals['old']:.0%} 被涵蓋，"
            f"新模型的 {totals['new_hit'] / totals['new']:.0%} 與 Haiku 重疊；丟棄 {totals['rejected']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
