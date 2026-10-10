"""整個 Drive 資料夾的批次字幕。

與單支影片的 `psr run` 共用同一條 pipeline（轉錄 → 加標點 → 對回時間 →
修整 → 驗證），差別在編排：

- **Drive 的狀態就是待辦清單**。影片旁已有 `<集名>.zh-Hant.srt` 就跳過，
  不需要任何外部資料庫。排程每次重跑都從「還沒有 SRT 的集數」繼續。
- **轉錄結果先落地**。ASR 一完成就把 words.json 與 manifest 寫進 `_psr/`，
  之後加標點或上傳失敗時，下一輪直接重用而不必再搶一次 GPU。
- **一個 Colab session 跑一整批**，見 `psr.asr.colab.ColabSession`。
- **單集失敗不拖垮整批**，但連續失敗代表系統性問題（GPU 配額用完、API key
  失效），這時停下來比燒光剩下的時間預算有用。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import tempfile
import time
from dataclasses import dataclass

from psr import cleanup, drive, glossary as glossary_mod, manifest as manifest_mod
from psr.asr.colab import ColabSession, ColabUnavailable
from psr.cli import ASR_STAGE_VERSION, PUNCTUATE_STAGE_VERSION, _build_cues, _punctuate
from psr import punctuate as punct_mod
from psr.models import Word
from psr.segment import raw_segment
from psr.srt import render
from psr.timeline import coverage
from psr.validate import validate
from psr.youtube import drive_paths

WORK_SUBFOLDER = "_psr"
# 連續這麼多集失敗就停止整批。
MAX_CONSECUTIVE_FAILURES = 3
_FOLDER_ID = re.compile(r"^[A-Za-z0-9_-]{10,100}$")


class BatchError(ValueError):
    """批次設定錯誤（資料夾 ID 不合法等），在碰任何 API 之前就失敗。"""


@dataclass(frozen=True)
class Episode:
    id: str
    name: str
    stem: str
    md5: str
    cached_words_id: str | None = None
    cached_manifest_id: str | None = None


def validate_folder_id(folder_id: str) -> str:
    """資料夾 ID 來自 workflow 輸入，進 Drive 查詢字串之前先嚴格檢查。"""
    folder_id = (folder_id or "").strip()
    if not _FOLDER_ID.fullmatch(folder_id):
        raise BatchError("資料夾 ID 格式不對：只能包含英數、底線與連字號。")
    return folder_id


def plan(children: list[dict], work_children: list[dict], limit: int = 0) -> list[Episode]:
    """決定這一輪要處理哪些集數。純函式，依檔名排序以確保順序可重現。

    `children` 是影片資料夾的內容，`work_children` 是 `_psr/` 的內容。
    """
    present = {f["name"] for f in children}
    work = {f["name"]: f["id"] for f in work_children}
    pending: list[Episode] = []
    for f in sorted(children, key=lambda f: f["name"]):
        if not f.get("mimeType", "").startswith("video/"):
            continue
        stem = f["name"].rsplit(".", 1)[0]
        names = drive_paths(stem)
        if names["srt"] in present:
            continue
        pending.append(Episode(
            id=f["id"], name=f["name"], stem=stem, md5=f.get("md5Checksum", ""),
            cached_words_id=work.get(names["words"]),
            cached_manifest_id=work.get(names["manifest"]),
        ))
    return pending[:limit] if limit > 0 else pending


class Transcriber:
    """Colab 為主，有 GROQ_API_KEY 才有 Groq 備援。session 在第一次需要時才開。"""

    def __init__(self, creds, service, prompt: str):
        self.creds = creds
        self.service = service
        self.prompt = prompt
        self.session: ColabSession | None = None
        self.groq_ok = bool(os.environ.get("GROQ_API_KEY"))

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def _colab(self, ep: Episode, work: str):
        if self.session is None:
            self.session = ColabSession(episode_timeout_s=1800).open()
        token = drive.mint_readonly_access_token(self.creds)
        return self.session.transcribe("drive", ep.id, token, self.prompt, work)

    def __call__(self, ep: Episode, work: str):
        # VM 偶爾會被斷線；重開一次 session 再試。第二次還失敗多半是配額用完。
        last: ColabUnavailable | None = None
        for _ in range(2):
            try:
                words, meta = self._colab(ep, work)
                return words, meta, "colab"
            except ColabUnavailable as exc:
                last = exc
                print(f"[colab] {ep.stem}：{exc}", file=sys.stderr)
                self.close()
        if not self.groq_ok:
            raise last
        return self._groq(ep, work)

    def _groq(self, ep: Episode, work: str):
        from psr.asr import groq as groq_asr

        video = pathlib.Path(work) / "source.mp4"
        drive.download(self.service, ep.id, str(video))
        audio = pathlib.Path(work) / "audio.mp3"
        groq_asr.extract_audio(video, audio)
        video.unlink()
        words = groq_asr.transcribe(audio, self.prompt, work)
        return words, {"audio_duration": groq_asr.probe_duration(audio)}, "groq"


def _upload(service, name, folder_id, work, payload):
    p = pathlib.Path(work) / name
    p.write_text(payload, encoding="utf-8")
    return drive.find_or_create(service, name, folder_id, str(p), "text/plain")


def process_episode(ep: Episode, *, service, transcriber, gloss, llm,
                    folder_id: str, work_folder_id: str | None, dry_run: bool) -> dict:
    started = time.time()
    names = drive_paths(ep.stem)
    man = manifest_mod.Manifest(source=f"drive:{ep.id}", source_md5=ep.md5)
    man.stage_keys["asr"] = manifest_mod.stage_key(
        "asr", ASR_STAGE_VERSION, [ep.md5 or ep.id],
        {"engine": "faster-whisper-large-v3", "prompt": gloss.whisper_prompt()})

    with tempfile.TemporaryDirectory() as work:
        words_raw = None
        if ep.cached_words_id and ep.cached_manifest_id:
            old = manifest_mod.Manifest.from_json(drive.read_text(service, ep.cached_manifest_id))
            if old.is_complete("asr", man.stage_keys["asr"]):
                words_raw = json.loads(drive.read_text(service, ep.cached_words_id))
                meta = {"audio_duration": old.timings.get("audio_duration")}
                engine = f"{old.engine}（重用）"
                man.gpu_model = old.gpu_model
        if words_raw is None:
            words_raw, meta, engine = transcriber(ep, work)
            man.gpu_model = meta.get("gpu", "")
            man.timings.update(meta.get("seconds", {}))

        words = [Word(w["text"], w["start"], w["end"]) for w in words_raw]
        audio_duration = float(meta.get("audio_duration") or (words[-1].end if words else 0.0))
        man.engine = engine
        man.timings["audio_duration"] = audio_duration

        if not dry_run and not engine.endswith("（重用）"):
            # 先把轉錄落地：最貴的一步完成了，後面任何一步失敗都不該讓它白做。
            _upload(service, names["words"], work_folder_id, work,
                    json.dumps(words_raw, ensure_ascii=False))
            _upload(service, names["manifest"], work_folder_id, work, man.to_json())

        # 清理只作用在送去加標點的文字上；words.json 與 raw.srt 保留原始轉錄，
        # 之後統計誤聽、補術語表要看的正是它們。
        clean, dropped = cleanup.drop_hallucinations(words, list(gloss.hallucinations))
        clean, corrected = cleanup.apply_corrections(clean, gloss.corrections())
        clean, looped = cleanup.collapse_repeats(clean)
        dropped += looped

        man.stage_keys["punctuate"] = manifest_mod.stage_key(
            "punctuate", PUNCTUATE_STAGE_VERSION,
            [man.stage_keys["asr"], gloss.content_hash()], {"model": punct_mod.MODEL})
        punctuated, failed, total_chunks, ptok, ctok = _punctuate(clean, llm)
        cues = _build_cues(clean, punctuated, audio_duration)

        man.degraded_window_count = len(failed)
        man.cost = round(ptok / 1e6 * 0.14 + ctok / 1e6 * 0.28, 4)
        man.timings["total"] = round(time.time() - started, 1)
        violations = validate(cues, audio_duration)
        srt_text = render(cues)

        if not dry_run:
            _upload(service, names["raw_srt"], work_folder_id, work, render(raw_segment(words)))
            _upload(service, names["manifest"], work_folder_id, work, man.to_json())
            # 順序即 checkpoint：最終 SRT 最後上傳，它的存在代表這集完成。
            _upload(service, names["srt"], folder_id, work, srt_text)

    return {
        "stem": ep.stem, "status": "ok", "engine": engine,
        "minutes": audio_duration / 60, "cues": len(cues),
        "failed_chunks": f"{len(failed)}/{total_chunks}",
        "coverage": coverage(clean, cues), "violations": len(violations),
        "dropped": dropped, "corrected": corrected,
        "cost": man.cost, "seconds": man.timings["total"],
        "preview": "\n".join(srt_text.split("\n\n")[:6]) if dry_run else "",
    }


def _report(results, pending_total, remaining, stop_reason, dry_run) -> str:
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    lines = [
        f"### 批次字幕{'（dry-run，未寫入 Drive）' if dry_run else ''}",
        "",
        f"- 本輪待處理 {pending_total} 集：完成 **{len(ok)}**、失敗 {len(bad)}、"
        f"未開始 {remaining}",
        f"- 潤稿成本：約 ${sum(r['cost'] for r in ok):.4f}",
    ]
    if stop_reason:
        lines.append(f"- 提前停止：{stop_reason}")
    if ok:
        lines += ["", "| 集數 | 引擎 | 分鐘 | 字幕 | 標點失敗 | 覆蓋率 | 違規 | 幻覺刪除 | 名稱修正 | 秒 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['stem']} | {r['engine']} | {r['minutes']:.1f} | {r['cues']} | "
                  f"{r['failed_chunks']} | {r['coverage'] * 100:.1f}% | {r['violations']} | "
                  f"{r['dropped']} | {r['corrected']} | {r['seconds']:.0f} |" for r in ok]
    if bad:
        lines += ["", "失敗："] + [f"- {r['stem']}：{r['error']}" for r in bad]
    previews = [r for r in ok if r["preview"]]
    if previews:
        lines += ["", f"`{previews[0]['stem']}` 前 6 條預覽：", "", "```",
                  previews[0]["preview"], "```"]
    return "\n".join(lines) + "\n"


def run_batch(folder_id: str, report_path: str, *, limit: int = 0,
              budget_s: float = 5 * 3600, dry_run: bool = False,
              token_path: pathlib.Path) -> int:
    from googleapiclient.discovery import build
    from openai import OpenAI

    folder_id = validate_folder_id(folder_id)
    started = time.time()
    creds = drive.load_credentials(token_path.read_text(encoding="utf-8"))
    service = build("drive", "v3", credentials=creds)
    gloss = glossary_mod.load("glossary.yml")
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    children = drive.list_children(service, folder_id)
    work_meta = next((f for f in children if f["name"] == WORK_SUBFOLDER
                      and f["mimeType"] == "application/vnd.google-apps.folder"), None)
    work_children = drive.list_children(service, work_meta["id"]) if work_meta else []
    pending = plan(children, work_children, limit)
    print(f"待處理 {len(pending)} 集", flush=True)

    work_folder_id = None
    if pending and not dry_run:
        work_folder_id = work_meta["id"] if work_meta else drive.ensure_folder(
            service, WORK_SUBFOLDER, folder_id)

    transcriber = Transcriber(creds, service, gloss.whisper_prompt())
    results: list[dict] = []
    stop_reason = ""
    consecutive = 0
    try:
        for i, ep in enumerate(pending):
            if time.time() - started > budget_s:
                stop_reason = "用完本輪時間預算，下一輪排程會接著做"
                break
            print(f"[{i + 1}/{len(pending)}] {ep.stem}", flush=True)
            try:
                r = process_episode(ep, service=service, transcriber=transcriber, gloss=gloss,
                                    llm=llm, folder_id=folder_id,
                                    work_folder_id=work_folder_id, dry_run=dry_run)
                consecutive = 0
            except ColabUnavailable as exc:
                results.append({"stem": ep.stem, "status": "failed", "error": f"Colab：{exc}"})
                stop_reason = "Colab 連續拿不到 GPU（多半是免費層配額用完），等下一輪排程"
                break
            except Exception as exc:  # 單集失敗不拖垮整批
                r = {"stem": ep.stem, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                consecutive += 1
            results.append(r)
            print(f"    → {r['status']}", flush=True)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                stop_reason = f"連續 {consecutive} 集失敗，疑似系統性問題，先停下來"
                break
    finally:
        transcriber.close()

    remaining = len(pending) - len(results)
    text = _report(results, len(pending), remaining, stop_reason, dry_run)
    pathlib.Path(report_path).write_text(text, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 1 if any(r["status"] != "ok" for r in results) else 0
