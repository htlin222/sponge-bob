"""Groq Whisper fallback——只在 Colab 拿不到 GPU 或超時時使用。

與 Colab 路徑不同，這裡影片必須先下載到 runner，所以磁碟壓力大得多
（runner 只有約 14GB 可用）。這也是它是備援而非主力的原因之一。
"""

import json
import math
import os
import pathlib
import subprocess

MODEL = "whisper-large-v3-turbo"
# 免費層單檔 25MB。實測 16kHz 單聲道 32kbps 約 14MB/小時，所以 100 分鐘
# 以內的影片整支送得進去，切塊邏輯多數情況根本不會啟用。
MAX_CHUNK_BYTES = 20 * 1024 * 1024


def extract_audio(video_path, dest):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "libmp3lame", "-b:a", "32k", str(dest)], check=True)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def silence_points(audio_path, want_chunks):
    """用 ffmpeg silencedetect 找靜音處當切點，避免切在字中間。

    回傳的是候選切點（秒），呼叫端再挑最接近等分位置的那些。
    """
    if want_chunks <= 1:
        return []
    r = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-af", "silencedetect=noise=-32dB:d=0.4",
         "-f", "null", "-"],
        capture_output=True, text=True)
    points = []
    for line in (r.stderr or "").splitlines():
        if "silence_end:" in line:
            try:
                points.append(float(line.split("silence_end:")[1].split()[0]))
            except (IndexError, ValueError):
                continue
    return points


def plan_chunks(audio_path):
    """回傳 [(start_s, end_s), ...]。單檔夠小就回傳單一區間。"""
    size = pathlib.Path(audio_path).stat().st_size
    duration = probe_duration(audio_path)
    if size <= MAX_CHUNK_BYTES:
        return [(0.0, duration)]

    n = math.ceil(size / MAX_CHUNK_BYTES)
    ideal = [duration * i / n for i in range(1, n)]
    candidates = silence_points(audio_path, n)
    cuts = []
    for target in ideal:
        if candidates:
            cuts.append(min(candidates, key=lambda p: abs(p - target)))
        else:
            cuts.append(target)          # 找不到靜音就硬切，總比不切好
    bounds = [0.0] + sorted(set(cuts)) + [duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def transcribe(audio_path, whisper_prompt, work_dir, client=None):
    """切塊 → 逐塊轉錄 → **加上時間偏移量**合併。

    偏移量是這裡唯一容易出錯的地方：每一塊的時間戳都從 0 開始，忘記加回
    起始秒數，後半段字幕就會整個疊在影片開頭。
    """
    if client is None:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    work_dir = pathlib.Path(work_dir)
    words = []
    for i, (start, end) in enumerate(plan_chunks(audio_path)):
        piece = work_dir / f"chunk{i:03d}.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
                        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-c", "copy", str(piece)], check=True)
        with open(piece, "rb") as fh:
            r = client.audio.transcriptions.create(
                file=(piece.name, fh.read()),
                model=MODEL,
                language="zh",
                prompt=whisper_prompt or None,
                temperature=0,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        payload = r if isinstance(r, dict) else json.loads(r.model_dump_json())
        for w in payload.get("words", []):
            words.append({
                "text": w["word"],
                "start": round(w["start"] + start, 3),
                "end": round(w["end"] + start, 3),
            })
    return words
