"""在 Colab VM 上執行的轉錄工作。

這支檔案會被原樣送進 Colab VM 由它的直譯器執行，**不能 import 任何 psr
模組**，也不能假設 repo 的依賴存在。VM 上是 Python 3.12（runner 是 3.11），
所以只用兩者都有的語法。

設計上刻意讓 VM 直接從來源抓影片（Drive API 或 yt-dlp），而不是由 runner
下載後上傳：VM 到 Google 的頻寬好得多，而且 1–8GB 的影片檔完全不必經過
GitHub runner 那顆只有約 14GB 的磁碟。

VM 只拿到一個**唯讀**的短效 access token，從頭到尾沒有寫入權，也沒見過
refresh token。產物由 runner 用 colab download 拉回去再上傳。
"""

import json
import os
import subprocess
import sys
import time


def sh(*args):
    subprocess.run(list(args), check=True)


def pip(*pkgs):
    sh(sys.executable, "-m", "pip", "install", "-q", *pkgs)


def fetch_from_drive(file_id, token, dest):
    """用唯讀 access token 從 Drive 下載。分塊寫入，不整份讀進記憶體。"""
    import urllib.request

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)


def fetch_from_youtube(video_id, dest):
    pip("yt-dlp")
    sh("yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
       "-o", dest, f"https://www.youtube.com/watch?v={video_id}")


def extract_audio(src, dest):
    """16kHz 單聲道 MP3 @32kbps。

    Whisper 內部本來就重採樣到 16kHz，所以這個設定不損失任何辨識資訊，
    但體積極小——實測 87 分鐘影片只有 21MB，這讓整支音訊都還在 Groq
    免費層 25MB 上限內（fallback 路徑因此多數情況不必切塊）。
    """
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src,
       "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", dest)


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def load_model(model_size="large-v3"):
    """批次模式下同一個 kernel 會連續執行本檔數十次，模型掛在 builtins 上
    跨 exec 保留——每集重新載入 large-v3 要多花 30–60 秒。若 CLI 的 exec
    其實每次開新 kernel，這裡只是退化成每集載入一次，結果不受影響。"""
    import builtins

    cached = getattr(builtins, "_psr_whisper", None)
    if cached is not None and cached[0] == model_size:
        return cached[1]
    pip("faster-whisper==1.2.0")
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    builtins._psr_whisper = (model_size, model)
    return model


def transcribe(audio, prompt, model_size="large-v3"):
    model = load_model(model_size)
    segments, info = model.transcribe(
        audio,
        language="zh",
        initial_prompt=prompt or None,
        # 以下每一項都是為了確定性，不是為了品質，改動前請先讀設計文件 §5 鎖 2。
        beam_size=1,                       # 貪婪，不做 beam search
        temperature=0.0,                   # 單一值——預設會在壓縮率/logprob
                                           # 不合格時自動重試 0.2/0.4/0.6…
        condition_on_previous_text=False,  # 切斷前文影響：阻止幻覺迴圈擴散、
                                           # 錯誤不滾雪球，也讓切塊與整檔結果可比較
        vad_filter=True,
        word_timestamps=True,
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"text": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
    return words, info.duration


def main():
    cfg = json.load(open("/content/job.json"))
    # 批次模式：前一集的產物還在 VM 上。先清掉，免得這一集中途失敗時
    # runner 把上一集的 words.json 當成這一集的拉回去。
    for stale in ("/content/words.json", "/content/meta.json", "/content/source.mp4"):
        if os.path.exists(stale):
            os.remove(stale)
    t0 = time.time()

    video = "/content/source.mp4"
    if cfg["source_kind"] == "drive":
        fetch_from_drive(cfg["source_id"], cfg["access_token"], video)
    else:
        fetch_from_youtube(cfg["source_id"], video)
    t_fetch = time.time() - t0

    t = time.time()
    audio = "/content/audio.mp3"
    extract_audio(video, audio)
    duration = probe_duration(audio)
    os.remove(video)  # 單集最大近 1GB，轉完音訊就不需要了
    t_audio = time.time() - t

    t = time.time()
    words, _ = transcribe(audio, cfg.get("whisper_prompt", ""))
    t_asr = time.time() - t

    json.dump(words, open("/content/words.json", "w"), ensure_ascii=False)

    import torch
    meta = {
        "words": len(words),
        "audio_duration": duration,
        "gpu": torch.cuda.get_device_name(0),
        "seconds": {"fetch": round(t_fetch, 1), "audio": round(t_audio, 1), "asr": round(t_asr, 1)},
        "realtime_factor": round(duration / t_asr, 1) if t_asr else None,
    }
    json.dump(meta, open("/content/meta.json", "w"))
    print("REMOTE_JOB_OK " + json.dumps(meta))


if __name__ == "__main__":
    main()
