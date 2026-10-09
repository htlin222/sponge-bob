"""Colab CLI 編排——主轉錄路徑。

實測（2026-08-16）：`colab run --gpu T4` 從配置到釋放全程 19.8 秒，
87 分鐘音訊在 T4 上轉錄 397 秒（13.2 倍實時）。

批次模式（228 集卡通）下改成**一個 session 跑完一整批**：每集重新配置 T4
的固定成本約 20 秒加模型載入，乘上兩百多集就是一個多小時的純浪費，而且
免費層頻繁 new/stop 更容易撞上 TooManyAssignmentsError。
"""

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

REMOTE_JOB = pathlib.Path(__file__).with_name("remote_job.py")


class RemoteJobFailed(RuntimeError):
    """遠端工作本身丟了 Python 例外——是程式錯誤，不是基礎設施問題。

    與 ColabUnavailable 分開：重開 session 或換 Groq 都救不了程式錯誤，
    把它當成「拿不到 GPU」只會浪費一次重試，還把真正原因藏在錯的訊息後面
    （2026-10-09 第一次 dry-run 就是這樣：PyAV 不相容被回報成配額用完）。
    """


class ColabUnavailable(RuntimeError):
    """基礎設施層失敗——配不到 GPU、VM 被斷、超過硬超時。

    只有這一類才該觸發 Groq fallback。程式邏輯錯誤（ffmpeg 死、檔案找不到）
    必須直接失敗，否則會被 fallback 掩蓋成「反正 Groq 跑得出來」，
    而 Colab 這條路壞掉了你永遠不會知道。
    """


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# IPython 風格的例外結尾，例如 "TypeError: open() got an unexpected keyword"。
_PY_EXCEPTION = re.compile(r"^\w+(?:Error|Exception): ", re.MULTILINE)


def _tail(text, n=600):
    """取錯誤輸出的**尾端**並剝掉 ANSI。

    colab CLI 用 rich 印 traceback，前面幾百字元全是框線與檔案路徑，真正的
    例外訊息在最後。取前 300 字元會把唯一有用的資訊丟掉——這在第一次 CI
    執行時實際發生過，導致失敗原因無法判讀。
    """
    return _ANSI.sub("", text or "").strip()[-n:]


def _colab(*args, timeout=None):
    # --auth adc 必須顯式指定：CLI 的預設是 oauth2（會開瀏覽器），
    # 在 CI 裡只有 ADC 能無頭運作。
    cmd = ["colab", "--auth", "adc", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class ColabSession:
    """一台 T4 runtime，可連續轉錄多支影片。

    VM 只收到唯讀 access token，產物由這裡拉回來，上傳一律在 runner 端做。
    用法：`with ColabSession() as s: s.transcribe(...)`；離開時一定 stop，
    workflow 另有 `if: always()` 的回收步驟作縱深防禦。
    """

    def __init__(self, name="psr", episode_timeout_s=1800):
        self.name = name
        self.episode_timeout_s = episode_timeout_s
        self.opened = False

    def open(self):
        if shutil.which("colab") is None:
            raise ColabUnavailable("找不到 colab CLI")
        # 先回收同名的殘留 session。Colab 免費層對同時配置的 runtime 有上限，
        # 超過就是 TooManyAssignmentsError——實測第一次 CI 執行正是這樣掛的。
        _colab("stop", "-s", self.name, timeout=120)
        try:
            r = _colab("new", "-s", self.name, "--gpu", "T4", timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise ColabUnavailable("配置 T4 超時") from exc
        if r.returncode != 0:
            raise ColabUnavailable(f"無法配置 T4：{_tail(r.stderr or r.stdout)}")
        self.opened = True
        return self

    def close(self):
        if self.opened:
            _colab("stop", "-s", self.name, timeout=300)
            self.opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def transcribe(self, source_kind, source_id, access_token, whisper_prompt, out_dir):
        """轉錄一支影片，回傳 (words, meta)。session 必須已開啟。"""
        out_dir = pathlib.Path(out_dir)
        timeout = self.episode_timeout_s
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = pathlib.Path(tmp) / "job.json"
                job.write_text(json.dumps({
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "access_token": access_token,
                    "whisper_prompt": whisper_prompt,
                }), encoding="utf-8")
                up = _colab("upload", "-s", self.name, str(job), "/content/job.json", timeout=300)
                if up.returncode != 0:
                    raise ColabUnavailable(f"上傳工作設定失敗：{_tail(up.stderr)}")

                # --timeout 預設只有 30 秒，轉錄一定要顯式加大。
                run = _colab("exec", "-s", self.name, "--timeout", str(timeout),
                             "-f", str(REMOTE_JOB), timeout=timeout + 300)
                if "REMOTE_JOB_OK" not in (run.stdout or ""):
                    detail = _tail(run.stderr or run.stdout)
                    if _PY_EXCEPTION.search(detail):
                        raise RemoteJobFailed(f"遠端工作丟出例外：{detail}")
                    raise ColabUnavailable(f"遠端工作未完成：{detail}")

                for remote, local in (("/content/words.json", out_dir / "words.json"),
                                      ("/content/meta.json", out_dir / "meta.json")):
                    d = _colab("download", "-s", self.name, remote, str(local), timeout=600)
                    if d.returncode != 0:
                        raise ColabUnavailable(f"取回 {remote} 失敗：{_tail(d.stderr)}")
        except subprocess.TimeoutExpired as exc:
            raise ColabUnavailable(f"超過硬超時 {timeout}s") from exc

        words = json.loads((out_dir / "words.json").read_text(encoding="utf-8"))
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        return words, meta


def transcribe(source_kind, source_id, access_token, whisper_prompt,
               out_dir, session="psr", hard_timeout_s=3600):
    """單支影片：開一個 session、轉錄、關掉。"""
    with ColabSession(session, hard_timeout_s) as s:
        return s.transcribe(source_kind, source_id, access_token, whisper_prompt, out_dir)
