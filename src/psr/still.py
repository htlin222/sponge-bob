"""`bob still`：從 Drive 上的影片抓台詞畫面，下方加字幕與出處，一次出多張讓人挑。"""

from __future__ import annotations

import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WIDTH = 1280
FONTS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)
_NOISE = re.compile(r"[\s\W_]+")


def frame_times(
    start: float, end: float, count: int = 6, step: float = 0.5, window: float = 3.0
) -> list[float]:
    """以台詞中點為中心、間距 `step` 排 `count` 個時間點；全部落在台詞前後 `window` 秒內，
    排不下就縮小間距均分。"""
    if count == 1:
        return [round((start + end) / 2, 2)]
    lo, hi = max(0.0, start - window), end + window
    step = min(step, (hi - lo) / (count - 1))
    first = (start + end) / 2 - step * (count - 1) / 2
    first = min(max(first, lo), hi - step * (count - 1))
    return [round(first + i * step, 2) for i in range(count)]


def segment_title(title: str, t: float, total: float) -> str:
    """一集有多個單元（「月球遊記／蟹老闆渡假去」）時，依時間點落在第幾段取單元名。"""
    parts = title.split("／")
    return (
        parts[min(int(t / total * len(parts)), len(parts) - 1)]
        if total > 0
        else parts[0]
    )


def caption_lines(text: str) -> list[str]:
    lines = [ln for ln in text.split("\n") if ln]
    lines = [f"「{lines[0]}"] + [f"　{ln}" for ln in lines[1:]]
    lines[-1] += "」"
    return lines


def source_line(
    speaker: str, season: int, episode: int, title: str, start: float
) -> str:
    who = f"{speaker}｜" if speaker else ""
    s = int(start)
    return f"—— {who}《海綿寶寶》S{season:02d} ep{episode:03d}〈{title}〉{s // 60}:{s % 60:02d}"


def match_cues(
    cues: list[tuple[int, float, float, str]], needle: str
) -> list[tuple[int, float, float, str]]:
    """找出含有 `needle` 的連續字幕（忽略標點、空白、換行，可跨句）。"""
    want = _NOISE.sub("", needle)
    spans, joined = [], ""
    for cue in cues:
        norm = _NOISE.sub("", cue[3])
        spans.append((len(joined), len(joined) + len(norm)))
        joined += norm
    at = joined.find(want) if want else -1
    if at < 0:
        return []
    return [c for c, (a, b) in zip(cues, spans) if a < at + len(want) and b > at]


def grab(url: str, token: str, t: float, dest: Path) -> Path:
    """用 HTTP range 直接從 Drive 串流取單張畫面，不下載整部影片。"""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-headers",
            f"Authorization: Bearer {token}\r\n",
            "-ss",
            f"{t:.2f}",
            "-i",
            url,
            "-frames:v",
            "1",
            str(dest),
        ],
        check=True,
    )
    return dest


def compose(frame: Path, lines: list[str], source: str, font_path: str | None = None):
    from PIL import Image, ImageDraw, ImageFont

    font_path = font_path or next((f for f in FONTS if Path(f).exists()), None)
    if not font_path:
        raise SystemExit("找不到中文字型，請用 --font 指定")
    im = Image.open(frame).convert("RGB")
    im = im.resize((WIDTH, im.height * WIDTH // im.width), Image.LANCZOS)
    big, small = ImageFont.truetype(font_path, 46), ImageFont.truetype(font_path, 28)
    margin, pad, gap = 70, 44, 16
    lines = [part for ln in lines for part in _wrap(ln, big, WIDTH - 2 * margin)]
    lh = big.getbbox("永")[3] + gap
    panel = pad + lh * len(lines) + 18 + small.getbbox("永")[3] + pad
    canvas = Image.new("RGB", (WIDTH, im.height + panel), (20, 22, 30))
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, im.height, WIDTH, im.height + 6], fill=(255, 214, 10))
    y = im.height + pad
    for ln in lines:
        draw.text((margin, y), ln, font=big, fill=(255, 255, 255))
        y += lh
    draw.text(
        (WIDTH - margin, y + 18), source, font=small, fill=(180, 186, 200), anchor="ra"
    )
    return canvas


def _wrap(line: str, font, width: int) -> list[str]:
    out, cur = [], ""
    for ch in line:
        if cur and font.getlength(cur + ch) > width:
            out.append(cur)
            cur = "　" + ch
        else:
            cur += ch
    return out + [cur]


def contact_sheet(images, times: list[float], dest: Path, cols: int = 3) -> Path:
    """把候選圖排成編號格，讓人一眼挑。"""
    from PIL import Image, ImageDraw, ImageFont

    thumbs = [im.resize((WIDTH // 2, im.height // 2)) for im in images]
    w, h = thumbs[0].size
    rows = -(-len(thumbs) // cols)
    sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font_path = next((f for f in FONTS if Path(f).exists()), None)
    font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()
    for i, (th, t) in enumerate(zip(thumbs, times)):
        x, y = i % cols * w, i // cols * h
        sheet.paste(th, (x, y))
        label = f" {i + 1}  {int(t) // 60}:{t % 60:04.1f} "
        draw.rectangle([x, y, x + font.getlength(label), y + 42], fill=(255, 214, 10))
        draw.text((x, y + 2), label, font=font, fill=(0, 0, 0))
    sheet.save(dest)
    return dest


def render_stills(
    *,
    url: str,
    token: str,
    times: list[float],
    lines: list[str],
    source: str,
    out_dir: Path,
    prefix: str,
    font: str | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=len(times)) as pool:
            frames = list(
                pool.map(
                    lambda t: grab(url, token, t, Path(tmp) / f"{t:.2f}.png"), times
                )
            )
        images = [compose(f, lines, source, font) for f in frames]
    paths = []
    for i, im in enumerate(images, 1):
        path = out_dir / f"{prefix}_{i}.png"
        im.save(path)
        paths.append(path)
    contact_sheet(images, times, out_dir / f"{prefix}_contact.png")
    return paths
