#!/usr/bin/env python3
"""Render the clef-evals showcase video (brag.mp4) and cover still (brag.jpg).

Deterministic PIL frame rendering piped to ffmpeg. Style: warm bone canvas,
charcoal text, one muted terracotta accent, 1px hairline cards. See
composition-brief.md for the scene plan.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30
BONE = "#F7F6F3"
CHARCOAL = "#2F3437"
MUTED = "#787774"
HAIRLINE = "#EAEAEA"
ACCENT = "#D97A3D"
CARD = "#FFFFFF"

SANS = "C:/Windows/Fonts/arialbd.ttf"
SANS_REG = "C:/Windows/Fonts/arial.ttf"
MONO = "C:/Windows/Fonts/consola.ttf"

HERE = Path(__file__).resolve().parent


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def alpha(hex_color: str, a: float) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, int(a * 255))


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), BONE)
    return img, ImageDraw.Draw(img, "RGBA")


def draw_card(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], a: float = 1.0) -> None:
    d.rounded_rectangle(box, radius=8, fill=alpha(CARD, a), outline=alpha(HAIRLINE, a), width=1)


def text_at(d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, f: ImageFont.FreeTypeFont,
            color: str = CHARCOAL, a: float = 1.0) -> None:
    d.text(xy, s, font=f, fill=alpha(color, a))


def scene_title(img: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
    a = ease(t / 0.7)
    dot_r = 6 + 2 * math.sin(t * 4)
    text_at(d, (90, 250), "clef-evals", font(SANS, 108), CHARCOAL, a)
    text_at(d, (94, 392), "Judge cheap, audit confidence.", font(SANS_REG, 40), MUTED, ease((t - 0.35) / 0.7))
    d.ellipse([94, 470 - dot_r, 94 + 2 * dot_r, 470 + dot_r], fill=alpha(ACCENT, a))
    text_at(d, (120, 460), "accuracy · ECE · Brier · latency · cost", font(MONO, 24), MUTED, ease((t - 0.6) / 0.7))


def card_box(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return (x, y, x + w, y + h)


def scene_pipeline(img: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
    title_a = ease(t / 0.4)
    text_at(d, (90, 96), "One client, typed errors, retries built in", font(SANS, 44), CHARCOAL, title_a)
    cards = [
        (90, 210, 340, 120, "your eval set", "state + instructions", "criteria + gold"),
        (470, 210, 340, 120, "ClefJudge", "sync + async", "choice · binary · custom"),
        (850, 210, 340, 120, "clef-eval CLI", "run --min-accuracy", "--max-ece"),
    ]
    for i, (x, y, w, h, title, l1, l2) in enumerate(cards):
        a = ease((t - 0.25 * i - 0.2) / 0.6)
        if a <= 0:
            continue
        draw_card(d, card_box(x, y, x + w, y + h), a)
        text_at(d, (x + 24, y + 20), title, font(SANS, 28), CHARCOAL, a)
        text_at(d, (x + 24, y + 62), l1, font(MONO, 19), MUTED, a)
        text_at(d, (x + 24, y + 88), l2, font(MONO, 19), MUTED, a)

    a = ease((t - 1.0) / 0.6)
    if a > 0:
        draw_card(d, card_box(290, 420, 700, 110), a)
        text_at(d, (314, 440), "ClefClient", font(SANS, 30), CHARCOAL, a)
        text_at(d, (314, 484), "retries · backoff + jitter · Retry-After · typed errors",
                font(MONO, 21), MUTED, a)
        # traveling dots along the three wires
        for i, x_start in enumerate((260, 640, 1020)):
            prog = ((t - 1.2 - 0.3 * i) / 1.4) % 1.0
            if 0 <= prog <= 1:
                y = 330 + (420 - 330) * ease(prog)
                x = x_start + (640 - x_start) * min(1.0, prog * 1.6)
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=alpha(ACCENT, min(a, 1.0) * (1 - abs(prog - 0.5) * 0.6)))
        text_at(d, (290, 560), "POST /ai/run/@cf/cloudflare/clef -> probabilities + usage",
                font(MONO, 22), CHARCOAL, ease((t - 1.6) / 0.6))


BARS = [("BFCL case exact", 98.5, 38.1), ("BANKING77 macro-F1", 94.2, 14.3), ("CLINC150+OOS macro-F1", 97.4, 3.2)]


def scene_benchmarks(img: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
    text_at(d, (90, 90), "Published quality gap", font(SANS, 46), CHARCOAL, ease(t / 0.4))
    text_at(d, (90, 150), "Cloudflare Decision Index 0.2.1 (model card, huggingface.co/Cloudflare/clef)",
            font(MONO, 19), MUTED, ease(t / 0.4))
    for i, (label, clef, laya) in enumerate(BARS):
        row_t = ease((t - 0.35 - 0.45 * i) / 0.7)
        if row_t <= 0:
            continue
        y = 230 + i * 130
        text_at(d, (90, y), label, font(SANS, 26), CHARCOAL, row_t)
        full = 760
        d.rounded_rectangle((90 + 330, y + 4, 90 + 330 + int(full * clef / 100) * row_t, y + 30),
                            radius=6, fill=alpha(CHARCOAL, 0.9))
        d.rounded_rectangle((90 + 330, y + 42, 90 + 330 + int(full * laya / 100) * row_t, y + 68),
                            radius=6, fill=alpha(ACCENT, 0.85))
        text_at(d, (90 + 330 + int(full * clef / 100) * row_t + 14, y + 2), f"{clev_str(clef)}",
                font(MONO, 22), CHARCOAL, row_t)
        text_at(d, (90 + 330 + int(full * laya / 100) * row_t + 14, y + 40), f"{clev_str(laya)}",
                font(MONO, 22), MUTED, row_t)
    text_at(d, (90, 640), "dark: Clef   ·   terracotta: Laya   ·   published Decision Index scores",
            font(MONO, 18), MUTED, ease((t - 1.8) / 0.5))


def clev_str(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def scene_gate(img: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
    text_at(d, (90, 100), "Gate your CI on calibration", font(SANS, 46), CHARCOAL, ease(t / 0.4))
    draw_card(d, card_box(90, 200, 1100, 300))
    typed = ("clef-eval run evals/data/support_routing.jsonl "
             "--min-accuracy 0.90 --max-ece 0.15")
    n_chars = int(len(typed) * ease((t - 0.3) / 1.3))
    text_at(d, (120, 240), "$ " + typed[:n_chars], font(MONO, 26), CHARCOAL)
    if t > 1.8:
        blink = 0.5 + 0.5 * math.sin(t * 8)
        d.rectangle((120 + 26 + len(typed[:n_chars]) * 14, 238, 120 + 28 + len(typed[:n_chars]) * 14, 268),
                    fill=alpha(CHARCOAL, blink))
    lines = [
        ("samples=32 failures=0", MUTED, 0.0),
        ("accuracy=0.9375", MUTED, 0.25),
        ("ece=0.0612", MUTED, 0.5),
        ("gate: PASSED", "#346538", 1.9),
    ]
    for i, (line, color, start) in enumerate(lines):
        a = ease((t - 2.0 - start) / 0.4)
        if a > 0:
            text_at(d, (120, 300 + i * 40), line, font(MONO, 26), color, a)


def scene_close(img: Image.Image, d: ImageDraw.ImageDraw, t: float) -> None:
    a = ease(t / 0.6)
    text_at(d, (90, 280), "pip install clef-evals", font(MONO, 52), CHARCOAL, a)
    text_at(d, (90, 370), "github.com/Gjusev/clef-evals", font(SANS_REG, 34), MUTED, ease((t - 0.3) / 0.6))
    dot_r = 7 + 2 * math.sin(t * 4)
    d.ellipse([90 - dot_r, 470 - dot_r, 90 + dot_r, 470 + dot_r], fill=alpha(ACCENT, a))


SCENES = [(2.4, scene_title), (4.0, scene_pipeline), (3.6, scene_benchmarks), (2.6, scene_gate), (1.4, scene_close)]


def render_frame(frame_index: int, total: int) -> Image.Image:
    time_global = frame_index / FPS
    img, d = new_frame()
    elapsed = 0.0
    for duration, scene in SCENES:
        if elapsed <= time_global < elapsed + duration:
            scene(img, d, time_global - elapsed)
            break
        elapsed += duration
    # gentle cross-scene fade handled by drawing overlap: none (hard cut, minimalist)
    return img


def main() -> None:
    total_frames = int(sum(duration for duration, _ in SCENES) * FPS)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "frames"
        raw.mkdir()
        for i in range(total_frames):
            render_frame(i, total_frames).convert("RGB").save(raw / f"{i:05d}.png")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(raw / "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             str(HERE / "brag.mp4")],
            check=True,
        )
    # cover still: title scene at its most settled moment
    render_frame(int(2.0 * FPS), total_frames).convert("RGB").save(HERE / "brag.jpg", quality=92)
    print(f"wrote {HERE / 'brag.mp4'} and {HERE / 'brag.jpg'} ({total_frames} frames)")


if __name__ == "__main__":
    main()
