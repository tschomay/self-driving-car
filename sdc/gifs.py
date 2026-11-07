"""Annotated rollout GIFs: every frame says what you are looking at.

Each GIF shows ONE episode of ONE policy, played until it ends, at real speed.
Overlays:
  - top bar: who is driving and which held-out episode this is
  - live readout: step / max steps, return so far, the action just taken, the
    quantity that ends the episode (e.g. pole angle vs. its limit)
  - bottom progress bar: how far through the episode got (step cap, or lap distance)
  - end card (held ~2 s): why the episode ended, then "↻ replays from step 0"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

INK = (23, 32, 43)
MUTED = (91, 104, 120)
BAR_BG = (232, 236, 241)
GOOD = (15, 138, 106)
BAD = (180, 70, 43)
ACCENT = (31, 111, 235)


def _font(size: int, bold: bool = False):
    path = _FONT_PATHS[1 if bold else 0]
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def annotate(frame: np.ndarray, title: str, lines: list[str], progress: float, end: str | None = None,
             end_good: bool = True, width: int = 480, end_y: int | None = None) -> np.ndarray:
    """Return `frame` scaled to `width` with a header band, readout lines and a progress bar."""
    img = Image.fromarray(frame)
    h = round(img.height * width / img.width)
    img = img.resize((width, h), Image.LANCZOS)
    head, foot = 30, 10
    canvas = Image.new("RGB", (width, h + head + foot), (255, 255, 255))
    canvas.paste(img, (0, head))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, width, head], fill=INK)
    d.text((10, head // 2), title, font=_font(14, bold=True), fill=(255, 255, 255), anchor="lm")
    f = _font(13)
    for i, line in enumerate(lines):
        d.text((10, head + 8 + i * 18), line, font=f, fill=MUTED if i else INK)
    # progress bar through the step cap
    y0 = head + h
    d.rectangle([0, y0, width, y0 + foot], fill=BAR_BG)
    color = (GOOD if end_good else BAD) if end else ACCENT
    d.rectangle([0, y0, round(width * min(progress, 1.0)), y0 + foot], fill=color)
    if end:
        box_w, box_h = width - 60, 64
        x0 = 30
        yb = head + 66 if end_y is None else end_y  # default: CartPole's empty sky, clear of cart and pole
        d.rounded_rectangle([x0, yb, x0 + box_w, yb + box_h], radius=8, fill=(255, 255, 255),
                            outline=color, width=3)
        d.text((width // 2, yb + 22), end, font=_font(15, bold=True), fill=color, anchor="mm")
        d.text((width // 2, yb + 46), "↻ GIF replays from step 0", font=_font(12), fill=MUTED, anchor="mm")
    return np.asarray(canvas)


def save_gif(frames: list[np.ndarray], path: str | Path, frame_ms: float, hold_last_ms: float = 2200):
    import imageio.v2 as imageio

    durations = [frame_ms] * (len(frames) - 1) + [hold_last_ms]
    imageio.mimsave(path, frames, duration=durations, loop=0)
