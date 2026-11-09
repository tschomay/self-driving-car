"""Top-down frames and annotated GIFs of one lap attempt on one track.

The whole track fits in the frame (north up). Drawn: road, start line, the car,
its 11 sensor rays (what the policy sees), and a trail colored by speed
(blue = slow, orange = fast).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sdc.racer import sim
from sdc.racer import track as trk

SIZE = 480
PAD_TOP = 64      # white band for the readout lines
GRASS = (214, 226, 206)
ROAD = (120, 126, 134)
EDGE = (245, 245, 245)
RAY = (31, 111, 235)
CAR = (180, 70, 43)
DIRT = (176, 138, 92)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class Painter:
    def __init__(self, t: trk.Track, size: int = SIZE, margin: float = 8.0, dirt: np.ndarray | None = None):
        self.t = t
        pts = np.vstack([t.left, t.right])
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        self.scale = size / max(hi - lo)
        self.lo = lo
        self.size = size
        self.off = (size - (hi - lo) * self.scale) / 2
        outer, inner = (t.left, t.right) if _area(t.left) > _area(t.right) else (t.right, t.left)
        base = Image.new("RGB", (size, size + PAD_TOP), (255, 255, 255))
        d = ImageDraw.Draw(base)
        d.rectangle([0, PAD_TOP, size, size + PAD_TOP], fill=GRASS)
        d.polygon([tuple(p) for p in self.px(outer)], fill=ROAD)
        d.polygon([tuple(p) for p in self.px(inner)], fill=GRASS)
        if dirt is not None:   # low-grip patches: brown road quads
            N = len(t.center)
            for k in np.flatnonzero(dirt):
                q = np.stack([t.left[k], t.left[(k + 1) % N], t.right[(k + 1) % N], t.right[k]])
                d.polygon([tuple(p) for p in self.px(q)], fill=DIRT)
        d.line([tuple(p) for p in self.px(np.vstack([outer, outer[:1]]))], fill=EDGE, width=2)
        d.line([tuple(p) for p in self.px(np.vstack([inner, inner[:1]]))], fill=EDGE, width=2)
        a, b = self.px(np.stack([t.left[0], t.right[0]]))
        for k in range(6):  # checkered start line
            f = k / 6
            g = (k + 1) / 6
            d.line([tuple(a + (b - a) * f), tuple(a + (b - a) * g)], fill=(0, 0, 0) if k % 2 else (255, 255, 255),
                   width=4)
        self.base = base

    def px(self, p):
        p = np.atleast_2d(p)
        x = (p[:, 0] - self.lo[0]) * self.scale + self.off[0]
        y = self.size - ((p[:, 1] - self.lo[1]) * self.scale + self.off[1]) + PAD_TOP
        return np.stack([x, y], axis=1)

    def frame(self, s: dict, trail: list[tuple[np.ndarray, float]]) -> np.ndarray:
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        if len(trail) > 1:
            pts = self.px(np.array([p for p, _ in trail]))
            for k in range(1, len(pts)):
                f = trail[k][1] / sim.V_MAX
                col = (round(31 + f * (194 - 31)), round(111 + f * (65 - 111)), round(235 + f * (12 - 235)))
                d.line([tuple(pts[k - 1]), tuple(pts[k])], fill=col, width=3)
        pos, h = np.asarray(s["pos"]), s["heading"]
        for ang, dist in zip(sim.RAY_ANGLES, s["rays"]):
            end = pos + dist * np.array([np.cos(h + ang), np.sin(h + ang)])
            a, b = self.px(np.stack([pos, end]))
            d.line([tuple(a), tuple(b)], fill=RAY, width=1)
            d.ellipse([b[0] - 2, b[1] - 2, b[0] + 2, b[1] + 2], fill=RAY)
        fwd = np.array([np.cos(h), np.sin(h)])
        left = np.array([-fwd[1], fwd[0]])
        # draw the car a little larger than life so it reads on a phone
        L, W = sim.CAR_LEN * 1.4, sim.CAR_WID * 1.6
        c = pos + fwd * sim.WHEELBASE / 2
        corners = [c + fwd * L / 2 + left * W / 2, c + fwd * L / 2 - left * W / 2,
                   c - fwd * L / 2 - left * W / 2, c - fwd * L / 2 + left * W / 2]
        d.polygon([tuple(p) for p in self.px(np.array(corners))], fill=CAR, outline=(255, 255, 255))
        return np.asarray(img)


def lap_gif(track_seed: int, traj: list[dict], row: dict, label: str, path: str | Path, every: int = 1,
            dirt: np.ndarray | None = None) -> None:
    """One lap attempt on one track, at real speed (10 Hz), with readouts and an end card.

    dirt: the track's dirt mask (Phase 3), drawn brown. States with 'slip' get a slip readout.
    """
    from sdc.gifs import annotate, save_gif

    t = trk.generate(track_seed)
    p = Painter(t, dirt=dirt)
    frames, trail = [], []
    for k, s in enumerate(traj):
        trail.append((np.asarray(s["pos"]), s["v"]))
        last = k == len(traj) - 1
        if k % every and not last:
            continue
        lines = [f"time {s['t'] * sim.DT:5.1f} s   ·   lap {100 * s['progress']:3.0f}% done",
                 f"{'held-out' if track_seed >= trk.HELD_OUT_BASE else 'training'} track {track_seed}, start line, standstill, real speed",
                 f"speed {s['v'] * 3.6:3.0f} km/h   ·   steer {s['steer']:+.2f}   ·   throttle {s['throttle']:+.2f}"]
        if "slip" in s:
            ph = row.get("physics", {})
            lines[1] = (f"held-out track {track_seed}" + (f" · grip {ph['grip']:.2f}× · delay {1000 * ph['delay']:.0f} ms"
                        if ph else "") + (" · on dirt" if s.get("dirt") else ""))
            lines.append(f"tail slide (rear slip) {abs(np.degrees(s['slip'][1])):4.1f}°   ·   front slip "
                         f"{abs(np.degrees(s['slip'][0])):4.1f}°")
        end, good = None, True
        if last:
            if row["finished"]:
                end = f"Lap finished in {row['lap_time']:.1f} s"
            elif row["crashed"]:
                end, good = f"Off the road at {s['t'] * sim.DT:.1f} s ({100 * row['laps']:.0f}% of the lap)", False
            else:
                end, good = f"Timed out at {s['t'] * sim.DT:.0f} s ({100 * row['laps']:.0f}% of the lap)", False
        img = p.frame(s, trail)
        if "image" in s:
            img = camera_inset(img, s["image"])
        frames.append(annotate(img, label, lines, s["progress"], end=end, end_good=good,
                               end_y=30 + PAD_TOP + SIZE // 2 - 32))
    save_gif(frames, path, frame_ms=1000 * sim.DT * every)


def camera_inset(frame: np.ndarray, cam_img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Paste the car's 64×64 camera image (enlarged, labelled) into the bottom-right corner."""
    from sdc.gifs import _font

    img = Image.fromarray(frame)
    c = Image.fromarray(cam_img).resize((cam_img.shape[1] * scale, cam_img.shape[0] * scale), Image.NEAREST)
    w, h = c.size
    x, y = img.width - w - 8, img.height - h - 8
    d = ImageDraw.Draw(img)
    d.rectangle([x - 3, y - 21, x + w + 2, y + h + 2], fill=(255, 255, 255))
    d.text((x, y - 19), "what the car sees", fill=(23, 32, 43), font=_font(13))
    img.paste(c.convert("RGB"), (x, y))
    from sdc.racer.camera import CAR_PIXEL   # mark where the car sits in its own view
    cy, cx = y + CAR_PIXEL[0] * scale, x + CAR_PIXEL[1] * scale
    d.polygon([(cx, cy - 5), (cx - 4, cy + 4), (cx + 4, cy + 4)], fill=CAR)
    return np.asarray(img)
