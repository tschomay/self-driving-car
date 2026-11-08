"""Ego-centred top-down camera: what the Phase 2 car sees instead of rays.

A 64×64 grayscale image looking straight down, centred on the car and turned
with it, so "up" in the image is always the car's heading. One pixel is
0.625 m, so the image covers 40 m × 40 m: 35 m ahead, 5 m behind, 20 m to
each side. The car itself is not drawn (it is always at the same pixel).

Colors (0-255): grass 40, road 110, white edge lines 230 (0.75 m wide).
Other parts of the track can appear in the image too, as on a real camera.

How it is drawn fast: each track is rasterized once into a 0.25 m/pixel
bitmap (cached per track seed), and each camera pixel samples its nearest
bitmap pixel after rotating into the car's frame.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

from sdc.racer import track as trk

SIZE = 64            # pixels per side
M_PER_PX = 0.625     # camera resolution, metres per pixel
BEHIND = 5.0         # metres visible behind the car
BITMAP_RES = 0.25    # metres per pixel of the cached track bitmap
GRASS, ROAD, EDGE = 40, 110, 230
EDGE_WIDTH = 0.75    # metres (wider than a camera pixel, so lines never break up)
_MARGIN = 60.0       # bitmap margin around the track (> camera reach), metres

# camera pixel (row, col) -> metres (forward, left) in the car's frame; row 0 is the far edge
_rows = np.arange(SIZE)[:, None]
_cols = np.arange(SIZE)[None, :]
FWD = np.broadcast_to((SIZE - _rows - 0.5) * M_PER_PX - BEHIND, (SIZE, SIZE)).astype(np.float64)
LEFT = np.broadcast_to((SIZE / 2 - _cols - 0.5) * M_PER_PX, (SIZE, SIZE)).astype(np.float64)
CAR_PIXEL = (int(SIZE - BEHIND / M_PER_PX), SIZE // 2)   # (row, col) where the car sits


def rasterize(t: trk.Track) -> tuple[np.ndarray, np.ndarray]:
    """Bitmap (H, W) uint8 of one track, and the world coordinate of pixel (0, 0)."""
    pts = np.vstack([t.left, t.right])
    lo = pts.min(0) - _MARGIN
    hi = pts.max(0) + _MARGIN
    w, h = np.ceil((hi - lo) / BITMAP_RES).astype(int)
    img = Image.new("L", (int(w), int(h)), GRASS)
    d = ImageDraw.Draw(img)
    px = lambda p: [tuple(q) for q in (np.atleast_2d(p) - lo) / BITMAP_RES]
    n = len(t.center)
    for i in range(n):
        j = (i + 1) % n
        d.polygon(px([t.left[i], t.left[j], t.right[j], t.right[i]]), fill=ROAD)
    lw = max(1, round(EDGE_WIDTH / BITMAP_RES))
    for edge in (t.left, t.right):
        d.line(px(np.vstack([edge, edge[:1]])), fill=EDGE, width=lw, joint="curve")
    return np.asarray(img), lo


class Camera:
    """Renders camera images for any (track seed, position, heading); caches track bitmaps."""

    def __init__(self, cache_size: int = 256):
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.cache_size = cache_size

    def bitmap(self, seed: int):
        if seed in self._cache:
            self._cache.move_to_end(seed)
        else:
            self._cache[seed] = rasterize(trk.generate(seed))
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return self._cache[seed]

    def view(self, seed: int, pos, heading: float) -> np.ndarray:
        """One (SIZE, SIZE) uint8 image."""
        img, lo = self.bitmap(int(seed))
        c, s = np.cos(heading), np.sin(heading)
        wx = pos[0] + FWD * c - LEFT * s
        wy = pos[1] + FWD * s + LEFT * c
        ix = np.clip(((wx - lo[0]) / BITMAP_RES).astype(np.intp), 0, img.shape[1] - 1)
        iy = np.clip(((wy - lo[1]) / BITMAP_RES).astype(np.intp), 0, img.shape[0] - 1)
        return img[iy, ix]

    def views(self, seeds, pos, heading) -> np.ndarray:
        """(B, SIZE, SIZE) uint8 images for a batch of cars."""
        return np.stack([self.view(s, p, h) for s, p, h in zip(seeds, pos, heading)])

    def batch_views(self, b) -> np.ndarray:
        """Images for every car in a RacerBatch, as it is now."""
        return self.views(b.seeds, b.pos, b.heading)


class FrameStack:
    """Keeps the last n camera images per car, newest last; refilled on reset."""

    def __init__(self, n_cars: int, n_frames: int):
        self.n = n_frames
        self.frames = np.zeros((n_cars, n_frames, SIZE, SIZE), np.uint8)

    def reset(self, i, img):
        self.frames[i] = img

    def push(self, imgs):
        self.frames = np.concatenate([self.frames[:, 1:], imgs[:, None]], axis=1)
        return self.frames
