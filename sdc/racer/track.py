"""Procedural closed race tracks.

A track is a smooth closed centerline with constant width. Shape: 8-14 control
points at random radii (40-100% of R0) around a circle, joined by a closed
Catmull-Rom spline, rounded off with a Gaussian blur (σ = 4 m) and resampled to
N points evenly spaced along the centerline. Half the tracks run clockwise. Each seed gives one
track, so a seed is a track's name. Training uses seeds < 10_000; held-out
validation tracks use seeds from 10_000 (HELD_OUT_SEEDS); the gate's test tracks 20_000-20_019 (TEST_SEEDS).

Rejected and redrawn (deterministically from the same seed):
  - corners tighter than MIN_RADIUS (sharper than the car's full-lock turning circle)
  - two parts of the track closer than 2 × width (edges would overlap)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_POINTS = 500
WIDTH = 6.0          # metres, edge to edge
MIN_RADIUS = 6.0     # tightest allowed corner radius on the centerline, metres
SMOOTH = 4.0         # Gaussian blur of the spline, metres
HELD_OUT_BASE = 10_000
HELD_OUT_SEEDS = tuple(range(HELD_OUT_BASE, HELD_OUT_BASE + 20))
# Test set (from Phase 3's gate on; owner's decision 2026-11-10): tracks never used for any experiment or
# choice. HELD_OUT_SEEDS is the validation set that experiments report on; TEST_SEEDS are looked at
# once, at a gate.
TEST_SEEDS = tuple(range(20_000, 20_020))


@dataclass
class Track:
    seed: int
    center: np.ndarray     # (N, 2) centerline points, evenly spaced
    tangent: np.ndarray    # (N, 2) unit direction of travel at each point
    normal: np.ndarray     # (N, 2) unit left normal
    curvature: np.ndarray  # (N,) signed, 1/m (positive = left turn)
    left: np.ndarray       # (N, 2) left edge
    right: np.ndarray      # (N, 2) right edge
    length: float          # lap length, metres
    ds: float              # spacing between centerline points, metres
    width: float = WIDTH


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _resample(pts: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    length = float(s[-1])
    t = np.linspace(0, length, n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1), length


def _build(seed: int, center: np.ndarray, length: float) -> Track:
    n = len(center)
    ds = length / n
    d = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent = d / np.linalg.norm(d, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    seg = np.roll(center, -1, axis=0) - center
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    curvature = _wrap(heading - np.roll(heading, 1)) / ds
    curvature = (curvature + np.roll(curvature, 1) + np.roll(curvature, -1)) / 3
    half = WIDTH / 2
    return Track(seed, center, tangent, normal, curvature, center + half * normal, center - half * normal, length, ds)


def _catmull_rom(P: np.ndarray, n_per: int) -> np.ndarray:
    """Closed uniform Catmull-Rom spline through control points P (passes through each one)."""
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    out = []
    for i in range(len(P)):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % len(P)], P[(i + 2) % len(P)]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def _blur(pts: np.ndarray, length: float, sigma: float) -> np.ndarray:
    n = len(pts)
    s = sigma / (length / n)
    h = int(3 * s) + 1
    w = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
    w /= w.sum()
    return np.stack([np.convolve(np.r_[pts[-h:, j], pts[:, j], pts[:h, j]], w, "valid") for j in range(2)], 1)


def generate(seed: int) -> Track:
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        k = int(rng.integers(8, 15))
        r0 = rng.uniform(45, 70)
        ang = np.sort((np.arange(k) + rng.uniform(-0.35, 0.35, k)) * 2 * np.pi / k)
        rad = r0 * rng.uniform(0.4, 1.0, k)
        dense, length = _resample(_catmull_rom(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), 300), 2000)
        dense = _blur(dense, length, SMOOTH)
        if rng.random() < 0.5:  # clockwise half the time
            dense = dense[::-1]
        center, length = _resample(dense, N_POINTS)
        track = _build(seed, center, length)
        if np.max(np.abs(track.curvature)) > 1 / MIN_RADIUS:
            continue
        # no two parts of the track (more than ~3 widths apart along it) closer than 2 widths
        dist = np.linalg.norm(center[:, None] - center[None], axis=2)
        idx = np.arange(N_POINTS)
        sep = np.abs(idx[:, None] - idx[None])
        sep = np.minimum(sep, N_POINTS - sep) * track.ds
        if np.any(dist[sep > 3 * WIDTH] < 2 * WIDTH):
            continue
        # start line: rotate so index 0 sits on the straightest stretch
        start = int(np.argmin(np.convolve(np.abs(np.r_[track.curvature, track.curvature[:30]]),
                                          np.ones(30), "valid")[:N_POINTS]))
        return _build(seed, np.roll(center, -start, axis=0), length)
    raise RuntimeError(f"could not generate a valid track for seed {seed}")
