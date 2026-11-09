"""Saliency maps: which camera pixels the policy's steering and throttle depend on.

Method: SmoothGrad (Smilkov et al., 2017). For one camera image, take the gradient of the
policy's mean steering (or throttle) with respect to every input pixel, averaged over 16
copies of the image with a little noise added (plain gradients are speckly). The absolute
value says how much a small change to that pixel would move the action.

Edge check (the gate's "keys on track edges, not artifacts"): the white edge lines are a
known set of pixels in each image. Edge enrichment = (share of total saliency that falls on
edge pixels and their 1-pixel neighbours) ÷ (share of the image those pixels cover). 1 means
no preference; above 1 means the network looks at edges more than at a random pixel.

Usage:
    python -m sdc.phase2.saliency runs/p2-e5-mine-s0 --out reports/p2-saliency-s0.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sdc.racer import camera, evaluate
from sdc.racer import track as trk


def _policy_mean(model, image, speed):
    import torch

    pol = model.policy
    obs = {"image": image}
    if speed is not None:
        obs["speed"] = speed
    feats = pol.extract_features(obs, pol.features_extractor)
    return pol.action_net(pol.mlp_extractor.forward_actor(feats))


def smoothgrad(model, img: np.ndarray, v: float | None, n: int = 16, sigma: float = 0.1, seed: int = 0):
    """(2, H, W) absolute saliency for steering and throttle; img (C, H, W) uint8."""
    import torch

    rng = torch.Generator().manual_seed(seed)
    x0 = torch.from_numpy(img.astype(np.float32))[None]
    out = np.zeros((2, *img.shape[1:]))
    for k in range(2):
        g = torch.zeros_like(x0)
        for _ in range(n):
            x = (x0 + torch.randn(x0.shape, generator=rng) * sigma * 255).clamp(0, 255).requires_grad_(True)
            sp = None if v is None else torch.tensor([[v]], dtype=torch.float32)
            a = _policy_mean(model, x, sp)[0, k]
            (gx,) = torch.autograd.grad(a, x)
            g += gx
        out[k] = (g / n).abs()[0].sum(0).numpy()   # sum over stacked frames
    return out


def edge_mask(img: np.ndarray, grow: int = 1) -> np.ndarray:
    m = img == camera.EDGE
    if grow:
        p = np.pad(m, grow)
        h, w = m.shape
        m = np.zeros_like(m)
        for dy in range(2 * grow + 1):
            for dx in range(2 * grow + 1):
                m |= p[dy:dy + h, dx:dx + w]
    return m


def enrichment(sal: np.ndarray, img: np.ndarray) -> float:
    m = edge_mask(img)
    return float((sal[m].sum() / sal.sum()) / m.mean()) if m.any() and sal.sum() > 0 else float("nan")


def _blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(3) for dx in range(3)) / 9


def _overlay(img: np.ndarray, sal: np.ndarray, scale: int = 4) -> Image.Image:
    """Grayscale camera image with saliency (3×3-blurred for display) in orange; top 0.5% = full strength, gamma 1.5 to quiet the background."""
    sal = _blur(sal)
    s = (sal / (np.quantile(sal, 0.995) + 1e-12)) ** 1.5
    s = np.clip(s, 0, 1)[..., None]
    base = np.repeat(img[..., None].astype(float) * 0.55, 3, axis=2)
    hot = np.array([255, 140, 20], float)
    rgb = (base * (1 - s) + hot * s).astype(np.uint8)
    im = Image.fromarray(rgb).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    cy, cx = camera.CAR_PIXEL[0] * scale, camera.CAR_PIXEL[1] * scale
    d.polygon([(cx, cy - 8), (cx - 6, cy + 6), (cx + 6, cy + 6)], outline=(80, 170, 255))
    return im


def analyze(run_dir: str | Path, tracks=trk.HELD_OUT_SEEDS, every: int = 5, picks: int = 4, out: str | Path | None = None,
            track_for_figure: int = trk.HELD_OUT_SEEDS[0]) -> dict:
    """Saliency over held-out laps: edge enrichment stats, and a figure of `picks` moments on one track."""
    import torch
    from stable_baselines3 import PPO

    from sdc.racer.camera_env import CameraPolicy
    from sdc.gifs import _font

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    nf, use_speed = m["env"].get("n_frames", 1), m["env"].get("speed", True)
    model = PPO.load(run_dir / "model.zip", device="cpu")

    class Rec(CameraPolicy):
        def __call__(self, b):
            a = super().__call__(b)
            if not hasattr(self, "done"):
                self.done = np.zeros(b.n, bool)
                self.moved = np.zeros(b.n, bool)
            self.done |= self.moved & (b.steps == 0)    # a car back at step 0 has finished its one attempt
            self.moved |= b.steps > 0
            self.frames_log.append((self.stack.frames.copy(), b.v.copy(), b.curv[np.arange(b.n), b.idx].copy(),
                                    ~self.done))
            return a

    pol = Rec(lambda o: model.predict(o, deterministic=True)[0], nf, use_speed)
    pol.frames_log = []
    seeds = list(tracks)
    evaluate.run_laps(pol, seeds)
    enr = {"steer": [], "throttle": []}
    fig_rows = []
    fig_i = seeds.index(track_for_figure) if track_for_figure in seeds else 0
    for t, (frames, v, curv, active) in enumerate(pol.frames_log):
        if t % every:
            continue
        for i in np.flatnonzero(active):   # only the lap attempt itself
            sal = smoothgrad(model, frames[i], v[i] / 25.0 if use_speed else None, n=8, seed=t)
            for k, name in enumerate(("steer", "throttle")):
                enr[name].append(enrichment(sal[k], frames[i][-1]))
            if i == fig_i:
                fig_rows.append((t, frames[i], frames[i][-1], v[i], curv[i]))
    stats = {k: dict(median=float(np.nanmedian(x)), mean=float(np.nanmean(x)), n=len(x)) for k, x in enr.items()}
    if out and fig_rows:
        # pick moments: a straight, and the approach to / middle of the sharpest corner seen
        fig_rows = [r for r in fig_rows if r[3] > 1.0] or fig_rows
        order = np.argsort([abs(r[4]) for r in fig_rows])
        chosen = sorted({int(order[0]), int(order[-1]), int(order[len(order) // 2]),
                         int(order[int(len(order) * 0.85)])})[:picks]
        S = 3
        W = 64 * S
        tiles = []
        for j in chosen:
            t, fr, img, v, c = fig_rows[j]
            sal = smoothgrad(model, fr, v / 25.0 if use_speed else None, n=64, sigma=0.05, seed=t)   # cleaner maps
            col = Image.new("RGB", (W, 3 * W + 60), (255, 255, 255))
            d = ImageDraw.Draw(col)
            d.text((2, 2), f"t {t / 10:.1f} s · {v * 3.6:.0f} km/h", fill=(23, 32, 43), font=_font(13))
            d.text((2, 18), f"corner radius {1 / abs(c):.0f} m" if abs(c) > 1e-3 else "straight", fill=(91, 104, 120),
                   font=_font(12))
            col.paste(Image.fromarray(img).resize((W, W), Image.NEAREST).convert("RGB"), (0, 38))
            col.paste(_overlay(img, sal[0], S), (0, 38 + W + 2))
            col.paste(_overlay(img, sal[1], S), (0, 38 + 2 * W + 4))
            tiles.append(col)
        lab = 70
        fig = Image.new("RGB", (lab + len(tiles) * (W + 8), 3 * W + 100), (255, 255, 255))
        d = ImageDraw.Draw(fig)
        d.text((4, 4), f"{m['label']}, seed {m['seed']}: held-out track {seeds[fig_i]}. Orange = pixels the action "
                       "depends on most.", fill=(23, 32, 43), font=_font(13))
        for r, name in enumerate(("camera", "steering", "throttle")):
            d.text((4, 40 + 38 + r * (W + 2) + W // 2 - 8), name, fill=(91, 104, 120), font=_font(12))
        for k, tile in enumerate(tiles):
            fig.paste(tile, (lab + k * (W + 8), 40))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.save(out)
    return dict(run=str(run_dir), label=m["label"], seed=m["seed"], edge_enrichment=stats,
                edge_area_share=float(np.mean([edge_mask(f[0][i][-1]).mean() for f in pol.frames_log[::every]
                                               for i in np.flatnonzero(f[3])])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", default=None)
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--tracks", type=int, default=20, help="first N held-out tracks")
    a = p.parse_args()
    r = analyze(a.run, trk.HELD_OUT_SEEDS[:a.tracks], a.every, out=a.out)
    print(json.dumps(r, indent=1))
    (Path(a.run) / "saliency.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
