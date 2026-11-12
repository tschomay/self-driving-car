"""Side-view GIFs of one run down one jump course (real speed, 50 Hz sim shown at 25 fps).

Main view: the terrain profile along the lane, the car (chassis, wheels, nose marker), its
forward rays, and the trail of where it has been (orange while airborne). The camera follows
the car. Under it: a strip with the whole course and a marker for the car (the progress bar),
and gauges for throttle/brake and steering. The readout says who drives, which course, the time,
the speed, the pitch vs. the ground slope, the airtime, and each jump's result so far.
End card: why the run ended and how many jumps it landed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mujoco

from sdc.gifs import _font, save_gif
from sdc.jumps import car as cr
from sdc.jumps import course as crs
from sdc.jumps import env as je

W, VIEW_H = 480, 200
PAD_TOP = 74
STRIP = 84
SKY = (236, 242, 248)
DIRT = (176, 138, 92)
DIRT_DARK = (128, 98, 62)
INK = (23, 32, 43)
MUTED = (91, 104, 120)
CAR = (31, 111, 235)
NOSE = (240, 180, 40)
AIR = (230, 120, 30)
GOOD = (15, 138, 106)
BAD = (180, 70, 43)
GAS = (15, 138, 106)
BRAKE = (190, 50, 40)
SPAN_BACK, SPAN_FWD = 2.5, 5.5    # metres shown behind / ahead of the car
Z_LO, Z_HI = -0.15, 1.75


def _snap(c: je.JumpCar, a):
    d = c.d
    R = d.xmat[1].reshape(3, 3)
    p = d.xpos[1]
    hx, hz = 0.23, 0.035
    corners = [p + R @ np.array([sx * hx, 0, sz * hz]) for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    wheels = [d.xpos[c.m.body(f"{w}_wheel").id].copy() for w in ("fl", "rl")]
    st = c.state()
    rays = []
    for i, ang in enumerate(cr.RAY_ANGLES):
        sid = c.m.site(f"ray{i}").id
        o = d.site_xpos[sid]
        dirn = d.site_xmat[sid].reshape(3, 3)[:, 2]
        dist = c.rays[i]
        rays.append((o[[0, 2]].copy(), (o + dirn * dist)[[0, 2]].copy(), dist < cr.RAY_CUTOFF))
    return dict(x=st["x"], z=st["z"], pitch=np.degrees(st["pitch"]), v=st["v_fwd"],
                slope=float(np.degrees(np.arctan(c.course.slope(st["x"])))), air=c.airborne,
                airtime=(c.t * cr.DT - c.takeoff[0]) if c.airborne and c.takeoff else 0.0,
                corners=[q[[0, 2]] for q in corners], wheels=[w[[0, 2]] for w in wheels], rays=rays,
                act=np.asarray(a, float).copy(), t=c.t * cr.DT, results=list(c.results),
                pending=c.pending is not None)


def record(policy, course_seed, max_steps=1500):
    """Drive one course with `policy(batch)`; return the snapshots and the episode summary."""
    b = je.JumpBatch(1, courses=[course_seed], max_steps=max_steps)
    c = b.cars[0]
    b._obs = c.reset(course_seed)[None]
    if hasattr(policy, "reset"):
        policy.reset(1)
    snaps = [_snap(c, np.zeros(2))]
    while True:
        a = np.asarray(policy(b))[0]
        _, term, trunc, reason = c.step(a, b.jump_bonus, max_steps)
        b._obs = c.obs()[None]
        snaps.append(_snap(c, a))
        if term or trunc:
            return snaps, c.summary(reason), c.course


def _frame(s, course: crs.Course, title, lines, trail, end=None, end_good=True):
    img = Image.new("RGB", (W, PAD_TOP + VIEW_H + STRIP), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x0 = s["x"] - SPAN_BACK
    sx = W / (SPAN_BACK + SPAN_FWD)
    sz = VIEW_H / (Z_HI - Z_LO)

    def P(x, z):
        return (x - x0) * sx, min(PAD_TOP + VIEW_H - (z - Z_LO) * sz, PAD_TOP + VIEW_H)

    d.rectangle([0, PAD_TOP, W, PAD_TOP + VIEW_H], fill=SKY)
    xs = np.linspace(x0, x0 + SPAN_BACK + SPAN_FWD, 240)
    zs = course.height(xs)
    poly = [P(xs[0], Z_LO)] + [P(x, z) for x, z in zip(xs, zs)] + [P(xs[-1], Z_LO)]
    d.polygon(poly, fill=DIRT)
    d.line([P(x, z) for x, z in zip(xs, zs)], fill=DIRT_DARK, width=2)
    # finish line
    if x0 < crs.FINISH_X < x0 + SPAN_BACK + SPAN_FWD:
        fx, _ = P(crs.FINISH_X, 0)
        d.line([fx, PAD_TOP, fx, PAD_TOP + VIEW_H], fill=INK, width=2)
    # trail
    for (ax, az, aair), (bx, bz, _) in zip(trail[:-1], trail[1:]):
        d.line([P(ax, az), P(bx, bz)], fill=AIR if aair else (150, 160, 175), width=2)
    # rays
    for o, e, hit in s["rays"]:
        d.line([P(*o), P(*e)], fill=(31, 111, 235, 90) if hit else (170, 190, 220), width=1)
        if hit:
            ex, ez = P(*e)
            d.ellipse([ex - 2, ez - 2, ex + 2, ez + 2], fill=CAR)
    # car
    d.polygon([P(*q) for q in s["corners"]], fill=CAR, outline=INK)
    nose = (s["corners"][0] + s["corners"][1]) / 2
    nx, nz = P(*nose)
    d.ellipse([nx - 3, nz - 3, nx + 3, nz + 3], fill=NOSE)
    r = cr.WHEEL_R * sx
    for wx, wz in s["wheels"]:
        cx, cz = P(wx, wz)
        d.ellipse([cx - r, cz - r, cx + r, cz + r], fill=(40, 40, 40))
    # readout
    f, fb = _font(13), _font(14, bold=True)
    d.rectangle([0, 0, W, 26], fill=INK)
    d.text((10, 13), title, font=fb, fill=(255, 255, 255), anchor="lm")
    for i, line in enumerate(lines):
        d.text((10, 32 + i * 18), line, font=f, fill=INK if i == 0 else MUTED)
    # strip: whole course + car marker (progress), then gauges
    y0 = PAD_TOP + VIEW_H + 6
    xs = np.linspace(0, crs.LENGTH, 300)
    zs = course.height(xs)
    pts = [(10 + x / crs.LENGTH * (W - 20), y0 + 22 - z * 22) for x, z in zip(xs, zs)]
    d.line(pts, fill=DIRT_DARK, width=2)
    cx = 10 + min(s["x"], crs.LENGTH) / crs.LENGTH * (W - 20)
    d.ellipse([cx - 4, y0 + 18 - 4, cx + 4, y0 + 18 + 4], fill=CAR)
    for k, j in enumerate(course.jumps):
        res = s["results"][k]
        jx = 10 + j.lip / crs.LENGTH * (W - 20)
        mark = "·" if res is None else ("✓" if res["landed"] else "✗")
        col = MUTED if res is None else (GOOD if res["landed"] else BAD)
        d.text((jx, y0 + 34), mark, font=_font(14, bold=True), fill=col, anchor="mm")
    # throttle gauge (centre = 0, right = gas, left = brake) and steering gauge
    gy = y0 + 50
    for label, val, gx, pos_col, neg_col in (("brake ◀ throttle ▶ gas", s["act"][1], 10, GAS, BRAKE),
                                             ("right ◀ steer ▶ left", s["act"][0], W // 2 + 10, CAR, CAR)):
        gw = W // 2 - 30
        d.rectangle([gx, gy, gx + gw, gy + 8], fill=(232, 236, 241))
        mid = gx + gw / 2
        end_x = mid + val * gw / 2
        d.rectangle([min(mid, end_x), gy, max(mid, end_x), gy + 8], fill=pos_col if val >= 0 else neg_col)
        d.line([mid, gy - 2, mid, gy + 10], fill=INK)
        d.text((gx + gw / 2, gy + 16), label, font=_font(10), fill=MUTED, anchor="mm")
    if end:
        col = GOOD if end_good else BAD
        bx0, by0, bw, bh = 30, PAD_TOP + 20, W - 60, 60
        d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=8, fill=(255, 255, 255), outline=col, width=3)
        d.text((W // 2, by0 + 22), end, font=_font(15, bold=True), fill=col, anchor="mm")
        d.text((W // 2, by0 + 44), "↻ GIF replays from the start", font=_font(12), fill=MUTED, anchor="mm")
    return np.asarray(img)


def gif(policy, course_seed, path, who, every=2):
    """Record and save one run as an annotated GIF. Returns the episode summary."""
    snaps, ep, course = record(policy, course_seed)
    frames, trail = [], []
    n_j = len(course.jumps)
    for i, s in enumerate(snaps):
        trail.append((s["x"], s["z"] - 0.05, s["air"]))
        if i % every and i != len(snaps) - 1:
            continue
        done = sum(r is not None for r in s["results"])
        landed = sum(bool(r and r["landed"]) for r in s["results"])
        lines = [f"t {s['t']:4.1f} s · speed {s['v']:4.1f} m/s · jumps landed {landed}/{done} of {n_j}",
                 f"pitch {s['pitch']:+5.0f}° (ground {s['slope']:+4.0f}°) · "
                 + (f"AIRBORNE {s['airtime']:.2f} s" if s["air"] else "on the ground")]
        frames.append(_frame(s, course, f"{who} · course {course_seed}", lines, trail[-400:]))
    end = f"{ep['reason']} · landed {ep['landed']}/{ep['n_jumps']} jumps"
    last = snaps[-1]
    frames.append(_frame(last, course, f"{who} · course {course_seed}",
                         [f"t {last['t']:4.1f} s · end of run", ""], trail[-400:],
                         end=end, end_good=ep["landed"] == ep["n_jumps"]))
    save_gif(frames, path, frame_ms=1000 * cr.DT * every)
    return ep
