"""3D chase-camera GIFs of one run (MuJoCo's renderer, software OpenGL; needs libosmesa6).

The camera rides behind and to the right of the car, keeping a fixed offset from it, so you
see the jumps coming and the car's pitch and roll. Overlays match the side-view GIFs: who drives,
the course, time, speed, airtime, jumps landed so far, throttle/brake and steering gauges, and an
end card. Played at real speed (every 3rd sim step, about 17 fps, to keep files small).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import mujoco

from sdc.gifs import _font, save_gif
from sdc.jumps import car as cr
from sdc.jumps import env as je
from sdc.jumps.render import BRAKE, CAR, GAS, GOOD, BAD, INK, MUTED

W, H = 400, 225


def _overlay(img, who, lines, act, end=None, good=True):
    im = Image.fromarray(img)
    canvas = Image.new("RGB", (W, H + 26 + 40 + 30), (255, 255, 255))
    canvas.paste(im, (0, 26 + 40))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, 26], fill=INK)
    d.text((10, 13), who, font=_font(14, bold=True), fill=(255, 255, 255), anchor="lm")
    for i, line in enumerate(lines):
        d.text((10, 30 + i * 18), line, font=_font(13), fill=INK if i == 0 else MUTED)
    gy = 26 + 40 + H + 8
    for label, val, gx, pc, nc in (("brake ◀ throttle ▶ gas", act[1], 10, GAS, BRAKE),
                                   ("left ◀ steer ▶ right", -float(act[0]), W // 2 + 10, CAR, CAR)):
        gw = W // 2 - 30
        d.rectangle([gx, gy, gx + gw, gy + 8], fill=(232, 236, 241))
        mid = gx + gw / 2
        e = mid + float(val) * gw / 2
        d.rectangle([min(mid, e), gy, max(mid, e), gy + 8], fill=pc if val >= 0 else nc)
        d.line([mid, gy - 2, mid, gy + 10], fill=INK)
        d.text((gx + gw / 2, gy + 16), label, font=_font(10), fill=MUTED, anchor="mm")
    if end:
        col = GOOD if good else BAD
        d.rounded_rectangle([30, 90, W - 30, 150], radius=8, fill=(255, 255, 255), outline=col, width=3)
        d.text((W // 2, 112), end, font=_font(15, bold=True), fill=col, anchor="mm")
        d.text((W // 2, 134), "↻ GIF replays from the start", font=_font(12), fill=MUTED, anchor="mm")
    return np.asarray(canvas)


def gif(policy, course_seed, path, who, every=3, max_steps=1500):
    b = je.JumpBatch(1, courses=[course_seed], max_steps=max_steps, **getattr(policy, "env_kw", {}))
    c = b.cars[0]
    b._obs = c.reset(course_seed)[None]
    if hasattr(policy, "reset"):
        policy.reset(1)
    r = mujoco.Renderer(c.m, H, W)
    frames, n_j = [], len(c.course.jumps)
    a = np.zeros(2)
    while True:
        if c.t % every == 0:
            r.update_scene(c.d, camera="chase")
            s = c.state()
            landed = sum(bool(x and x["landed"]) for x in c.results)
            done = sum(x is not None for x in c.results)
            air = f"AIRBORNE {c.t * cr.DT - c.takeoff[0]:.2f} s" if c.airborne and c.takeoff else "on the ground"
            frames.append(_overlay(r.render(), f"{who} · course {course_seed}",
                                   [f"t {c.t * cr.DT:4.1f} s · {s['v_fwd']:4.1f} m/s · landed {landed}/{done} of {n_j}",
                                    f"pitch {np.degrees(s['pitch']):+4.0f}° · {air}"], a))
        a = np.asarray(policy(b))[0]
        _, term, trunc, reason = c.step(a, b.jump_bonus, max_steps)
        b._obs = c.obs()[None]
        if term or trunc:
            ep = c.summary(reason)
            r.update_scene(c.d, camera="chase")
            frames.append(_overlay(r.render(), f"{who} · course {course_seed}", ["end of run", ""], a,
                                   end=f"{reason} · landed {ep['landed']}/{n_j} jumps", good=ep["landed"] == n_j))
            break
    save_gif(frames, path, frame_ms=1000 * cr.DT * every)
    return ep
