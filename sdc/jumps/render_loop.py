"""GIFs of one lap of a loop course (Phase 4.6), at real speed.

gif()        top-down map of the whole loop: the lane (lighter = higher: banks, jumps), the start/finish line, each
             jump numbered (grey = not reached, green = landed, red = missed), the car as an arrow, and its trail
             (blue on the ground, orange in the air, dark dots where the car slides sideways: skid marks).
             Under the map: throttle/brake and steering gauges and a lap progress bar.
chase_gif()  3D camera riding behind the car, turning with the track (MuJoCo's software renderer; needs libosmesa6).
             Drawn on a coarser copy of the terrain (0.2 m grid) to keep rendering fast; the physics is unchanged.
Both carry the same overlays: who drives, which loop and start, time, speed, lap %, jumps landed, and an end card.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import mujoco

from sdc.gifs import _font, save_gif
from sdc.jumps import car as cr
from sdc.jumps import env as je
from sdc.jumps import loop as lp
from sdc.jumps.render import AIR, BAD, BRAKE, CAR, GAS, GOOD, INK, MUTED

W = 420
MAP = 420
HEAD, READ, GAUGE, BAR = 26, 40, 32, 8
GRASS = (214, 222, 205)
SKID = (60, 40, 30)
PENDING = (120, 130, 142)


class _View:
    """World metres -> map pixels for one loop (the loop's bounding box, with a margin, fills the map)."""

    def __init__(self, course):
        lo, hi = course.center.min(0) - 4.0, course.center.max(0) + 4.0
        self.scale = MAP / float((hi - lo).max())
        mid = (lo + hi) / 2
        self.x0, self.y0 = mid[0] - MAP / 2 / self.scale, mid[1] + MAP / 2 / self.scale

    def px(self, X, Y):
        return (X - self.x0) * self.scale, (self.y0 - Y) * self.scale

    def background(self, course):
        hf = course.hfield() * lp.ZMAX
        u = np.arange(MAP)
        X = self.x0 + (u + 0.5) / self.scale
        Y = self.y0 - (u + 0.5) / self.scale
        ix = np.clip(np.rint((X + lp.BOX / 2) / lp.D).astype(int), 0, lp.N - 1)
        iy = np.clip(np.rint((Y + lp.BOX / 2) / lp.D).astype(int), 0, lp.N - 1)
        z = hf[np.ix_(iy, ix)]
        t = np.clip((z - (lp.BASE - 0.3)) / 1.0, 0, 1)[..., None]
        img = (np.array([118, 90, 58]) * (1 - t) + np.array([236, 214, 170]) * t)
        img[np.abs(z - (lp.BASE + lp.BANK_H)) < 1e-4] = GRASS
        return Image.fromarray(img.astype(np.uint8))


def _labels(d, view, course, results):
    X, Y, th = course.pose(0.0)
    n = np.array([-np.sin(th), np.cos(th)]) * lp.DRIVE_W / 2
    d.line([view.px(X + n[0], Y + n[1]), view.px(X - n[0], Y - n[1])], fill=(255, 255, 255), width=3)
    a = view.px(X + 2.6 * n[0], Y + 2.6 * n[1])
    d.text(a, "start/finish", font=_font(10), fill=INK, anchor="mm")
    for k, j in enumerate(course.jumps):
        X, Y, th = course.pose(j.lip)
        n = np.array([-np.sin(th), np.cos(th)])
        p = view.px(X + 3.4 * n[0], Y + 3.4 * n[1])
        r = results[k]
        col = PENDING if r is None else (GOOD if r["landed"] else BAD)
        d.ellipse([p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9], fill=col)
        d.text(p, str(k + 1), font=_font(11, bold=True), fill=(255, 255, 255), anchor="mm")
        d.text((p[0], p[1] + 16), j.kind, font=_font(10), fill=INK, anchor="mm")


def _gauges(d, y, act):
    for label, val, gx, pc, nc in (("brake ◀ throttle ▶ gas", act[1], 10, GAS, BRAKE),
                                   ("left ◀ steer ▶ right", -float(act[0]), W // 2 + 10, CAR, CAR)):
        gw = W // 2 - 30
        d.rectangle([gx, y, gx + gw, y + 8], fill=(232, 236, 241))
        mid = gx + gw / 2
        e = mid + float(val) * gw / 2
        d.rectangle([min(mid, e), y, max(mid, e), y + 8], fill=pc if val >= 0 else nc)
        d.line([mid, y - 2, mid, y + 10], fill=INK)
        d.text((gx + gw / 2, y + 17), label, font=_font(10), fill=MUTED, anchor="mm")


def _frame(img, who, lines, act, progress, end=None, good=True, end_y=None):
    canvas = Image.new("RGB", (W, HEAD + READ + img.height + GAUGE + BAR), (255, 255, 255))
    canvas.paste(img, (0, HEAD + READ))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, HEAD], fill=INK)
    d.text((10, HEAD // 2), who, font=_font(13, bold=True), fill=(255, 255, 255), anchor="lm")
    for i, line in enumerate(lines):
        d.text((10, HEAD + 4 + i * 18), line, font=_font(13), fill=INK if i == 0 else MUTED)
    y = HEAD + READ + img.height
    _gauges(d, y + 6, act)
    col = (GOOD if good else BAD) if end else CAR
    d.rectangle([0, y + GAUGE, W, y + GAUGE + BAR], fill=(232, 236, 241))
    d.rectangle([0, y + GAUGE, round(W * min(max(progress, 0), 1)), y + GAUGE + BAR], fill=col)
    if end:
        yb = (HEAD + READ + img.height // 2 - 30) if end_y is None else end_y
        d.rounded_rectangle([30, yb, W - 30, yb + 60], radius=8, fill=(255, 255, 255), outline=col, width=3)
        d.text((W // 2, yb + 22), end, font=_font(14, bold=True), fill=col, anchor="mm")
        d.text((W // 2, yb + 44), "↻ GIF replays from the start", font=_font(12), fill=MUTED, anchor="mm")
    return np.asarray(canvas)


def _readout(c, s):
    n_j = len(c.course.jumps)
    landed = sum(bool(x and x["landed"]) for x in c.results)
    done = sum(x is not None for x in c.results)
    air = f"AIRBORNE {c.t * cr.DT - c.takeoff[0]:.2f} s" if c.airborne and c.takeoff else "on the ground"
    return [f"t {c.t * cr.DT:4.1f} s · {s['v_fwd']:4.1f} m/s · lap {100 * min(s['x'] / c.course.length, 1):3.0f}%",
            f"jumps landed {landed}/{done} of {n_j} · {air}"]


def _run(policy, course_seed, x0, max_steps, on_frame, every, **env_kw):
    kw = {**getattr(policy, "env_kw", {}), **env_kw, "loop": True, "max_steps": max_steps}
    b = je.JumpBatch(1, courses=[course_seed], **kw)
    c = b.cars[0]
    b._obs = c.reset(course_seed, x0=x0)[None]
    if hasattr(policy, "reset"):
        policy.reset(1)
    a = np.zeros(2)
    while True:
        on_frame(c, a, None, c.t % every == 0)
        a = np.asarray(policy(b))[0]
        _, term, trunc, reason = c.step(a, b.jump_bonus, max_steps, b.steer_cost, b.finish_bonus)
        b._obs = c.obs()[None]
        if term or trunc:
            ep = c.summary(reason)
            on_frame(c, a, ep, True)
            return ep


def _end_text(ep):
    return f"{ep['reason']} · landed {ep['landed']}/{ep['n_jumps']} jumps · {ep['l'] * cr.DT:.1f} s"


def gif(policy, course_seed, path, who, x0=0.0, every=5, max_steps=3000, **env_kw):
    """Top-down GIF of one run. Returns the episode summary."""
    frames, st = [], {}

    def on_frame(c, a, ep, show):
        if "view" not in st:
            st["view"] = _View(c.course)
            st["bg"] = st["view"].background(c.course)
            st["trail"] = ImageDraw.Draw(st["bg"])
            st["prev"] = None
        v, s = st["view"], c.state()
        p = v.px(s["X"], s["Y"])
        if st["prev"] is not None:
            st["trail"].line([st["prev"], p], fill=AIR if c.airborne else CAR, width=2)
            R = c.d.xmat[1].reshape(3, 3)
            if not c.airborne and abs(float(R[:, 1] @ c.d.qvel[:3])) > 1.0:   # sliding sideways faster than 1 m/s
                st["trail"].ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=SKID)
        st["prev"] = p
        if not show:
            return
        img = st["bg"].copy()
        d = ImageDraw.Draw(img)
        _labels(d, v, c.course, c.results)
        h = s["heading"]
        f, l = np.array([np.cos(h), -np.sin(h)]), np.array([-np.sin(h), -np.cos(h)])
        tip, b1, b2 = np.array(p) + 9 * f, np.array(p) - 6 * f + 5 * l, np.array(p) - 6 * f - 5 * l
        d.polygon([tuple(tip), tuple(b1), tuple(b2)], fill=CAR, outline=(255, 255, 255))
        end = ep and _end_text(ep)
        frames.append(_frame(img, f"{who} · loop {course_seed}", _readout(c, s) if not ep else ["end of run", ""], a,
                             s["x"] / c.course.length, end=end, good=bool(ep and ep["finished"])))

    ep = _run(policy, course_seed, x0, max_steps, on_frame, every, **env_kw)
    save_gif(frames, path, frame_ms=1000 * cr.DT * every)
    return ep


def chase_gif(policy, course_seed, path, who, x0=0.0, every=4, max_steps=3000, t_range=None, stride=4, **env_kw):
    """3D chase-camera GIF of one run (or only the seconds in t_range = (from, to)). Returns the episode summary."""
    frames, st = [], {}
    Hh, Ww = 236, W

    def on_frame(c, a, ep, show):
        if "r" not in st:
            n = (lp.N - 1) // stride + 1
            m = cr.build_model(c.real, dict(lp.TERRAIN, nrow=n, ncol=n))
            m.hfield_data[:] = c.course.hfield()[::stride, ::stride].ravel()
            st["m"], st["d"], st["r"] = m, mujoco.MjData(m), mujoco.Renderer(m, Hh, Ww)
            st["cam"] = mujoco.MjvCamera()
            st["cam"].type = mujoco.mjtCamera.mjCAMERA_FREE
            st["az"] = None
        t = c.t * cr.DT
        s = c.state()
        az = np.degrees(s["heading"])
        st["az"] = az if st["az"] is None else st["az"] + 0.12 * ((az - st["az"] + 180) % 360 - 180)   # eased
        if not show or (t_range and not (t_range[0] <= t <= t_range[1]) and not ep):
            return
        st["d"].qpos[:], st["d"].qvel[:] = c.d.qpos, c.d.qvel
        mujoco.mj_forward(st["m"], st["d"])
        cam = st["cam"]
        cam.lookat[:] = c.d.qpos[:3] + np.array([0, 0, 0.1])
        cam.azimuth, cam.elevation, cam.distance = st["az"] + 18.0, -20.0, 2.4
        st["r"].update_scene(st["d"], camera=cam)
        img = Image.fromarray(st["r"].render())
        end = ep and _end_text(ep)
        frames.append(_frame(img, f"{who} · loop {course_seed}", _readout(c, s) if not ep else ["end of run", ""], a,
                             s["x"] / c.course.length, end=end, good=bool(ep and ep["finished"])))

    ep = _run(policy, course_seed, x0, max_steps, on_frame, every, **env_kw)
    st["r"].close()
    save_gif(frames, path, frame_ms=1000 * cr.DT * every)
    return ep
