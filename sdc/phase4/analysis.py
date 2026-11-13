"""How a driver uses the throttle in the air (Phase 4).

For every airborne control step: the throttle change since takeoff (what turns the body), and the
pitch error the oracle would correct (landing-slope pitch minus current pitch, and the pitch rate).
A driver that uses pitch control moves the throttle with the pitch error: positive correlation
(nose too low -> more throttle). Holding the throttle gives zero change and no correlation.
"""

from __future__ import annotations

import numpy as np

from sdc.jumps import course as crs
from sdc.jumps import driver as dr
from sdc.jumps import env as je


def air_steps(policy, courses=crs.VAL_SEEDS, max_steps=1500, starts=dr.STARTS):
    rows, eps = [], []
    for seed, x0 in [(c, x0) for x0 in starts for c in courses]:
        b = je.JumpBatch(1, courses=[seed], max_steps=max_steps, **getattr(policy, "env_kw", {}))
        c = b.cars[0]
        b._obs = c.reset(seed, x0=x0)[None]
        if hasattr(policy, "reset"):
            policy.reset(1)
        u0 = None
        while True:
            a = np.asarray(policy(b))[0]
            if c.airborne:
                s = c.state()
                if u0 is None:
                    u0 = float(c.last_act[1])
                target, _ = dr.landing_slope(c.course, s)
                rows.append((a[1] - u0, np.degrees(target - s["pitch"]), np.degrees(s["pitch_rate"]),
                             c.t * 0.02 - c.takeoff[0]))
            else:
                u0 = None
            _, term, trunc, reason = c.step(a, b.jump_bonus, max_steps)
            b._obs = c.obs()[None]
            if term or trunc:
                eps.append(c.summary(reason))
                break
    r = np.array(rows) if rows else np.zeros((0, 4))
    return r, eps


def air_summary(policy, courses=crs.VAL_SEEDS) -> dict:
    r, eps = air_steps(policy, courses)
    jumps = [j for e in eps for j in e["jumps"] if j.get("pitch") is not None]
    err = np.array([j["pitch"] - j["slope"] for j in jumps]) if jumps else np.zeros(0)
    out = dict(air_steps=len(r), score=dr.score(eps),
               touch_err_mean=round(float(np.mean(err)), 1) if len(err) else None,
               touch_err_abs=round(float(np.mean(np.abs(err))), 1) if len(err) else None,
               touch_err=[round(float(e), 1) for e in err],
               touch_landed=[bool(j["landed"]) for j in jumps])
    if len(r) > 5:
        out.update(dthrottle_abs=round(float(np.mean(np.abs(r[:, 0]))), 3),
                   corr_err=round(float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]), 2) if r[:, 0].std() > 1e-6 else 0.0,
                   corr_rate=round(float(np.corrcoef(r[:, 0], r[:, 2])[0, 1]), 2) if r[:, 0].std() > 1e-6 else 0.0)
    return out


def pitch_probe(policy, courses=None, starts=(0.0,)):
    """How well the car's pitch in the air can be read out of (a) the current sensors, (b) the GRU's memory.

    Drives validation courses, records every airborne step's observation, the actor's memory (if the
    policy has one) and the true pitch; fits a linear read-out on half the courses and reports R^2 on
    the other half. R^2 near 1 = the pitch is written down there; near 0 = it isn't.
    """
    courses = courses or crs.VAL_SEEDS
    X_obs, X_mem, y, grp = [], [], [], []
    for gi, (seed, x0) in enumerate([(c, x0) for x0 in starts for c in courses]):
        b = je.JumpBatch(1, courses=[seed], **getattr(policy, "env_kw", {}))
        c = b.cars[0]
        b._obs = c.reset(seed, x0=x0)[None]
        if hasattr(policy, "reset"):
            policy.reset(1)
        while True:
            a = np.asarray(policy(b))[0]
            if c.airborne:
                X_obs.append(b._obs[0].copy())
                if getattr(policy, "h", None) is not None:
                    X_mem.append(policy.h[0, 0].numpy().copy())   # actor's memory after reading this step
                y.append(c.state()["pitch"])
                grp.append(gi % 2)
            _, term, trunc, _ = c.step(a)
            b._obs = c.obs()[None]
            if term or trunc:
                break
    y, grp = np.degrees(np.array(y)), np.array(grp)

    def r2(X):
        X = np.c_[np.asarray(X), np.ones(len(X))]
        tr_, te = grp == 0, grp == 1
        w, *_ = np.linalg.lstsq(X[tr_], y[tr_], rcond=None)
        res = y[te] - X[te] @ w
        return round(float(1 - res.var() / y[te].var()), 2)

    out = dict(air_steps=len(y), pitch_sd=round(float(y.std()), 1), r2_sensors=r2(X_obs))
    if X_mem:
        out["r2_memory"] = r2(X_mem)
        out["r2_both"] = r2(np.c_[np.asarray(X_obs), np.asarray(X_mem)])
    return out


def steering_stats(policy, courses=None):
    """Steering smoothness on the flats (away from jumps): mean |steer|, mean |change per step|, sign flips."""
    courses = courses or crs.VAL_SEEDS[:10]
    st, dst, flips = [], [], 0
    for seed in courses:
        b = je.JumpBatch(1, courses=[seed], **getattr(policy, "env_kw", {}))
        c = b.cars[0]
        b._obs = c.reset(seed)[None]
        if hasattr(policy, "reset"):
            policy.reset(1)
        prev = None
        while True:
            a = np.asarray(policy(b))[0]
            x = c.state()["x"]
            flat = x > 3 and all(abs(x - (j.start + j.end) / 2) > (j.end - j.start) / 2 + 2 for j in c.course.jumps)
            if flat:
                st.append(abs(a[0]))
                if prev is not None:
                    dst.append(abs(a[0] - prev))
                    flips += np.sign(a[0]) != np.sign(prev)
            prev = a[0] if flat else None
            _, term, trunc, _ = c.step(a)
            b._obs = c.obs()[None]
            if term or trunc:
                break
    return dict(steer_abs=round(float(np.mean(st)), 2), steer_change=round(float(np.mean(dst)), 2),
                sign_flips=round(flips / max(len(dst), 1), 2))
