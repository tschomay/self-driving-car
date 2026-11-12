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
        b = je.JumpBatch(1, courses=[seed], max_steps=max_steps)
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
