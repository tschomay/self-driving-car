"""Phase 4.6: the hand-coded teacher for loop courses (it reads the true state: an expert, not a gate car).

LoopExpert = Phase 4.5's ExpertDriver (speed into each jump, pitch control in the air) plus turns:
  steering   feed-forward for the curve just ahead (the angle a car of this wheelbase needs for that radius) plus
             the same feedback as on the straight lanes (back to the centreline, heading along it).
  speed      the lowest of: the jump speed, and for every point within braking distance the speed that corner can
             be taken at (lateral acceleration A_LAT) plus what braking at A_BRAKE would still shed before it.
A_LAT is one fixed number, not the car's true grip: a student can't sense grip ahead, so the teacher doesn't use it.
"""

from __future__ import annotations

import numpy as np

from sdc.jumps import car as cr
from sdc.jumps import driver as dr

WHEELBASE = 0.30


class LoopExpert(dr.ExpertDriver):
    def __init__(self, a_lat=3.5, a_brake=3.0, v_cruise=None, look=0.15, k_ff=1.0, **kw):
        kw.setdefault("plan", False)
        kw.setdefault("v_target", 7.4)
        kw.setdefault("kd", 1.0)
        super().__init__(**kw)
        self.a_lat, self.a_brake, self.look, self.k_ff = a_lat, a_brake, look, k_ff
        self.v_cruise = v_cruise if v_cruise is not None else self.v_target
        self._plans = {}

    def corner_speed(self, course):
        """Per centreline point: the speed limit from the corners at and ahead of it."""
        if course.seed not in self._plans:
            v = np.minimum(np.sqrt(self.a_lat / np.maximum(np.abs(course.kappa), 1e-6)), 20.0)
            ds = course.ds
            for _ in range(2):                       # backward pass (twice round, for the wrap): brake in time
                for i in range(len(v) - 1, -1, -1):
                    nxt = v[(i + 1) % len(v)]
                    v[i] = min(v[i], np.sqrt(nxt * nxt + 2 * self.a_brake * ds))
            self._plans[course.seed] = v
        return self._plans[course.seed]

    def jump_speed(self, car, s):
        """The speed to carry into the next jump (as ExpertDriver: steady, the course plan, or the car's map)."""
        if self.use_map and getattr(car, "map_next_speed", None):
            return car.map_next_speed + self.plan_offset
        if self.plan:
            x = s["x"] % car.course.length
            nxt = [j for j in car.course.jumps if j.lip > x]
            return (nxt[0] if nxt else car.course.jumps[0]).design_speed + self.plan_offset
        return self.v_target

    def act(self, car):
        if car.airborne:
            return super().act(car)
        s = car.state()
        c = car.course
        self._u0[id(car)] = None
        v = max(s["v_fwd"], 0.5)
        kap = float(c.curvature(s["x"] + self.look * v))
        ff = self.k_ff * np.arctan(WHEELBASE * kap) / cr.STEER_MAX
        steer = np.clip(ff - 1.0 * s["y"] - 1.5 * s["yaw"] - 0.15 * s["yaw_rate"], -1, 1)
        i = int((s["x"] % c.length) / c.ds) % len(c.center)
        x = s["x"] % c.length
        near_jump = any(j.start - 6.0 <= x <= j.end for j in c.jumps)
        target = self.jump_speed(car, s) if near_jump else self.v_cruise
        target = min(target, float(self.corner_speed(c)[i]))
        e = target - s["v_fwd"]
        throttle = 0.8 * e + 0.06 * target
        return np.array([steer, np.clip(throttle, -1, 1)], np.float32)


class OutLapDriver(LoopExpert):
    """The out-lap (owner, 2026-11-16): a person drives one slow lap by remote while the car records. This
    stand-in reads the true state, as a person watching the car does: it rolls round at V, keeps to the middle,
    and gives it more throttle up the ramps so it doesn't stall on a kicker (up to V_RAMP while climbing)."""

    V, V_RAMP = 2.0, 3.0

    def __init__(self, **kw):
        super().__init__(v_target=self.V, v_cruise=self.V, **kw)

    def act(self, car):
        a = super().act(car)
        if car.airborne:
            return a
        s = car.state()
        target = self.V_RAMP if s["pitch"] > np.deg2rad(6) else self.V
        a[1] = np.clip(0.8 * (target - s["v_fwd"]) + 0.06 * target + 1.2 * np.sin(s["pitch"]), -1, 1)
        return a
