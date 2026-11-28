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
from sdc.jumps import env as je

WHEELBASE = 0.30


class LoopExpert(dr.ExpertDriver):
    def __init__(self, a_lat=3.5, a_brake=3.0, v_cruise=None, look=0.15, k_ff=1.0, use_turn_map=False, jump_v=None,
                 kv=0.8, kv_ff=0.06, tc=None, k_rate=0.15, **kw):
        kw.setdefault("plan", False)
        kw.setdefault("v_target", 7.4)
        kw.setdefault("kd", 1.0)
        super().__init__(**kw)
        self.a_lat, self.a_brake, self.look, self.k_ff = a_lat, a_brake, look, k_ff
        # E3: plan from the car's own out-lap map (curvature along its odometer) instead of the true track: the same
        # knowledge the student gets as inputs. Lane-centring feedback still reads the true state.
        self.use_turn_map = use_turn_map
        self._mplans = {}
        self.v_cruise = v_cruise if v_cruise is not None else self.v_target
        self._plans = {}
        # E9 (owner's idea: try settings, stitch the best): approach speed per jump kind, e.g. {"double": 8.0}
        self.jump_v = dict(jump_v or {})
        # truggy physics (stronger motor): speed-loop gain and feed-forward, and optional traction control: throttle
        # cut back when the driven wheels spin faster than the car moves by more than tc m/s (a driver's feel)
        self.kv, self.kv_ff, self.tc = kv, kv_ff, tc
        self.k_rate = k_rate            # yaw-rate damping in the steering (more stops landing fish-tails on low grip)

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

    def map_speed(self, car):
        """The speed limit at the car's odometer position, planned on its mapped curvature."""
        ds, kap = car.tmap
        if not len(kap):
            return 20.0
        key = id(kap)
        if key not in self._mplans:
            v = np.minimum(np.sqrt(self.a_lat / np.maximum(np.abs(kap), 1e-6)), 20.0)
            for _ in range(2):
                for i in range(len(v) - 1, -1, -1):
                    nxt = v[(i + 1) % len(v)]
                    v[i] = min(v[i], np.sqrt(nxt * nxt + 2 * self.a_brake * ds))
            if len(self._mplans) > 2000:
                self._mplans.clear()
            self._mplans[key] = (kap, v)          # keeps the array alive, so its id stays unique
        v = self._mplans[key][1]
        return float(v[int(round(car.dead.s / ds)) % len(v)])

    def jump_speed(self, car, s):
        """The speed to carry into the next jump (as ExpertDriver: steady, the course plan, or the car's map)."""
        if self.use_map and getattr(car, "map_next_speed", None):
            return car.map_next_speed + self.plan_offset
        if self.jump_v:
            x = s["x"] % car.course.length
            nxt = [j for j in car.course.jumps if j.end >= x]
            j = nxt[0] if nxt else car.course.jumps[0]
            return self.jump_v.get(j.kind, self.v_target)
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
        kap = car.map_kappa(self.look * v) if self.use_turn_map else float(c.curvature(s["x"] + self.look * v))
        ff = self.k_ff * np.arctan(WHEELBASE * kap) / cr.STEER_MAX
        steer = np.clip(ff - 1.0 * s["y"] - 1.5 * s["yaw"] - self.k_rate * s["yaw_rate"], -1, 1)
        i = int((s["x"] % c.length) / c.ds) % len(c.center)
        x = s["x"] % c.length
        near_jump = any(j.start - 6.0 <= x <= j.end for j in c.jumps)
        target = self.jump_speed(car, s) if near_jump else self.v_cruise
        target = min(target, self.map_speed(car) if self.use_turn_map else float(self.corner_speed(c)[i]))
        e = target - s["v_fwd"]
        throttle = self.kv * e + self.kv_ff * target
        if self.tc is not None and throttle > 0:
            spin = float(car.d.qvel[car.spin_dof].mean()) * car.wheel_r - s["v_fwd"]
            throttle *= float(np.clip(1.0 - (spin - self.tc) / self.tc, 0.0, 1.0)) if spin > self.tc else 1.0
        if getattr(car, "expo", 0.0):   # the same steering angle through the car's expo curve
            steer = je.expo_inverse(float(steer), car.expo)
        return np.array([steer, np.clip(throttle, -1, 1)], np.float32)


# The teacher re-tuned for the truggy physics (2026-11-18, 100 practice loops at gate range, 75 -> 94 laps):
# cautious corners, traction control, more yaw-rate damping (landing fish-tails on sliding grip), firmer air control.
TRUGGY = dict(a_lat=2.5, a_brake=2.5, tc=0.5, v_target=6.8, v_cruise=7.4, k_rate=0.3, kp=6.0)


# validation, gate range: 50 / 60 out-laps as tuned for the old physics; 54 with traction control and a coasting
# roll-back (the ESC has no reverse); 55 carrying 4.0 m/s over jumps (4.5: 50)
OUTLAP_TRUGGY = dict(tc=0.5, k_rate=0.3, kp=6.0, v_jump=4.0)   # the out-lap stand-in's settings on the truggy physics


class OutLapDriver(LoopExpert):
    """The out-lap (owner, 2026-11-16): a person drives one slow lap by remote while the car records. This
    stand-in reads the true state, as a person watching the car does: it rolls round at V and keeps to the middle.
    Over a jump (from 3 m before the kicker to its end) it carries V_JUMP and doesn't brake on the way down, so
    it has the momentum for the next face: at V a heavy car stalls in the valley of a double (E2)."""

    V, V_JUMP, V_RETRY = 2.0, 3.5, 4.5

    def __init__(self, v_jump=None, **kw):
        super().__init__(v_target=self.V, v_cruise=self.V, **kw)
        if v_jump is not None:
            self.V_JUMP, self.V_RETRY = v_jump, v_jump + 1.0
        self._st = {}

    def act(self, car):
        a = super().act(car)
        st = self._st.setdefault(id(car), dict(stall=0, back=0, charge=False))
        if car.t <= 1:
            st.update(stall=0, back=0, charge=False)
        if car.airborne:
            return a
        s = car.state()
        x = s["x"] % car.course.length
        on_jump = any(j.start - 3.0 <= x <= j.end for j in car.course.jumps)
        if not on_jump:
            st.update(stall=0, back=0, charge=False)
        # stalled on a face (a heavy car on low grip can't climb from a standstill): roll back and take a run at it,
        # as a person would
        st["stall"] = st["stall"] + 1 if (on_jump and s["v_fwd"] < 0.3 and s["pitch"] > np.deg2rad(10)) else 0
        if st["stall"] >= 15 and st["back"] == 0:
            st["back"], st["charge"] = 45, True
        if st["back"] > 0:
            st["back"] -= 1
            # an ESC brake has no reverse (truggy physics): coast and let the face roll the car back
            return np.array([0.0, 0.0 if "esc" in car.real else -0.8], np.float32)
        target = (self.V_RETRY if st["charge"] else self.V_JUMP) if on_jump else self.V
        thr = self.kv * (target - s["v_fwd"]) + self.kv_ff * target + 1.2 * np.sin(s["pitch"])
        if self.tc is not None and thr > 0:   # traction control, as the race teacher (truggy physics)
            spin = float(car.d.qvel[car.spin_dof].mean()) * car.wheel_r - s["v_fwd"]
            thr *= float(np.clip(1.0 - (spin - self.tc) / self.tc, 0.0, 1.0)) if spin > self.tc else 1.0
        a[1] = np.clip(thr, -0.1 if on_jump else -1, 1)
        if on_jump:
            a[0] = np.clip(a[0], -0.3, 0.3)       # easy on the wheel over a jump: wheels spinning on a face slew the car
        return a
