"""Hand-coded drivers for the jump course (they read the oracle state, so they're experts, not gate cars).

ExpertDriver:
  on the ground: steer back to the lane centre; hold a target speed (enough to clear the jumps).
  in the air: predict where it will land (ballistic arc vs. the terrain), and use the motor's
  reaction torque to match the car's pitch to the landing slope: throttle tips the nose up,
  brake tips it down (a PD controller on pitch).
NoAirControl: the same driver, but it holds the throttle steady in the air (what pitch control buys).

Phase 4.5 options: `ki` adds an integral term to the speed control (holds the target speed on a weak motor or a
flat battery), and `adapt=True` aims the in-air pitch control at the pitch expected when the command arrives (pitch + rate x the
car's true control delay: an oracle, like Phase 3's hand-coded driver knowing the grip).
"""

from __future__ import annotations

import numpy as np

from sdc.jumps import env as je

G = 9.81


def landing_slope(course, s):
    """Pitch angle of the terrain where the ballistic arc from state `s` meets the ground."""
    x, z, vx, vz = s["x"], s["z"] - 0.10, s["vx"], s["vz"]
    for _ in range(150):
        x += vx * 0.01
        vz -= G * 0.01
        z += vz * 0.01
        if z <= course.height(x) and vz < 0:
            break
    return float(np.arctan(course.slope(x))), x


class ExpertDriver:
    def __init__(self, v_target=6.5, kp=4.0, kd=0.5, air_control=True, lead=0.0, plan=True, plan_offset=0.0,
                 ki=0.0, adapt=False):
        self.plan, self.plan_offset = plan, plan_offset
        self.ki, self.adapt, self._i = ki, adapt, {}
        self.v_target, self.kp, self.kd, self.air_control, self.lead = v_target, kp, kd, air_control, lead
        self._u0 = {}

    def act(self, car: je.JumpCar):
        s = car.state()
        # straight in the air (steered spinning wheels twist the car); gentle on the ground
        steer = 0.0 if car.airborne else np.clip(-1.0 * s["y"] - 1.5 * s["yaw"] - 0.15 * s["yaw_rate"], -0.6, 0.6)
        if not car.airborne:
            self._u0[id(car)] = None
        if car.airborne and self.air_control:
            # The wheels' spin is what turns the body: a throttle equal to the one at takeoff leaves
            # the spin (and so the pitch) alone; more spins them up (nose up), less slows them (nose down).
            u0 = self._u0.get(id(car))
            if u0 is None:
                u0 = self._u0[id(car)] = float(car.last_act[1])
            target, _ = landing_slope(car.course, s)
            pitch = s["pitch"]
            if self.adapt and "delay" in car.real:   # act on the pitch expected when this command arrives
                pitch += s["pitch_rate"] * car.phys.delay
            throttle = u0 + self.kp * (target + self.lead - pitch) - self.kd * s["pitch_rate"]
        elif car.airborne:
            throttle = float(car.last_act[1])     # hold whatever it took off with
        else:
            v = self.v_target
            if self.plan:   # approach each jump at the speed it was built for (oracle knowledge)
                nxt = [j for j in car.course.jumps if j.lip > s["x"]]
                v = (nxt[0] if nxt else car.course.jumps[-1]).design_speed + self.plan_offset
            e = v - s["v_fwd"]
            i = 0.0
            if self.ki:
                i = self._i[id(car)] = float(np.clip(self._i.get(id(car), 0.0) + e * 0.02, -0.4 / self.ki, 0.4 / self.ki))
                if car.t <= 1:
                    i = self._i[id(car)] = 0.0
            throttle = 0.8 * e + 0.45 + self.ki * i
        return np.array([steer, np.clip(throttle, -1, 1)], np.float32)

    def __call__(self, batch: je.JumpBatch):
        return np.stack([self.act(c) for c in batch.cars])


STARTS = (0.0, -0.3, 0.3)   # start offsets (m) for steadier evaluation: each course driven from 3 starts


def run(policy, courses, n_envs=None, starts=(0.0,), **batch_kw):
    """Drive each course once per start offset with `policy(batch) -> actions`; return the episode summaries.

    The car's outcome on a jump is chaotic (tiny differences, even float rounding in a batched
    network, can flip a landing), so experiments score from several starts (STARTS: 3 x 60 jumps).
    """
    batch_kw = {**getattr(policy, "env_kw", {}), **batch_kw}   # a trained policy carries its env options (e.g. fan)
    jobs = [(c, x0) for x0 in starts for c in courses]
    n_envs = n_envs or min(len(jobs), 10)
    out = []
    for s in range(0, len(jobs), n_envs):
        chunk = jobs[s:s + n_envs]
        b = je.JumpBatch(len(chunk), courses=[c for c, _ in chunk], **batch_kw)
        b._obs = np.zeros((len(chunk), b.obs_dim), np.float32)
        for i, c in enumerate(b.cars):
            b._obs[i] = c.reset(chunk[i][0], x0=chunk[i][1])
        done = np.zeros(len(chunk), bool)
        ep = [None] * len(chunk)
        reset_fn = policy.reset if hasattr(policy, "reset") else None
        if reset_fn:
            reset_fn(len(chunk))
        while not done.all():
            a = policy(b)
            _, _, term, trunc, infos = b.step(a)
            for i in np.flatnonzero((term | trunc) & ~done):
                ep[i] = infos[i]["episode"]
                done[i] = True
        out += ep
    return out


def score(episodes):
    jumps = [j for e in episodes for j in e["jumps"]]
    landed = sum(bool(j["landed"]) for j in jumps)
    airs = [j["air"] for j in jumps if j.get("air")]
    return dict(landed=landed, jumps=len(jumps), rate=round(landed / max(len(jumps), 1), 3),
                finished=round(np.mean([e["finished"] for e in episodes]), 3),
                mean_air=round(float(np.mean(airs)), 3) if airs else None,
                missed=sum(1 for j in jumps if j.get("missed")),
                by_kind={k: f"{sum(bool(j['landed']) for j in jumps if j['kind'] == k)}/"
                            f"{sum(1 for j in jumps if j['kind'] == k)}" for k in ("gap", "table", "drop")},
                reasons={r: sum(e["reason"] == r for e in episodes) for r in {e["reason"] for e in episodes}})
