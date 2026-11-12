"""Procedural jump courses: a straight dirt lane with ramps, as a heightfield.

The lane runs along +x from 0 to LENGTH. It is DRIVE_W wide with low banks on both sides
(so the car has something to steer away from). Every course has N_JUMPS jumps, each one of
three kinds, with randomized sizes:

  gap      kicker ramp up, a flat gap, then a downslope landing ramp (forgiving: land on the slope)
  table    kicker up, a flat table top, then a down ramp (too slow = land flat on the table)
  drop     kicker up, then flat ground (lands flat: the car has to arrive level)

Seeds: training uses any seed; validation is VAL_SEEDS (10000-10019) and the gate's test set
TEST_SEEDS (40000-40019), fresh for Phase 4 and looked at only by the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LENGTH = 75.0      # m, lane length (the hfield covers 0..LENGTH)
WIDTH = 5.0        # m, hfield width (y from -2.5 to 2.5)
DRIVE_W = 3.0      # m, drivable lane width between the banks
DX = 0.05          # m, grid spacing along x
DY = 0.125         # m, grid spacing along y
ZMAX = 1.5         # m, hfield elevation scale
N_JUMPS = 3
START_X = 1.5      # m, where the car starts
FINISH_X = LENGTH - 4.0
KINDS = ("gap", "table", "drop")
ANGLES = (24.0, 38.0)   # kicker lip angle range, deg (the face curves up from flat: z = h t^2)
HEIGHTS = (0.3, 0.6)    # kicker height range, m (times `size`)
DESIGN_SPEED = (6.0, 7.5)  # m/s, approach speed each jump is built for

VAL_SEEDS = list(range(10000, 10020))
TEST_SEEDS = list(range(40000, 40020))   # gate only

NCOL = int(round(LENGTH / DX)) + 1
NROW = int(round(WIDTH / DY)) + 1


@dataclass
class Jump:
    kind: str
    start: float      # x where the kicker starts rising
    lip: float        # x of the kicker's lip (takeoff edge)
    height: float     # kicker height, m
    angle_deg: float  # kicker lip angle
    land_start: float  # x where the landing surface begins (top of the landing ramp, or the drop's base)
    end: float        # x where the jump's terrain is back to ground level
    design_speed: float = 6.5  # approach speed the landing was built for, m/s

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Course:
    seed: int
    x: np.ndarray                     # (NCOL,) grid x
    profile: np.ndarray               # (NCOL,) centerline height
    jumps: list[Jump] = field(default_factory=list)

    def height(self, xq):
        return np.interp(xq, self.x, self.profile)

    def slope(self, xq, h=0.1):
        return (self.height(xq + h) - self.height(xq - h)) / (2 * h)

    def jump_at(self, xq):
        """Index of the jump whose stretch (from its start to the next jump's start) contains xq."""
        idx = None
        for i, j in enumerate(self.jumps):
            if xq >= j.start - 1.0:
                idx = i
        return idx

    def hfield(self, rng=None) -> np.ndarray:
        """(NROW, NCOL) elevations in [0, 1] (times ZMAX), rows along y, with banks and light roughness."""
        rng = rng or np.random.default_rng(self.seed + 7)
        y = np.linspace(-WIDTH / 2, WIDTH / 2, NROW)
        z = np.repeat(self.profile[None, :], NROW, axis=0)
        # banks: rise 0.35 m over 0.4 m outside the lane
        bank = np.clip((np.abs(y) - DRIVE_W / 2) / 0.4, 0, 1) * 0.35
        z = z + bank[:, None]
        # light roughness on flat ground only (a few cm, smooth), so ramps stay clean
        flat = np.ones(NCOL, bool)
        for j in self.jumps:
            flat &= ~((self.x > j.start - 0.5) & (self.x < j.end + 0.5))
        bumps = np.zeros(NCOL)
        for _ in range(6):
            wl = rng.uniform(1.5, 5.0)
            bumps += rng.uniform(0.003, 0.008) * np.sin(2 * np.pi * self.x / wl + rng.uniform(0, 2 * np.pi))
        z = z + (bumps * flat)[None, :]
        z = np.clip(z - z.min() + 0.02, 0, None)  # keep everything above the base
        return (z / ZMAX).astype(np.float32)

    def to_dict(self):
        return dict(seed=self.seed, jumps=[j.to_dict() for j in self.jumps])


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _arc_x(lip_x, h, v, ang_deg, z_hit):
    """x where a launch from (lip_x, h) at speed v, angle ang_deg first comes down to height z_hit."""
    a = np.deg2rad(ang_deg)
    vx, vz = v * np.cos(a), v * np.sin(a)
    dz = h - z_hit
    t = (vz + np.sqrt(vz * vz + 2 * 9.81 * max(dz, -vz * vz / (2 * 9.81) + 1e-6))) / 9.81
    return lip_x + vx * t


def make(seed: int, kinds=KINDS, size=1.0) -> Course:
    """Generate the course for `seed`. `size` scales the kicker heights (curricula, experiments).

    Like a real track, each jump is built for a design speed: the landing ramp sits where a car
    approaching at that speed comes down (aimed about 40% down the landing slope). The design
    speed varies a little between jumps (DESIGN_SPEED); approach slower and you case the gap or
    land on the flat table, faster and you overshoot onto the flat.
    Launch model (measured on the expert, 2026-11-12): leaves the lip at about the approach speed
    minus 0.6 m/s, at about the lip angle minus 3 degrees.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, LENGTH, NCOL)
    z = np.zeros(NCOL)
    jumps = []
    pos = 10.0 + rng.uniform(0, 3)    # first kicker after a run-up
    for _ in range(N_JUMPS):
        kind = kinds[rng.integers(len(kinds))]
        h = rng.uniform(*HEIGHTS) * size
        ang_deg = rng.uniform(*ANGLES)
        ang = np.deg2rad(ang_deg)
        vd = rng.uniform(*DESIGN_SPEED)
        face = 2 * h / np.tan(ang)
        start, lip = pos, pos + face
        # kicker: a transition curving up from flat ground to the lip angle, then a flat 0.15 m lip
        m = (x >= start) & (x <= lip)
        t = (x[m] - start) / face
        z[m] = np.maximum(z[m], h * t * t)
        top = (x > lip) & (x <= lip + 0.15)
        z[top] = np.maximum(z[top], h)
        lip_end = lip + 0.15
        launch = (lip + 0.1, h, vd - 0.6, ang_deg - 3.0)
        if kind == "gap":
            lh = h * rng.uniform(0.7, 1.0)
            ll = rng.uniform(2.0, 3.5)
            aim = 0.4
            x_aim = _arc_x(*launch, lh * (1 - _smoothstep(aim)))
            land_start = max(x_aim - aim * ll, lip_end + 0.6)
            m = (x >= land_start) & (x <= land_start + ll)
            z[m] = np.maximum(z[m], lh * (1 - _smoothstep((x[m] - land_start) / ll)))
            # back face of the kicker: steep drop to the gap floor
            m = (x > lip_end) & (x < lip_end + 0.3)
            z[m] = np.maximum(z[m], h * (1 - (x[m] - lip_end) / 0.3))
            end = land_start + ll
        elif kind == "table":
            dl = rng.uniform(2.0, 3.5)
            aim = 0.35
            x_aim = _arc_x(*launch, h * (1 - _smoothstep(aim)))
            land_start = max(x_aim - aim * dl, lip_end + 0.6)
            m = (x > lip_end) & (x <= land_start)
            z[m] = np.maximum(z[m], h)
            m = (x > land_start) & (x <= land_start + dl)
            z[m] = np.maximum(z[m], h * (1 - _smoothstep((x[m] - land_start) / dl)))
            end = land_start + dl
        else:  # drop
            m = (x > lip_end) & (x < lip_end + 0.3)
            z[m] = np.maximum(z[m], h * (1 - (x[m] - lip_end) / 0.3))
            land_start = lip_end + 0.3
            end = land_start
        jumps.append(Jump(kind, float(start), float(lip), float(h), float(ang_deg), float(land_start),
                          float(end), float(vd)))
        pos = end + rng.uniform(9.0, 13.0)
    assert jumps[-1].end < FINISH_X - 8.0, "course too long"
    return Course(seed, x, z, jumps)
