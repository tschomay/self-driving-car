# Physics: what is shared with the self-driving-car sim

Source of truth: `tschomay/self-driving-car` branch `ccr-e2fca4fb-g9rode` at `7b1cd69` (the full-size truggy, not yet
on main; loops and teacher logic as on main `0cc4e42`). Renderer: `web/shared/`, the same code as the race replay viewer;
each car is drawn at the physics car's geometry (`main.js` GEOM).

## The same

- **Engine:** MuJoCo 4.0.1 (the official WebAssembly build of the version in that repo's `requirements.txt`), 2 ms
  steps, `implicitfast`, elliptic friction cone.
- **Car:** `car.model_xml(real="truggy2")`: the full-size 1/8 truggy (`size8`: 380 mm wheelbase, 356 mm track,
  143 mm tyres, 4.0 kg, 34 mm ride height, 296 N/m springs, 1.25 N·m-per-wheel motor to 18 m/s), with the tyre's
  spin inertia set explicitly (`wheels`, 6.5e-4 kg·m²), on Phase 4.5's realistic car (capsule tires, soft suspension
  with 30% sag, one motor through diffs) plus the Phase 4.6 review switches: ESC brake (no reverse, capped at 0.65
  of motor torque, light drag brake at neutral), limited-slip diffs, heavy wheels, and the sliding-grip drop (up to
  25% from 0.3 to 1.5 m/s of contact-patch slip). Rollover counts only on the ground (`groundroll`).
- **Actuators:** 15 ms command delay (±1 ms), servo 0.12 s per 60° with a 0.01 rad deadband, battery at 95% with 7%
  sag, steering expo 0.4 (the Phase 4.6 gate car's).
- **Tracks:** Phase 4.6 loops from `loop.make`, exported by `tools/export_loops.py` and rebuilt in `src/loop.js`
  exactly as `LoopCourse.hfield()` builds them (test: `node test/loops.mjs`; 12 of 1.6 million cells differ by up to
  2 cm on bank slopes, from rounding in the export). Validation 10000–10019 and practice 100000–100019; the test
  loops are left out.
- **AI:** `LoopExpert` with `teacher.TRUGGY` (Teacher), corners at 3.0 m/s² (Teacher+), and Phase 4.7's
  `racer.TEACHER` on truggy2 (Racer: doubles at 7.8 m/s). The game adds a lane offset away from jumps so the three
  don't queue on the centreline.

## Checked against the training sim

`tools/py_feel.py` (Python, the sim itself) and `test/feel.mjs` (this port) run the same tests on flat ground, mean
of five runs nudged by 1 µm (the sim is chaotic: a 1 µm nudge moves the car metres within 4 s, so trajectories
can't be compared one to one). truggy2, grip 1.0:

| Test | Training sim | Port |
|---|---|---|
| 0–6 / 0–10 m/s | 0.84 / 1.48 s | 0.84 / 1.48 s |
| Stopping from 8.0 m/s | 3.93 m | 3.89 m |
| Half lock at 5 m/s | 2.70 m radius, 0.93 g | 2.70 m, 0.93 g |
| In the air at 7 m/s, 0.5 s full throttle / brake / neutral | +55.8° / −57.3° / −9.4° pitch | +55.9° / −57.5° / −9.5° |
| Lift at full lock from 7 m/s | 149° of rotation | 150° |

At grip 0.55 they agree as closely (stopping 7.66 / 7.67 m, lift 73° / 74°).

## Different

- Several cars share one world; chassis boxes collide with each other.
- Your car is nominal; each AI car draws from `real.RANGES["train"]`. The surface sets grip 1.0 / 0.75 / 0.55.
- No sensors; the game reads the true state.
- Crashes follow the Phase 4.7 race rules (rolled over, off the lane, stuck, spun round) but off the lane allows a
  second on the banks, and the marshal returns the car after 1 s instead of a 7 s penalty.
