# Phase 4.5 — Sim-to-real readiness: sensors and physics

- **Status:** in progress
- **Goal:** close the gaps Phase 4's handoff listed (sphere wheels, 3 mm of sag, perfect sensors, nominal physics only) on Phase 4's jump courses. Sensors are modelled on real hobby parts, the chassis is realistic, and Phase 3's physics randomization is carried into 3D. The course with turns moves to Phase 4.6.
- **Learn:** sim-to-real transfer, state estimation from noisy sensors (your pitch filter), how suspension and weight transfer shape handling, domain randomization vs. fidelity, compute and sensor delay as part of control.
- **Gate (owner, 2026-11-14):** at least 80% of jumps landed, using Phase 4's definition (airborne at least 0.3 s; within 1 s all four wheels down, no chassis contact, upright and still driving). Measured on fresh test courses 45000–45019, 3 starts each (180 jumps), with the parts-list sensors and physics drawn from a range **wider** than training (as in Phase 3).
- **Code:** `sdc/jumps/` (sim), `sdc/phase45/` (training, evaluation, reports).
- **How to reproduce the gate:** to be added.

## Phase-entry decisions (owner, 2026-11-14)

Measured first (agent, before the decisions):

| Check | Result |
|---|---|
| MuJoCo rendering an onboard camera on the container CPU, shadows off | About 30–40 frames/s per core, against about 3,000 sim steps/s. A camera makes training about 100× slower: about 5 h per 1M-step run instead of about 11 min. |
| The owner's Phase 2 CNN, one 64×64 frame, 1 thread | About 0.4 ms (about 1–2 ms expected on a Pi 5). The sensor-only network takes 0.02 ms. |

1. **Camera:** none for the gate. The gate car uses the parts-list sensors: a noisy IMU, motor RPM, a TFmini single-point lidar and a VL53L5CX 8×8 time-of-flight sensor. An optional camera step after the gate can clone the gate car into the Phase 2 CNN.
2. **Split:** Phase 4.5 covers sensors and physics on the existing jump courses, with its own gate. **Phase 4.6** covers the new course with turns (at least 80% of full courses finished, the same sensors and randomization, fresh test courses).
3. **Gate:** as above.
4. **Learner-owned code:** a pitch estimator from the noisy IMU, starting with a **complementary filter**. It can be upgraded to a Kalman filter as an experiment if gyro drift matters.
5. **Board:** a Raspberry Pi 5 plus a Pico microcontroller. In the sim, each policy gets a compute budget (about 2 ms), and its inference time counts as part of the control delay. No AI HAT or Jetson unless a camera step needs one.

## Plan

0. **Build (agent):** the realistic car, on Phase 4's courses, behind switches, so Phase 4's car can be scored on each change.
   - **Physics:** cylinder tires; soft suspension (about 30% sag) with real damping and centre-of-mass height; one motor through differentials; servo lag and deadband; battery voltage sag.
   - **Sensors:** IMU noise, bias drift, motor vibration and ±16 g clipping; one motor RPM signal; TFmini and 8×8 time-of-flight sensors with dropouts; separate sensor rates, latency and jitter.
   - **Randomization:** a training range and a wider gate range, Phase 3-style.
   - **Expert:** retuned so it still lands under the new physics, since it is the teacher.
1. **E1 — What breaks?** Phase 4's gate car and the expert, unchanged, on the realistic car. Each change is scored on its own, then all together.
2. **E2 — Your pitch filter.** How well does a complementary filter track pitch through noise, vibration and clipped landings, compared with the raw gyro and the raw accelerometer?
3. **E3 — Retrain for the real car.** Copy the expert, then fine-tune with the finish bonus, under the new sensors and physics. Compare nominal physics in training with randomized physics, scored on the gate range.
4. **E4 — Does the estimate help?** The E3 recipe with and without your filter's pitch as an input.
5. **E5 — How much delay can it take?** Add sensor and compute delay until landings collapse. This tells Phase 5 how fast the Pi's loop must be.
6. **Gate** on test courses 45000–45019.

## The realistic car (built 2026-11-14)

All switches are off by default, and then the car is Phase 4's exactly: the same model file, and the Phase 4 expert's validation score reproduces (92.8%). Code: `sdc/jumps/real.py`, plus options on `car.py`, `env.py` and `driver.py`. Tests: `tests/test_real.py`.

**Physics** (`real=`; `"all"` turns on all six):

| Switch | What changes |
|---|---|
| `tires` | Wide tires: a 3 cm tread with rounded shoulders (a capsule along the axle) instead of a sphere. A true cylinder was rejected: in MuJoCo it snags on the heightfield's grid and loses up to 2.8 m/s in a single landing step. |
| `susp` | Soft suspension: 2.4 cm of sag (30% of the 8 cm travel), damped to about critical. A 0.4 damping ratio was tried first; the springs rebounded off the kicker lips and threw the car (a pogo kick). A 0.4 kg battery pack sits on top, raising the centre of mass. |
| `drive` | One motor through open differentials: equal torque to every wheel, back-EMF drag on the mean wheel speed. A wheel in the air spins up and steals speed. |
| `servo` | The steering servo moves at a limited speed (0.12 s per 60° nominal), with a 0.01 rad deadband. |
| `battery` | Motor power scales with the pack's charge (95% nominal) and sags 7% at full throttle. |
| `delay` | Commands reach the servo and ESC 15 ms after the sensors are read (nominal): Pi compute plus the PWM frame. It is applied at 2 ms resolution with ±1 ms jitter. |

**Sensors** (`sensors=`):
- `ideal`: Phase 4's perfect sensors.
- `noisy`: Phase 4's layout, so its gate car can drive unchanged, with realistic signals:
  - The IMU is sampled every 2 ms. Each sample clips at ±16 g and ±2000°/s, and the samples are averaged per step. On top: sensor noise, motor vibration that grows with speed, a per-run bias, bias drift, and a randomly tilted mount.
  - One motor-speed signal, from hall-sensor counts every 20 ms (about 0.15 m/s steps), goes in both wheel-speed slots. It reads wrong when wheels spin in the air.
  - Every ray gets range noise and 2% dropouts.
  - The servo slot reports the command, since a hobby servo has no position feedback.
- `parts`: the parts list's layout, 76 numbers:
  - IMU (6) and motor speed (1);
  - TFmini-S, one beam 3° down, 8 m outdoors (1);
  - VL53L5CX, 8×8 zones over 45°, looking 25° down, updated at 15 Hz, about 3 m range outdoors, 4% of zones invalid per frame (64);
  - two side VL53L1X, 3 m (2);
  - last action (2).

  The two side sensors (about $10) were added to the parts list: without them, nothing sees the lane edges, and Phase 4 showed that heading matters.

**Randomized physics** (`dr=`; drawn per run from the course, the start and a physics seed, so scores don't depend on batching):

| | Training range | Gate range |
|---|---|---|
| Mass | ×0.85–1.15 | ×0.75–1.3 |
| Tire grip | 0.7–1.1 | 0.55–1.25 |
| Spring, damping | ×0.8–1.25, ×0.7–1.4 | ×0.7–1.4, ×0.6–1.7 |
| Motor torque, battery charge | ×0.85–1.1, 85–100% | ×0.8–1.15, 80–100% |
| Servo speed (s per 60°) | 0.08–0.16 | 0.06–0.2 |
| Control delay | 5–30 ms | 0–45 ms |
| IMU mount tilt | ±1° | ±2° |
| Sensor noise | ×0.5–1.5 | ×0.5–2 |
| ToF range (sunlight) | 2–4 m | 1.5–4 m |

**Expert (the teacher), retuned:** each jump's design speed + 0.3 m/s (the damper soaks up some of the kicker), and twice the in-air pitch damping (with up to 45 ms of delay, Phase 4's gains overshoot to 20–33° nose down). Settings: `sdc/phase45/common.py`. Two other ideas didn't help and were dropped: an integral term on speed, and in-air gains scaled by power-to-weight.

**Speed:** about 400–700 sim steps/s per core, against about 1,500 in Phase 4: physics now runs 2 ms at a time in Python, with the IMU sampled at each step. A 1M-step run takes about 30–40 min.

**Phase 4's gate car was rebuilt.** Its weights were lost with the old container (`runs/` is not committed), so it is being retrained from the logged commands with the same seeds.

**Test set:** courses 45000–45019 with physics seed 45, never used by experiments.

**Code review (owner's request, 2026-11-14, before E1).** A background agent reviewed the new code. It found and verified:
1. **Randomized grip could only go up.** MuJoCo uses the larger of the two surfaces' friction, and only the tires were scaled. Now the ground is scaled too.
2. **"Sensors only" also changed the physics.** The steering command was computed in float64 instead of Phase 4's float32. The difference was about 1e-9, but chaos grew it to 1.85 m within 200 steps. Fixed: with no physics switches, the realistic-sensor car now drives exactly Phase 4's trajectory (a test checks it).
3. **The 8×8 sensor updated at 12.5 Hz, not 15 Hz.** Fixed.
4. Minor: after mass changes, MuJoCo's mass-derived constants are now recomputed.

No oracle leakage was found. The `noisy` layout matches Phase 4's slot by slot, and the physics draws don't depend on batching. The numbers (sag, damping, battery sag, noise, hall counts) are plausible.

After the fixes, the expert lands 98% at nominal physics, 91% on the training range and 84% on the gate range (validation, 180 jumps each).

## Experiments

### E1 — What breaks?

- **Prediction (owner, before the run):** "I think the physics will hurt most. 45% of gates." (The physics changes hurt more than the sensors; about 45% landed with everything on at gate-range physics.)
- **Setup:**
  - Phase 4's gate car (rebuilt from its logged commands; its weights were lost with the old container) drives the realistic car unchanged.
  - Experts for comparison: Phase 4's, and the retuned 4.5 one.
  - Validation courses 10000–10019, 3 starts each (180 jumps), one car at a time.
  - Each change on its own, then stacked.
  - `python -m sdc.phase45.e1`, about 10 min on 4 cores.
- **Result:** [report](https://claude.ai/artifact/QaJLi1MkZ1Qps1DM7ShrwZ).

  | Setup | Gate car | Expert (Phase 4) | Expert (4.5) |
  |---|---|---|---|
  | Phase 4's car | 91% | 93% | 94% |
  | Wide tires | 94% | 91% | 95% |
  | Soft suspension | **52%** | 97% | 98% |
  | One motor + diffs | 88% | 94% | 96% |
  | Servo lag | 89% | 94% | 92% |
  | Battery sag | 84% | 95% | 94% |
  | Control delay | 81% | 92% | 95% |
  | Realistic sensors | 74% | 93% | 94% |
  | All physics | 73% | 84% | 98% |
  | All physics + sensors | 78% | 87% | 98% |
  | … + training-range physics | 59% | 76% | 90% |
  | … + gate-range physics | **61%** | 73% | 87% |

  - **Soft suspension alone breaks the learned car**: 38 of 60 runs end off the lane. Its side "fan" rays are fixed level to the chassis. In a turn, the car now rolls about 6° (1° with stiff springs), and the low side's rays hit the ground 0.3–0.9 m away instead of reaching the lane edge 2–5 m out. It reads a wall beside it. The experts, which read the true state, don't mind (97–98%).
  - Realistic sensors alone: 74%, again mostly off the lane.
  - Stacking isn't additive: all physics 73%; plus sensors 78% (within the noise of 180 jumps). Randomized physics is the next step down (59–61%). Drops are hit hardest (12/42 at gate range).
  - The retuned expert holds 87% at gate range, so a copy of it has room above the 80% bar, but not much.
  - The rebuilt gate car, scored as Phase 4 scored it, lands 91.1% on Phase 4's car: exactly the original's number.
  - Prediction vs. result: "physics hurts most": yes. The worst single change is physical (the suspension), but it breaks the car through its sensors (body roll tilts the rays). All physics (73%) and all sensors (74%) cost about the same. Predicted 45%, measured 61%.
- **Explanation (owner, after the run):** "Acceleration and angle are very important, and soft springs change both. Maybe moreover, the car can tip while the wheels stay flat, so the rays point different directions compared with the linear motion of the car."
- **Takeaway:** sensors bolted to a sprung body measure the body, not the car's path. Body roll and pitch must be estimated (the IMU, your filter in E2) or designed out (sensor placement), and a policy trained on a stiff car never learned to tell the two apart.

### E2 — Your pitch and roll filter

- **Owner's filter (`sdc/phase45/my_filter.py`, 2026-11-14):** a complementary filter.
  1. Blend α = 0.98 at 50 Hz: the gyro carries the angle, and the accelerometer pulls out drift within about 1 s.
  2. The accelerometer is ignored whenever the total isn't close to 1 g (in the air, landings, hard braking). Tolerance ±0.15 g (agent's default).
  3. Gyro bias: the average while the car sits still for 0.2 s before the start, subtracted.
  4. Both pitch and roll.
- **Sim support:** the car now records 0.2 s of IMU readings while sitting still before each run (like a power-up calibration). The physics is restored afterwards, so runs start exactly as before. The noise stream shifts, so E1 reruns would differ slightly.
- **Setup:**
  - The 4.5 expert drives the realistic car (all physics, noisy IMU) on the validation courses, 3 starts, at nominal, training-range and gate-range physics.
  - Estimators on the same logs: gyro only (with and without the bias fix), accelerometer only, your filter, and your filter without choice 2.
  - Errors against the true pitch and roll: on the ground, in the air, at touchdown, and the worst in a run.
  - `python -m sdc.phase45.e2`.

## Gate check

## Handoff to next phase
