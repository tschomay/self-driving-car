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

## Experiments

## Gate check

## Handoff to next phase
