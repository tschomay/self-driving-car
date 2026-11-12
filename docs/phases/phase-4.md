# Phase 4 — 3D terrain with jumps

- **Status:** in progress
- **Goal:** a 3D rigid-body sim with heightfield terrain, ramps, and an IMU (accelerometer + gyro) in the observation; a policy that jumps and lands.
- **Learn:** 3D dynamics, partial observability (airborne, the car must use memory), using throttle to control pitch mid-air, massively parallel simulation.
- **Gate (owner's decisions at entry, 2026-11-12):** ≥ 80% of jumps landed on a fresh test set of procedurally generated jump courses (ramp size and approach randomized), nominal physics. A jump counts only if the car is airborne ≥ 0.3 s. It is **landed** when all four wheels touch down, the car doesn't roll over (no body contact with the ground), and it is still driving 1 s later.
- **Code:** `sdc/jumps/` (the 3D sim), `sdc/phase4/` (training, evaluation, reports).
- **How to reproduce the gate:** to be filled at the gate.

## Phase-entry decisions (owner, 2026-11-12)

1. **Engine: MuJoCo** (4.0.1), on the CPU. A quick test ran about 117,000 physics steps/s on one core, so at 50 control steps/s training should fit the container. MJX (MuJoCo on a GPU) only if needed.
2. **Compute: CPU first, $0.** Set up Modal (account, API token, $30/month cap) only if a single run would take more than about an hour.
3. **Gate:** as above (landings, with the 0.3 s minimum airtime so slow crawls over ramps don't count), nominal physics; no physics randomization for the gate since 3D is enough new difficulty.
4. **Sensors:** IMU (accelerometer + gyro), wheel speed, and terrain-height rays ahead. A camera stays a stretch goal.
5. **Learner-owned code:** the owner writes the core of a GRU memory in their PPO (scaffolded with TODOs). Everything else is the agent's.
6. **Scope:** about 4–6 experiments plus the gate, with flexibility to go deeper when a result is worth chasing.

## The 3D sim (built 2026-11-12)

- **Car** (`sdc/jumps/car.py`): a 1/10-scale 4WD buggy in MuJoCo. Chassis 0.46 × 0.18 m (8 cm bumpers front and rear), 2.3 kg in all; four sprung wheels (8 cm travel); front steering servo (±0.4 rad); an electric motor on each wheel (top speed about 14 m/s). Physics every 2 ms, the driver acts 50 times a second (Phase 3 was 10).
- **Pitch control in the air:** the motors push against the chassis, so in the air throttle spins the wheels up and tips the nose **up**, and braking tips it **down**. Full throttle for half a second rotates the car about 70°.
- **Courses** (`sdc/jumps/course.py`): a straight 3 m dirt lane, 75 m long, with banks and light bumps, and three jumps per course. Each jump is one of:
  - **gap:** kicker, gap, downslope landing.
  - **table:** kicker, flat table top, down ramp.
  - **drop:** kicker, then flat ground.
- **Jump geometry:** kickers are 0.3–0.6 m high, curving up to a 24–38° lip. Like a real track, each landing is built for a design speed (6.0–7.5 m/s): too slow and the car cases the gap or lands flat on the table, too fast and it overshoots.
  - Validation: courses 10000–10019. Test (gate only): 40000–40019.
- **What the car senses** (20 numbers):
  - accelerometer and gyro;
  - front and rear wheel speed;
  - 8 forward "lidar" rays (3° up to 50° down) plus 2 sideways rays to the banks;
  - its last action.
  No position, orientation, or speed over the ground: in the air it must work out its pitch from the gyro and the rays.
- **Episode:** ends at the finish line, on a rollover, off the lane, when stuck for 2 s, or at 30 s. Reward: metres of forward progress, plus 10 per landed jump.
- **Landed** (gate definition): airborne ≥ 0.3 s; within 1 s of touchdown all four wheels down at once, no chassis contact, and at the end of that second upright and still driving. A ramp not jumped counts as not landed.
- **Hand-coded expert** (`sdc/jumps/driver.py`, reads the true state): approaches each jump at its design speed and keeps the wheels straight in the air. Its pitch control predicts the landing slope and matches it with throttle, measured against the throttle at takeoff (the wheels' spin is what turns the body). Validation: **60/60 jumps landed (100%)**. The same driver holding the throttle steady in the air lands 39/60 (65%) and rolls over on 9 of 20 courses.
- **Speed:** about 1,500 driver steps/s per CPU core, so 1M steps ≈ 11 min of simulation per run.

## Plan

1. **E1 — Can PPO learn to jump from scratch?** Your PPO with the Phase 1 network (no memory), real sensors only, 1M steps. Baselines: the expert (100%) and the expert without air control (65%).
2. **E2 — What does throttle in the air do?** Take E1's policy and lock the throttle while airborne, or train with in-air throttle disabled. Does the policy use pitch control, and what is it worth?
3. **E3 — Memory in the air.** In free fall the accelerometer reads zero, so pitch must be pieced together from the gyro over time. Your GRU (scaffolded, you choose the core logic) vs. the memoryless network, maybe vs. a stack of recent frames.
4. **E4 — Start from the expert** (Phase 3.5's lesson): behaviour-clone the expert, then fine-tune with low exploration noise. Only if E1–E3 fall short of 80%.
5. **Gate** on test courses 40000–40019.

## Experiments

### E1 — Can PPO learn to jump from scratch?

- **Prediction (owner, before the run):** "75% and only kind of" (it will only partly learn to use throttle in the air).
- **Setup:** the owner's PPO (textbook clip, GAE, squared-error critic) with the Phase 1 network (two 64-unit layers, no memory), real sensors only (20 inputs), tanh-squashed actions, γ 0.99, λ 0.95, linear LR decay from 3e-4, 8 cars × 256 steps per update, 1M steps, seeds 0–2, 1 torch thread. Early stopping: best validation landing rate among checkpoints every 50k steps. `python -m sdc.phase4.train --seed S --steps 1000000 --out runs/p4-e1-sS`.

## Gate check

## Handoff to next phase
