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

## Plan

To be written once the sim is built.

## Experiments

## Gate check

## Handoff to next phase
