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
- **Result (validation, 180 jumps = 20 courses × 3 starts):** [report](https://claude.ai/artifact/FgEYg5zyvbByT7YJpStNWW). Kept models land **36%, 35%, 55%** (mean 42%).

  | Driver | Landed | Air throttle ↔ pitch error | Pitch − slope at touchdown |
  |---|---|---|---|
  | Hand-coded expert | 99% | 0.83 | −2° |
  | Expert, throttle held in the air | 61% | 0 | +18° |
  | Straight ahead, throttle 0.5 | 43% | 0 | −15° |
  | Your PPO (3 seeds) | 35–55% | 0.01–0.13 | +6 to +13° (nose high) |

  - It barely uses the throttle in the air, and lands nose high.
  - About a quarter of runs end off the lane after a landing, and as many in a rollover.
  - Re-scored curves peak at 48–55% around 550k–750k steps, then slide (as in Phase 3).
  - Surprises:
    - Holding half throttle and driving straight does about as well as the trained policies.
    - Landings are chaotic: one network scored 55% evaluated as a batch and 40% one car at a time, from float rounding alone. Experiments now score from 3 start points (180 jumps), and checkpoints are re-scored that way (`python -m sdc.phase4.rescore`).
  - Prediction vs. result: 75% predicted, 42% measured. "Only kind of" using the air throttle: correct, it barely does.
- **Explanation (owner, after the run):** "I don't think what it does with the throttle matters too much while it's still spazzing out. It probably never learned because it's barely driving in the first place." From the GIFs: the car seems to jump even on flat ground, and the steering saws back and forth again ("no fast/reliable car drives like it's having a seizure"). Their guess: it aims for an exact centre line and overshoots; being content with "close enough", a PID-like approach, or knowing where the throttle and steering currently are might help. Asked for a background review of the sim first, then debugging, and to come back with guidance.

### After E1: sim review and steering debug (owner's request, 2026-11-13)

[Report](https://claude.ai/artifact/Y3T1ZzdKaVt9nC2pQnYgj1). A background code review of the sim, then four debug runs (1M steps each, before the sim fixes).

- **"Jumping on the flats" was mostly the renderer.**
  - The GIF drew the ground from the smooth course plan, 2–8 cm below the real surface, and stretched heights 1.75×.
  - In the sim, wheels never leave flat ground at any speed, even with full-lock sawing. Real flat-ground air is only bounces after landings and hits on the banks.
- **Sim fixes from the review:**
  - The GIF draws the true surface at one scale for both axes.
  - The tires no longer float 1 cm above the ground (a contact margin; its gap setting has no effect in MuJoCo 4).
  - "On the ground" now means within 1 cm, measured by distance.
  - The accelerometer is averaged over each 20 ms step.
  - "Off the lane" ends the run when the car's centre crosses the lane edge (|y| > 1.5 m).
  - The expert's landing guess uses the true ground.
  - After the fixes: expert 96%, expert without air control 54% (180 jumps).
- **The car couldn't see its heading.** Its rays pointed straight ahead and straight sideways, which read the same for ±9° of heading. Even the expert's steering law, given only the sensors, weaves off the lane.
  - Added an optional fan of 6 level rays (20°, 35°, 60° left and right): `--env fan=true`.
- **The sawing comes from the exploration noise.** The noise is about 1 (the whole steering range), 50 times a second. The servo can't follow it, so sawing costs nothing (as in Phase 3).

| Run | Steering change per step | Sign flips | Landed (180 jumps) |
|---|---|---|---|
| E1 seed 0 | 1.37 | 98% | 36% |
| Fan rays only | 1.88 | 98% | 55% |
| Noise schedule 0.5 → 0.1 only | 0.32 | 54% | 46% |
| Fan + noise schedule, seed 0 | 0.25 | 42% | 57% |
| Fan + noise schedule, seed 1 | 0.10 | 27% | 48% |

- **Owner's centre-line guess:** there is no centre-line target in the reward. The weaving came from missing heading information, and the sawing from the noise. The policy already sees its last command, though not the servo's actual angle.
- **Next:** fan rays + noise schedule on the fixed sim, 3 seeds (`runs/p4-e1b-s*`), as the new baseline for E2/E3.

### E3 — Does your GRU memory help in the air?

- **Owner's GRU (`sdc/phase4/my_gru.py`, 2026-11-13):**
  1. **Update:** blend with an update gate plus a reset gate (a standard GRU). "B is the best blend where older memories are kept but down weighted. The gate allows a cutoff ... so the rest of the lap doesn't impact this jump."
  2. **New course:** zero the memory, "so each course starts fresh".
  3. **Shape:** 64 numbers, beside the sensors (the actor and critic layers see sensors and memory), a separate memory for actor and critic "so the critic doesn't distort memory".
  4. **Training chunks:** 64 steps, which "covers a full flight plus the landing".
- **Also adopted for E3 on (owner agreed, 2026-11-13):**
  - heading fan rays;
  - the actual steering angle in the observation;
  - a steering-change cost of 0.03 × |Δsteer| per step;
  - the noise schedule (0.5 → 0.1).
- **Prediction (owner, before the run):** "70% and yes" (the GRU car lands about 70% and uses the throttle in the air).
- **Setup:** 1M steps, seeds 0–2 for each car, everything above on, 1 torch thread.
  - No memory: `python -m sdc.phase4.train --seed S --out runs/p4-e3-mlp-sS --env fan=true --env servo=true --env steer_cost=0.03 --set 'std_cap=[0.5,0.1]'`
  - GRU: the same with `sdc.phase4.train_rnn` and `--out runs/p4-e3-gru-sS`.
- **Result (validation, 180 jumps per seed):** [report](https://claude.ai/artifact/K5RJAHMJpfaPYvP8K3k7qH). No clear gain from memory.

  | Car | Landed (batched / one car at a time) | Air throttle ↔ pitch error | Steering change per step |
  |---|---|---|---|
  | No memory, seeds 0–2 | 69/61/41% · 57/51/42% | −0.36 to −0.08 | 0.02–0.07 |
  | Your GRU, seeds 0–2 | 57/58/49% · 65/61/43% | −0.08 to +0.14 | 0.00–0.05 |
  | Expert | 96% | +0.81 | 0.00 |

  - Steering is calm now (E1: 1.37 per step).
  - Neither car uses the throttle in the air: after takeoff it moves 0.02–0.19 (expert 0.37).
  - The GRU's memory holds the pitch (straight-line read-out R² up to 0.84). The current sensors give most of it too (0.8–0.9: the downward rays see the ground). The policy acts on neither.
  - Endings shifted to "off the lane" and "stuck" (casing a gap and stopping).
  - Learning curves swing from 0% to 60% between checkpoints in both cars.
  - The seed spread (about 20 points), and the batched vs one-at-a-time difference (up to 8 points per seed), are bigger than any memory effect.
  - Prediction vs. result: 70% predicted, about 55% measured. "Uses the throttle in the air": no.
- **Explanation (owner, after the run):** "The training curve looks like total noise. It isn't learning anything. Maybe the task is too hard or nuanced for our setup? Or the memory window is too long? Like your setup for a jump matters so much but it's far removed from the landing. And what you need to do driving is adapt in a lot of ways rather than one consistent thing."
- **Agent's follow-up:**
  - The validation curve (one pass per checkpoint) is noisy. The training episodes (about 2,500 per run) do show slow learning: E1 seed 2 lands 20% → 50% over training, GRU seed 1 9% → 52%, no-memory E3 seed 0 11% → 33%.
  - The owner's "setup far from the landing" point holds. With γ 0.99 a reward 2 s away counts 0.37 and one 4 s away 0.17. The jump bonus comes 1 s after touchdown, so the approach speed is credited weakly.
  - The 64-step chunk limits how far back the memory is trained, not how far back reward credit reaches. That is set by γ and λ.
- **Takeaway:** memory alone didn't help; credit reaching back to the approach (γ) and a starting point that already jumps (E4) are the next levers.
- **Resume:** training saves its state at every evaluation and resumes after a container restart. The first E3 baseline runs were lost to a restart at 650k–750k steps.

## Gate check

## Handoff to next phase
