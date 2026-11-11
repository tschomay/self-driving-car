# Phase 3.5 — Speed

- **Status:** in progress
- **Goal:** close the lap-time gap to the hand-coded driver without losing robustness. Phase 3's gate car laps in 41.9 s at nominal (test tracks); the same recipe's other seeds in about 34.5 s (validation); the hand-coded driver, which knows the grip and the dirt, in 25.4 s.
- **Learn:** the speed vs. reliability trade-off, multi-objective model selection, reward design for speed, what knowing the track (dirt locations) is worth.
- **Gate (owner's decisions at entry, 2026-11-11):**
  - **Reliability:** ≥ 85% of single laps from a standing start finished, with gate-range physics (grip 0.5–1.3×, mass ±35%, delay 0–120 ms, noise ≤ 4%), 5 draws per track = 100 laps. Same bar as Phase 3.
  - **Speed:** mean lap at **nominal** physics on the same 20 test tracks ≤ **1.2×** the hand-coded driver's mean lap there (about 30.5 s if it stays near 25.4 s). The owner chose 1.2× over the provisional 1.15×. For scale: the hand-coded driver *without* dirt knowledge laps in 29.1 s (1.15×).
  - **Test set:** fresh tracks 30000–30019 (`trk.TEST35_SEEDS`), physics seed 67890 (`robust.TEST35_PHYSICS_SEED`), looked at only by the gate. A retry reuses the same set, as in Phase 3. Experiments report on the validation set (tracks 10000–10019).
  - **What the car may know while driving:** only what a real car could: its sensors, plus anything it learns on the track itself (for example a grip map from an out-lap). Sim-only inputs such as the true grip are allowed in training only. If track memory is tried, the out-lap rule is decided then.
- **Scope:** about 4–6 experiments plus the gate. If progress stalls, go on to Phase 4 with what we have.
- **Code:** `sdc/phase3/` (training, grip estimator, evaluation) and `sdc/racer/` (sim), reused from Phase 3; new Phase 3.5 code goes in `sdc/phase35/`.
- **How to reproduce the gate:** to be written.

## Plan

Starting point: Phase 3's winning recipe (the owner's PPO, realism 3, true grip as a training-only input, lap bonus 100, `std_cap=[0.5, 0.1]`, γ 0.997, 1M steps, driving on the 1 s window grip estimate). Phase 3's checkpoints didn't survive the container, so E1 retrains it.

1. **E1 — Pick for speed, not just safety.** Score every checkpoint for both reliability and lap time; choose the fastest with ≥ 90% validation gate range. *Done: worth up to 4 s, but nothing gets under about 31 s.*
2. **E2 — The ceiling under the car's own constraints.** The hand-coded driver given only what the RL car knows. *Done: 34.8 s, the same as the RL car; the gap is knowledge of the track.*
3. **E3 — What a perfect track map is worth.** The RL car sees the true grip for the next 40 m (oracle map, exact position): the upper bound for track memory.
4. **E4 — A map the car builds itself.** On an out-lap the car records the grip it feels and localizes itself from telemetry and its sensors, with a lap-line signal.
5. **Gate.** (E3–E4 may take 2–3 experiments.)

Dropped after E2 (owner, 2026-11-11): a speed reward and estimator-in-the-loop training. E2 showed the car is already at the ceiling its information allows.

**Track memory decisions (owner, 2026-11-11):**
- **Gate protocol:** an out-lap, then a timed lap. Both must be finished to count toward the 85%; only the timed lap's time counts.
- **Position:** the car is told when it crosses the start line (a lap counter). Otherwise it must localize itself from its own telemetry (speed, gyro, accelerometer) and sensors (the rays). Fallback if that proves too hard: tracks are laid out to a plan, so the car can be given a technical drawing (the centerline). The owner prefers not to use this.
- **The map:** the grip the car felt on the out-lap (from the grip estimator), averaged over short stretches (about 5 m); while driving it sees the grip profile for the next ~40 m.

## Experiments

### E1 — Pick for speed, not just safety

- **Prediction (owner, before the run):** "Maybe 1 s faster but that's all."
- **Setup:** Phase 3's gate recipe retrained: `python -m sdc.phase3.my_ppo --seed S --steps 1000000 --env realism=3 --env sensors='["accel","grip"]' --env lap_bonus=100 --set std_cap='[0.5,0.1]' --set select=true --set gamma=0.997`, seeds 0–2, 1 torch thread, 3 runs in parallel. Every checkpoint (every 50k steps) is scored on validation gate range (100 laps, true grip, as Phase 3's early stopping did) and nominal lap time. Two picks per seed: **most reliable** (Phase 3's rule) and **fastest with ≥ 90%**. Both get a 1 s window grip estimator and are re-scored driving on the estimate.
  - Commands: `python -m sdc.phase35.e1 score|pick --run runs/p35-e1-sS`, then `python -m sdc.phase3.grip_estimator --policy runs/p35-e1-sS-{reliable,fast} --kind window --seed S`. Training 11.5 min per seed; whole pipeline about 45 min.
- **Result (validation set, driving on the grip estimate):** [report](https://claude.ai/artifact/PpaDqfZmue4PeaWXNPAiXy)

  | Seed | Most reliable: checkpoint, gate range, lap | Fastest with ≥ 90%: checkpoint, gate range, lap | Fastest checkpoint of the run (any reliability) |
  |---|---|---|---|
  | 0 | 250k, 100%, 41.1 s | 650k, 93%, **36.7 s** | 32.8 s (150k, 54%) |
  | 1 | 800k, 97%, 34.5 s | same | 31.1 s (100k, 25%) |
  | 2 | 800k, 95%, 34.6 s | same | 32.1 s (350k, 66%) |

  - Hand-coded driver: 25.4 s; gate bar 1.2× = 30.5 s.
  - Seed 0 reproduces Phase 3's gate car (picked at 250k again, 41 s); the new rule saves 4.4 s there. Seeds 1–2 already picked late checkpoints, so nothing changes.
  - Every checkpoint of every run laps in 31–43 s. The late, less reliable checkpoints are only about 1–2 s faster (32–34 s, 75–89% gate range), not the 28.5 s the Phase 3 handoff quoted (that was the E12 runs without the lap bonus).
- **Explanation (owner, after the run):** "It isn't learning good racing technique, it's just getting by. That has me wonder if we're missing something.. more/better sensors, bigger model, longer training, etc. But also has me wondering if that's just the max with the constraints we've imposed (noisy sensors etc)."
- **Takeaway:** checkpoint choice is worth up to about 4 s, but every checkpoint is at 31 s or slower, so the gap is in what the car learns, not in which snapshot we keep. Next: measure the ceiling under the car's own constraints (E2).

### E2 — The ceiling under the car's own constraints

- **Prediction (owner, before the run):** "30.5" (seconds, for the hand-coded driver given only what the RL car knows).
- **Setup:** the hand-coded driver (`dy.DynamicPursuit`) with its privileges removed step by step: (a) full knowledge; (b) no dirt map, nominal mass assumed, true grip; (c) as (b) but the grip from a 1 s window estimator. Safety margins swept (cornering 0.4–0.8, braking 0.3–0.5, 15 settings); the answer is the fastest setting with ≥ 90% validation gate range. `python -m sdc.phase35.e2 --variant full|blind|blind_est [--est DIR]`; `--fit-est runs/p35-e2-est` trains an estimator on the hand-coded driver's own driving (4 DAgger rounds, 240k samples). No RL training; about 15 min.
- **Result (validation set):** [report](https://claude.ai/artifact/6CMfCFdVTt42HGPhu8g7gx)

  | Driver | Fastest with ≥ 90% gate range |
  |---|---|
  | (a) hand-coded, knows grip, mass, dirt | 25.4 s (97%) |
  | (b) no dirt map, no mass, true grip | 29.9 s (91%) |
  | (c) no dirt map, no mass, estimated grip | **34.8 s** (91%) |
  | RL car (E1, seeds 1–2) | 34.5 s (95–97%) |

  - With the RL car's information, the hand-coded driver is no faster than the RL car. Not knowing the dirt costs about 4.5 s; estimating the grip instead of knowing it costs another 4.9 s.
  - Surprise, checked: the first try reused the RL car's estimator; on the hand-coded driver's driving it was off by 0.21 (15% low) against 0.05 on the RL car's (best: 86%, 38.4 s). An estimator trained on the hand-coded driver's own driving is still off by about 0.17: the hand-coded driver never pushes the tires near their limit, so its driving reveals little about the grip. The RL car slides more.
  - Caveat: the hand-coded driver still knows the track shape ahead and reads speed and position without noise, so (c) is a slightly generous ceiling.
- **Explanation (owner, after the run):** "Agreed on learning the track." (after the agent's summary: the RL car is at the ceiling its information allows, so the gap is knowledge, not technique.)
- **Takeaway:** with only its own sensors, even a tuned expert laps in about 35 s. The 30.5 s bar needs knowledge of the track's grip (dirt alone is worth about 4.5 s), so the phase turns to track memory.

### E3 — What a perfect track map is worth

- **Prediction (owner, before the run):** "28" (seconds).
- **Setup:** E1's recipe plus the oracle `gripmap` sensor (true grip / nominal at 5, 10, …, 40 m ahead on the centerline, from the exact position): `--env sensors='["accel","grip","gripmap"]'`. Seeds 0–2, 1M steps, 3 in parallel. Scored with true grip and true map (an upper bound, no estimator); pick = fastest checkpoint with ≥ 90% validation gate range (`sdc.phase35.e1 score|pick`). Report: `python -m sdc.phase35.e3_report`.
- **Result (validation set):** [report](https://claude.ai/artifact/PCNQgGd1MjTFUUUnthXRRg)

  | Seed | Fastest with ≥ 90% | True map | Map says no dirt | Map says all dirt |
  |---|---|---|---|---|
  | 0 | 550k, 92% | 90%, 32.8 s | 70%, 31.8 s | 100%, 43.5 s |
  | 1 | 950k, 97% | 95%, 33.7 s | 55%, 31.5 s | 100%, 41.7 s |
  | 2 | 600k, 95% | 95%, 34.2 s | 85%, 33.9 s | 95%, 35.5 s |

  - A perfect map is worth about 1 s to the RL car (34.1 s without it, E1 with true grip), against 4.5 s to the hand-coded driver.
  - It does read the map, but only to slow down for dirt: lied to that the track is clean, it crashes more and is barely faster (31.5–34 s).
  - Checked: lookahead isn't the limit. The hand-coded driver needs only 20 m of planning to lap in 25.4 s (95%); the rays reach 40 m.
  - This revises E2: with the grip estimate the RL car matches the hand-coded driver, but given better information only the hand-coded driver gets faster (29.9 s with true grip, 25.4 s with the map). The RL car's pace on clean track is the limit.
- **Explanation (owner, after the run):** "Maybe it hasn't explored that space enough? Like it just never tried going faster so it doesn't know that would be ok?"
- **Takeaway:** better information doesn't make the RL car faster; its clean-track pace is the limit. Owner's hypothesis to test next: it never explored fast driving.

### E4 — Show it fast driving, then PPO (did it never explore going fast?)

- **Prediction (owner, before the run):** "I think it'll drift back (even though this was my idea)."
- **Setup:**
  - **Behaviour cloning** (`python -m sdc.phase35.bc --seed S`): the owner's network copies the full-knowledge hand-coded driver (25.4 s) with E3's inputs. DAgger, 5 rounds × 50k states on training tracks and training-range physics; squared error on tanh(actor mean). The critic is then regressed on discounted returns (γ 0.997, lap bonus 100) of the clone's own noisy driving.
  - **Then PPO** from the clone (`my_ppo --set init=runs/p35-e4-bc-sS/model.pt`, new plumbing option), E3's recipe otherwise. Two noise schedules: `std_cap=[0.5, 0.1]` (as E3) and `[0.15, 0.05]` (low noise). Seeds 0–1, 4 runs in parallel. Same scoring and pick as E1/E3. Report: `python -m sdc.phase35.e4_report`.
- **Result (validation set, true grip and true map):** [report](https://claude.ai/artifact/9k5s4t67unL6qe2WYsrDEc)

  | Run | Clone only: gate range, lap | Pick (fastest ≥ 90%) | Last checkpoint (1M) |
  |---|---|---|---|
  | high noise, seed 0 | 62%, 25.0 s | 92%, **30.3 s** | 85%, 30.9 s |
  | high noise, seed 1 | 33%, 24.9 s | none ≥ 90% (best 89%, 28.0 s) | 77%, 28.2 s |
  | low noise, seed 0 | 62%, 25.0 s | 95%, **25.8 s** (51k) | 89%, 28.4 s |
  | low noise, seed 1 | 33%, 24.9 s | none ≥ 90% (best 72%, 25.6 s) | 40%, 23.9 s |

  - From scratch (E3) the same inputs and reward gave 33–34 s. Starting from fast driving, PPO keeps most of the speed: 28–31 s with high noise, 24–28 s with low noise.
  - The clone is fast but fragile (33–62% gate range): copying never shows it how to recover. PPO buys robustness with some speed; with high exploration noise it gives up more.
  - Seed spread is large: low-noise seed 1 stayed fast but never became reliable.
- **Explanation (owner, after the run):** "With highly varied exploration, it learns fear/caution because it knows how bad things can get. Less noise means it never hits the worst scenarios so it doesn't have to prepare for them."
- **Takeaway:** the car never found fast driving on its own; started there, PPO keeps most of it, and less exploration noise keeps more. Next: the same car on its own map and grip estimate.

### E5 — The car builds its own map

- **Prediction (owner, before the run):** "Seems like it wasn't very reliable even with perfect knowledge, so I expect this to less reliable. But similarly fast because it doesn't know to be less confident. But that balance comes from how conservative it is with grip estimation and location."
- **Setup:** `sdc/phase35/selfmap.py`. No oracle inputs: grip from a 1 s window estimator trained on this car's own driving (4 DAgger rounds, 240k samples); position from an odometer (noisy speed reading × dt), reset at each start-line crossing. On the out-lap the car records its grip estimate in 5 m odometer bins, while its map input says "everything ahead may be dirt" (running mean estimate × 0.65). On the timed lap the map input is its recording at odometer + 5…40 m. Gate protocol: out-lap + timed lap, both must finish, timed lap's time counts. Candidates: E4's seed 0 picks, low noise (25.8 s with oracle inputs) and high noise (30.3 s). Validation set; the same cars with oracle inputs on the same protocol for comparison. Report: `python -m sdc.phase35.e5_report`.
- **Result (validation set, out-lap + timed lap):** [report](https://claude.ai/artifact/DVJTYQG91DrfjfCDKLmNVv)

  | Car | Inputs | Gate range | Nominal | Timed lap |
  |---|---|---|---|---|
  | low noise | true grip and map | 86% | 100% | 24.2 s (1.01×) |
  | low noise | **own estimate and map** | **55%** | 100% | **26.7 s (1.11×)** |
  | high noise | true grip and map | 88% | 90% | 28.8 s (1.20×) |
  | high noise | own estimate and map | 77% | 95% | 30.7 s (1.28×) |

  - **The bar moves with the protocol:** after an out-lap the timed lap has a flying start. The hand-coded driver's flying lap is 23.9 s, so 1.2× is 28.7 s (not 30.5 s). The gate defined the bar as the hand-coded driver on the same tracks; the agent applies it on the same protocol too.
  - Failures by grip (low-noise car, own map): below 0.7×: 1/27 finished (24/27 with the true grip and map); 0.7–0.9×: 10/26; 0.9–1.1×: 25/28; above 1.1×: 19/19.
  - Checked: the estimator's readings pull toward the middle of its training range (0.7–1.1×). Low-noise car: +0.16 at 0.55× grip (overconfident), −0.03 at 0.8×, −0.20 at 1.0×, −0.38 at 1.2× (overcautious). Map error on the timed lap: 0.24 (high-noise car: 0.12). Odometer vs. centerline distance: about 3.5 m, under one 5 m bin.
- **Explanation (owner, after the run):** "Grip is hard to determine in a short spanse, so it ends up averaging over everything it sees."
- **Takeaway:** self-mapping and odometry work; the speed bar is met (26.7 s vs. 28.7 s), but the grip estimate shrinks toward its training range, which costs reliability on slippery tracks (55%).

**Decisions after E5 (owner, 2026-11-11):**
- Fix grip estimation with a **brake test on the out-lap** (E6), then the gate. Not chosen: wider training ranges (would make the gate interpolation), stopping here.
- **Speed bar on the gate protocol:** with an out-lap the timed lap has a flying start, so the bar is 1.2× the hand-coded driver's flying lap on the same tracks (23.9 s on validation, so 28.7 s), not 30.5 s.

### E6 — A brake test on the out-lap

- **Prediction (owner, before the run):** "Small improvements but it has to do multiple brake tests so it see the full range of surfaces."
- **Setup:** `sdc/phase35/braketest.py`. At the start of the out-lap the car keeps its own steering while the throttle is overridden: accelerate to 8 m/s, then full brake for up to 1 s. Locked-wheel deceleration (accelerometer, readings after 0.3 s and above 1.5 m/s) × K gives the grip level. K = 1.61 from a least-squares fit on training-range physics only (the tire model predicts 1/(0.75 × 0.85) = 1.57). Check on gate-range physics (not used for fitting): mean error 0.03 at grip 0.5–0.7×, 0.04 at 0.7–0.9×, 0.08 at 0.9–1.1×, 0.14 at 1.1–1.3×. After the test, grip input = test level × (window estimate / median estimate so far on the lap, clipped to 0.6–1); the map records the same (as E5). Known weakness: a start on dirt reads dirt grip (validation track 10000 does). Car: E5's low-noise car and estimator. Validation set, out-lap + timed lap. `python -m sdc.phase35.braketest calibrate|eval`; report `python -m sdc.phase35.e6_report`.
- **Result (validation set):** [report](https://claude.ai/artifact/9yULEP789F1wRukvj2ARPd). Gate range **39%** (E5: 55%), nominal 60% (E5: 100%), timed lap 24.7 s (E5: 26.7 s). By grip: 0.7–0.9×: 9/26, 0.9–1.1×: 13/28, <0.7×: 9/27, >1.1×: 8/19.
  - Failures are on the timed lap (61 of 61 at gate range past 5% of the attempt, mostly 60–90%).
  - Checked: the map is blind to dirt. On the timed lap its map rates true dirt at 0.945 of the brake-test level and clean track at 0.949; it flags 2% of dirt (1% false flags). The cautious out-lap barely loads the tires, so the window estimator can't tell dirt from tarmac. E5's overall underestimate (−0.2 at 1.0×) had hidden this by making the car cautious everywhere.
  - The brake test itself reads 1.05 at nominal grip (0.69 where the start is on dirt).
- **Explanation (owner, after the run):** the car needs to test the surface more than once to see every surface (as predicted). "[Probing repeatedly is] what I thought you were doing for the reason I gave."
- **Takeaway:** the brake test reads the grip level well, but one test reads one surface. The gentle out-lap leaves the map blind to dirt.
- **Decision (owner, 2026-11-11):** one more experiment (E7): probe the surface repeatedly on the out-lap, then the gate.

### E7 — Probe the surface repeatedly on the out-lap

- **Prediction (owner, before the run):** "Small improvement to both."
- **Setup:** `sdc/phase35/probe.py`. On the out-lap the car keeps its own steering; every 6 m of odometer, when its steering is nearly straight (|command| < 0.35), it gives a 0.4 s full-brake pulse and reads the locked-wheel deceleration (readings from the third step, above 1.5 m/s) × K. Below 6 m/s when a probe is due, the throttle goes to full first (only while nearly straight). Readings fill the 5 m odometer map (median per bin, gaps interpolated). While mapping: grip input = latest reading, map input = median reading × 0.65. Timed lap: grip input and map input both from the map; no window estimator. K = 1.53 from training-range physics (rmse 0.046, about 30 probes per out-lap); gate-range check (not fitted): mean error 0.034, bias −0.018. Car: E5's low-noise car. Validation set, out-lap + timed lap. `python -m sdc.phase35.probe calibrate|eval`; report `python -m sdc.phase35.e7_report`.
- **Result (validation set):** [report](https://claude.ai/artifact/ULDERFVfy38Q8ZnuDYT4pb)

  | Grip knowledge | Gate range | Nominal | Timed lap |
  |---|---|---|---|
  | true grip and map | 86% | 100% | 24.2 s |
  | E5: window estimate | 55% | 100% | 26.7 s |
  | E6: one brake test | 39% | 60% | 24.7 s |
  | **E7: probes every 6 m** | **78%** | **100%** | **24.5 s** |

  - By grip: below 0.7×: 18/27 (E5: 1/27); 0.7–0.9×: 19/26; 0.9–1.1×: 26/28; above 1.1×: 15/19.
  - Map error on the timed lap 0.05 (E5: 0.24); about 30 probes per out-lap. Of 22 failures, 3 on the out-lap and 19 on the timed lap: unprobed dirt in corners and the position shift between the out-lap and timed-lap lines.
  - Speed bar met (24.5 s vs. 28.7 s); reliability 7 points short on validation.
- **Explanation (owner, after the run):** _pending_

## Gate check

## Handoff to next phase
