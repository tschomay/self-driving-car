# Phase 4.7 — Beyond the teacher: drills and residual reinforcement learning

- **Status:** in progress
- **Goal:** the car learns to handle itself from outcomes, not only by copying, and ends up better than the teacher it was copied from. Phase 4.6's copied car is kept frozen; reinforcement learning (the owner's PPO) learns a small correction on top of it, first in drills (short episodes started in hard states), then on full laps.
- **Gate (set at entry, 2026-11-19):** on the test loops (46000–46019 × 3 starts, gate-range physics, truggy physics, timed lap only), the car beats the re-tuned teacher on the same 60 runs (teacher: 52 / 60 at the 4.6 gate retry) in **either** way (owner: "either option counts as success"):
  - **(a) more reliable:** more laps finished than the teacher, with a median lap no slower than the teacher's; or
  - **(b) faster:** at least as many laps finished as the teacher, with a faster median lap.
- **Code:** `sdc/phase47/` (to come); base car `sdc/phase46/bc_loop.py` + `ensemble.py`; rebuild in a fresh container with `bash sdc/phase46/rebuild.sh`.
- **How to reproduce the gate:** to come.

## Entry decisions (2026-11-19)

1. **Phase:** a new phase 4.7, not a 4.6 stretch (4.6's gate passed; this is a new method). Owner agreed.
2. **Success:** either more finishes than the teacher, or the same finishes with faster laps (gate above). Owner's choice: either counts.
3. **First drills:** too-fast corner entries and bad takeoffs, the base car's main ways of failing (off the lane, rolled over). Mid-slide drills later. Owner agreed.
4. **Compute:** the container (4 CPUs). Modal can't be reached from cloud sessions: its client uses gRPC, which the sandbox proxy doesn't carry (tested 2026-11-19 with the token and an unrestricted network). A 1M-step run is about 2 h on one core; 4 at once.

## Why reinforcement learning failed before, and what changes

- Phases 4 and 4.5 fine-tuned the **whole copied network** with PPO. It drifted: it traded landings for speed (Phase 4), or lost 10–15 points within 50k steps (Phase 4.5); a finish bonus slowed the slide but didn't stop it. Checkpoints were picked on noisy 20-course scores.
- Changes: (1) the copy is **frozen**; PPO learns only an additive **residual**, which starts at zero and is capped, so the car can't forget how to drive; (2) **drills** start the car where mistakes happen, so a run sees hundreds of near-crashes instead of about one crash per 7 laps; (3) a crash penalty and finish bonus on full laps; (4) checkpoints picked on 60 validation runs.

## Plan

0. **Build (agent):** rebuild the base car (E10 arm B, the averaged expo pair: 52 / 60 on validation, teacher 55). Drill generator: the base car drives practice loops (training-range physics) while snapshots are taken (physics, sensors, filter, and both students' memories; `sdc/phase46/hard.py`). Drill starts: 1 s before a corner with speed raised 10–40%; 1 s before a kicker with speed changed ±25% and the nose pitched up or down; 1–3 s before the base car's own crashes. Residual wrapper around the env, so the owner's PPO core (`sdc/phase1/my_ppo.py`) is used unchanged.
1. **E1 — Drills alone.** Residual trained only on drills (3 s each: metres along the lane, −50 for a crash), scored on full validation laps.
2. **E2 — Drills, then full laps.** E1's residual continued on full practice laps mixed with drills: progress, finish bonus, crash penalty.
3. **E3 — Speed** (if E1–E2 make it more reliable but not faster): a lap-time reward, or drills at higher target speeds.
4. **E4 — Mid-slide drills** (if useful): starts in a held slide.

Every experiment gets the owner's prediction before it runs.

## Experiments

### E1 — Drills alone

- **Prediction (owner, before the run):** "54 and a little slower" (validation laps finished of 60; base car 52, teacher 55).
- **Setup:** frozen base car (E10 arm B pair, averaged); residual = the owner's Phase 1 network (two 64-unit layers) on sensors + the base car's action, output capped ±0.3, starting at zero; owner's PPO core. Drills only (3 s; +1 per metre along the lane, −50 for a crash, small cost on the residual): too-fast corner entries, bad takeoffs, and starts 1–3 s before the base car's own crashes, on practice loops at training-range physics. 1M steps, seeds 0–1; checkpoint kept by 60 validation runs (gate range) every 100k steps.
- **Run:** 2026-11-19, `bash sdc/phase47/e1_runs.sh` (after `sdc/phase46/rebuild.sh`; base car rebuilt at 52 / 60 on validation, as before), about 2 h per seed on one core. Harvest per seed: 80 practice laps, 75 finished; drill starts: about 1,050 corners, 310 takeoffs, 11–14 crashes. Then `python -m sdc.phase47.check` (fresh loops 47000–47033 × 3 starts, 102 runs, never used before) and `python -m sdc.phase47.e1_report`.
- **Result:** [report](https://claude.ai/artifact/Ef5njcHVMDfGpWyw3zjxNU).

  | Car | Validation (60) | Fresh loops (102) | Median lap (fresh) |
  |---|---|---|---|
  | Teacher | 55 | 95 | 31.2 s |
  | Base car (step 0) | 52 | 84 | 31.5 s |
  | Seed 0, kept (400k) | 54 | 85 | 30.8 s |
  | Seed 1, kept (800k) | 54 | 78 | 31.1 s |
  | Seed 0, end (1M) | 53 | 82 | 30.6 s |
  | Seed 1, end (1M) | 46 | 81 | 31.1 s |

  - Validation jumped 46–54 between checkpoints with no trend; the kept checkpoint is the best of six, so its 54 is mostly selection luck. The fresh loops show no reliable gain in finishes.
  - What holds on both sets: laps 0.4–0.9 s faster, more jumps landed (fresh: 85% → 86–89%), more runs off the lane.
  - Drills taught little: corner drills crashed 2–4% even at the start (the base car already copes with 10–40% too fast); crash drills came from only 11–14 real crashes. Drill crash rates fell slightly (takeoff 11–15% → 9–11%), partly because exploration noise shrinks.
  - Prediction vs. result: 54 and a little slower predicted; 54 on validation (not confirmed on fresh loops), a little faster.
- **Explanation (owner, after the run):** "It seems like the drills weren't hard enough, but the extra training and variation helped it find a faster line or just faster technique." Wants to see more learning.
- **Takeaway:** a frozen base plus a capped residual is safe (no collapse, unlike Phases 4–4.5), but drills this easy give little to learn; 60 validation runs can't tell 52 from 54.

### E2 — Harder drills, and a drill exam

- **Why (owner, after E1):** the drills weren't hard enough and it didn't learn much; wants to see more learning. Full laps (the old E2) move to E3.
- **Setup:** E1's residual recipe (frozen base, residual capped ±0.3 from zero, owner's PPO core), with:
  - **Difficulty calibrated per drill kind** so the base car alone crashes about 40% of the time (binary search on a strength setting over 100 drills).
  - **New kinds:** *edge* (a sideways push toward the nearer bank) and *slide* (a yaw-rate burst), alongside *corner* (too fast) and *takeoff* (wrong speed and a pitch kick off the lip).
  - **More crash starts:** harvest with noise on the base car's actions, so it crashes far more often (1–3 s before each crash).
  - **Drill exam:** 300 drills from loops never used for training, fixed perturbations, driven with no noise every 100k steps: the crash rate shows learning directly. Base car about 40% by construction.
  - **Checkpoint picks** on 150 practice-loop runs (50 loops × 3 starts, gate range) instead of 60 validation runs; the fresh loops (47000–47033, 102 runs) stay the final check.
  - 1M steps, seeds 0–1.
- **Prediction (owner, before the run):** "25%. 90" (drill-exam crash rate at the end; fresh runs finished of 102: base 84, teacher 95).
- **Prep** (`python -m sdc.phase47.e2_prep`; practice loops split 181 train / 50 selection / 30 calibration / 30 exam): teacher on the 150 selection runs 141 finished. Calibration harvest (noisy base, 30 loops): 13 crashes, 36 crash starts. Strengths at about 40% base crashes: corner 0.64 (speed up to ×2), takeoff 0.58, edge 0.84 (push up to 5.9 m/s), slide 0.50 (yaw burst up to 8 rad/s).
- **Base car on the exam: 31% crashed, not 40%** (corner 33%, takeoff 53%, edge 38%, slide 25%, crash drills 7%). The crash drills aren't calibrated: their starts came from the noisy car's crashes, and the base car without noise recovers from most of them. The owner's 25% prediction was made against an expected 40%.

- **Result** (report: `reports/phase47-e2.html`; `bash sdc/phase47/e2_runs.sh`, then `python -m sdc.phase47.e2_report`):

  | Car | Exam crashed (300) | Selection (150) | Fresh loops (102) | Median lap (fresh) |
  |---|---|---|---|---|
  | Teacher | — | 141 | 95 | 31.2 s |
  | Base car (step 0) | 31% | 124 | 84 | 31.5 s |
  | Seed 0, kept (600k) | 31% | 133 | 83 | 31.0 s |
  | Seed 1, kept (400k) | 29% | 135 | 86 | 32.0 s |
  | Seed 0, end (1M) | 31% | 129 | 81 | 31.3 s |
  | Seed 1, end (1M) | 30% | 128 | 81 | 32.0 s |

  - **The exam stayed flat:** 29–34% at every checkpoint on both seeds, against 31% for the base car. One standard error on 300 drills is about 2.7 points, and on 60 drills of one kind about 6 points. No drill kind trended down on both seeds.
  - **Selection gains didn't carry over:** the kept checkpoints scored 133 and 135 of 150 against the base car's 124, but 83 and 86 of 102 on the fresh loops against its 84. As in E1, best-of-six picking flatters the kept checkpoint.
  - **E1's faster laps didn't come back:** seed 0 was 0.4 s faster on the fresh loops, and seed 1 was 0.5 s slower.
  - **Prediction vs. result:** predicted 25% and 90; got 29–31% and 83–86. Both are within noise of the base car.
- **Explanation (owner):** "That was my gut reaction too - is it possible to recover? We seem to be in a situation where we're having a number of busts again." (reacting to the flat exam: the drills taught little, which led to the recoverability check below)

### After E2 — Can these crashes be recovered at all? (agent, owner's request)

- **Question (owner):** is it possible to recover from these drills? If the check isn't a slam dunk, research before the next step.
- **Recoverability** (`python -m sdc.phase47.recoverable`, E2's 300 exam drills): the teacher, reading the true state, crashed 29% (base car 31%). The base car plus one fixed full-strength correction (9 options: steer and throttle each −0.3/0/+0.3, held 1 s): every option crashed 31–41% overall, but some option avoided the crash on 62 of the base car's 94 crashed drills, each option saving different drills and crashing others.
- **Chaos on drills** (`python -m sdc.phase47.chaos`): the base car drove each exam drill 5 times with tiny action noise (std 0.02, about 1% of full range):

  | Outcome over 5 runs | Drills |
  |---|---|
  | Never crashed | 178 (59%) |
  | Sometimes crashed | 65 (22%) |
  | Always crashed | 57 (19%) |

  39% of the base car's exam crashes disappear with a 1% wiggle. Of the 57 always-crash drills, 24 were saved by neither the teacher nor any fixed correction. Takeoff drills are the most chaotic (38% mixed).
- **Chaos on full laps** (`python -m sdc.phase47.lap_chaos`, 150 selection runs × 3 noisy repeats):

  | Driver | Finished per repeat | Always finished | Sometimes | Never |
  |---|---|---|---|---|
  | Base car | 130, 126, 128 | 102 | 44 | 4 |
  | Teacher | 137, 143, 138 | 122 | 27 | 1 |

- **What this means:**
  - The base car almost never fails a run every time: 4 of its ~22 failures per pass are certain; the rest are runs it finishes on other tries. The teacher's edge is fewer coin-flip runs (27 vs 44), not runs only it can handle.
  - One pass of 150 runs swings about ±3 laps from chance alone, about the size of every E1 and E2 "effect" (E1 validation 46–54; E2 fresh 81–86 vs base 84). Single-pass scores can't separate cars that close.
  - A residual trained on single crash/no-crash outcomes gets mostly chance as its signal. The literature names this the "curse of chaos": in systems sensitive to small differences, policy-gradient estimates become very noisy (Parmas et al., PIPPS, arXiv 1902.01240).
- **Takeaway:** the car doesn't need new recoveries; it needs a lower crash probability at many risky moments, mostly jumps. To see that, scores must be probabilities (repeated noisy runs), not single passes.

### After E2 — Where the coin flips happen (agent, owner: "go for it")

- **Run:** `python -m sdc.phase47.where collect` then `analyse`. The base car and the teacher each drove the 150 selection runs 5 times with tiny action noise (std 0.02), with traces, takeoffs and landings recorded (about 35 min). Each run keeps its physics on every repeat. Base finished 127, 133, 135, 126, 131; teacher 139, 136, 141, 127, 142.
- **Where they fail** (all 5 passes; "after landing" = off the lane 1–3 s after touching down, a fish-tail):

  | Place | Base car | Teacher |
  |---|---|---|
  | Corner | 35 | 2 |
  | Landing (within 1 s) | 16 | 26 |
  | After landing (1–3 s) | 34 | 35 |
  | Elsewhere | 13 | 2 |
  | **All** | **98** | **65** |

- **Corners are the whole gap.** 35 corner failures in 5 passes is 7 a pass, about the teacher's lead (137 vs 130 a pass).
  - The base car enters sharp turns (> 0.15 /m) 0.5 m/s faster than the teacher at the same corner, at every grip level.
  - The teacher slows more for sharper turns (4.19 → 3.95 m/s); the base car doesn't (4.53 → 4.66 m/s).
  - It only crashes when grip is low: base car failures by grip quarter 24%, 11%, 7%, 10% (teacher 5%, 10%, 9%, 11%).
  - Why: the base car (E10 arm B) has sensors only and copies a teacher that plans its braking from the true track. It can't see a sharp turn coming early enough to brake as the teacher does: Phase 4.5's lesson again.
- **Landings are where the teacher itself fails:** 61 of its 65 failures, mostly rollovers on touchdown and spins 1–3 s after. The base car fails there about as often (50). Its pitch at touchdown is further off the landing slope (doubles: −30° on landings it survived, −41° on failures; teacher −12°), but both cars mostly fail on doubles taken 1.6–2 m/s under design speed.
- **Takeaway:** two separate problems. Catching up to the teacher is a corner-speed problem (knowledge the student lacks), not a recovery problem. Beating the teacher needs better landings and post-landing stability, the one place the teacher fails.

### E3 — A map of the turns, so the student can brake where the teacher does

- **Why:** the base car's whole gap to the teacher is corners: it can't see a sharp turn coming and carries 0.5 m/s too much into it (see above).
- **Setup:**
  - E10 arm B's recipe (DAgger, 16 rounds, truggy physics, expo 0.4, seeds 0–1, averaged) plus the out-lap map of the turns as input (Phase 4.6 E3: 9 mapped curvatures ahead of the car's own odometer).
  - The teacher plans its corner speeds from the same map (`LoopExpert(use_turn_map=True)`), so it only knows what the student can know.
  - The maps are rebuilt on the truggy physics (`python -m sdc.phase47.e3_maps`; Phase 4.6's were on the old physics).
  - Scoring: 5 noisy passes (std 0.02) of the 150 selection runs, as above (`python -m sdc.phase47.where e3`). The map-following teacher is scored the same way, as the new ceiling.
  - `bash sdc/phase47/e3_runs.sh`.
- **Prediction (owner, before the run):** "133 and no time change" (selection runs finished per pass: base car about 130, teacher about 137).

- **Result** (`runs/p47-e3-where.json`; scores pooled with the earlier base/teacher passes):

  | Driver | Finished per pass (of 150) | Mean | Corner failures (5 passes) | Corner entry speed | Median lap |
  |---|---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | 35 | 4.59 m/s | 30.7 s |
  | Teacher (true track) | 139, 136, 141, 127, 142 | 137 | 2 | 4.12 m/s | 30.8 s |
  | **Map car (E3)** | 113, 108, 106, 99, 102 | **106** | 123 | 4.17 m/s | 30.5 s |
  | Teacher reading the map | 126, 126, 129, 128, 125 | 127 | 71 | 4.18 m/s | 30.6 s |

  - **The map car is much worse: 106 a pass, 24 below the base car.** Its corner failures went up from 35 to 123, even though it now slows for corners like the teacher (4.17 m/s at sharp turns, against the base car's 4.59).
  - **The map itself is the problem, not the student:** the teacher planning from the same map drops from 137 to 127 a pass, and its corner failures go from 2 to 71.
  - **10 of the 150 runs have no map** (their out-lap didn't finish). On those the map car finished 8% of the time and the map teacher 28%: with an empty map the car sees a straight road.
  - **On runs with a map:** base car 87%, teacher 91%, map teacher 89%, map car 75%.
  - **Odometer drift** (`python -m sdc.phase47.map_align`, map teacher, 12 selection loops, one run each): the map is read at the car's own odometer, which re-syncs only at kickers.
    - On the 8 laps it finished, the mapped curvature lined up with the true one (correlation 0.74–0.97, best shift within 1 m).
    - On the 3 laps it left the lane, the odometer was 8–15 m off by the end (one had no map at all).
    - _Corrected after the quick test below:_ this drift is not clearly the cause. Some of it builds up after the car has already left the lane (wheels spinning), and some laps finish with 13–16 m of drift.
  - Lap times differ by under 0.3 s. Medians are over the laps each car finished, so they aren't directly comparable.
  - **Prediction vs. result:** predicted 133 per pass and no time change. Got 106 (99–113) and about the same lap time.
- **Explanation (owner):** "A map only hurts if it's inaccurate."

### After E3 — Complete and accurate maps: the quick test (owner: "A map only hurts if it's inaccurate. Go ahead with that quick test")

- **Re-sync at turns:** I tried correcting the odometer at turns as well as kickers, by sliding the last 12 m of the car's own curvature along the map. On the 20 loops checked (map teacher, no noise), the match passed the confidence threshold almost nowhere, so the fix never fired. Where the map is wrong, the car's own curvature over 12 m is too noisy to place it. Not kept.
- **Missing maps:** 10 of the 150 runs had no map because their slow mapping lap failed (7 rolled over, 2 left the lane, 1 got stuck).
  - Driving the mapping lap again, at a jump speed of 3.5, then 3.0, 4.5, then 5.0 m/s (default 4.0) until it finished, gave all 10 a map.
  - The map car and the map-reading teacher then drove the 8 affected loops again (5 noisy passes); their other runs keep their E3 results (`python -m sdc.phase47.full_maps`).

  | Driver | Finished per pass (of 150) | Mean | On the 10 re-mapped runs |
  |---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | — |
  | Teacher (true track) | 139, 136, 141, 127, 142 | 137 | — |
  | Teacher reading the map, E3 maps | 126, 126, 129, 128, 125 | 127 | 1–4 of 10 |
  | **Teacher reading the map, complete maps** | 133, 133, 134, 135, 132 | **133** | 8–10 of 10 |
  | Map car, E3 maps | 113, 108, 106, 99, 102 | 106 | — |
  | **Map car, complete maps** (not retrained) | 117, 116, 112, 105, 111 | **112** | 6–9 of 10 |

- **What it shows:**
  - With complete maps, the teacher planning only from what the car can know gets within 4 a pass of the teacher that sees the true track. The map is good enough.
  - The map car is still 18 a pass below the base car with complete maps. Its gap is in the copying, not the map. The teacher reading the map loses 4 a pass on mapped runs; the student loses 17.


### E3b — Map students retrained on complete maps (owner: "Try both in parallel now")

- **Setup:** E3's recipe (`bc_loop --truggy --rounds 16 --expo 0.4 --map`, 2 seeds averaged), with every failed mapping lap in the practice and validation loops driven again as in the quick test (5 validation, 23 practice). Scored on the 150 selection runs with complete maps (`python -m sdc.phase47.where e3b`).
  - Not directly comparable to E3's training: up to 22 practice loops that had no map now take part, and they are among the harder loops. Validation scores during training also differ from E3's for the same reason.
- **Prediction:** none given (owner asked to run both overnight).
- **Result:**

  | Driver | Finished per pass (of 150) | Mean | Corner failures (5 passes) | Doubles landed | Median lap |
  |---|---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | 35 | 91.6% | 30.7 s |
  | Teacher | 139, 136, 141, 127, 142 | 137 | 2 | 94.6% | 30.8 s |
  | Map car E3, complete maps (not retrained) | 117, 116, 112, 105, 111 | 112 | — | — | — |
  | **Map car E3b (retrained on complete maps)** | 120, 117, 116, 119, 115 | **117** | 80 | 83.1% | 30.6 s |

  - Retraining on complete maps gained 5 a pass over E3 with complete maps (112 → 117), but the map car is still 13 below the base car.
  - Corner failures fell from 123 (E3) to 80, still more than twice the base car's 35. Doubles also got worse (83% landed against the base car's 92%).
  - On runs both finished, lap times match the base car's (median difference 0.0 s).
- **Explanation (owner):** "I don't know why 3b and 4 aren't helping."

### E4 — A teacher re-tuned for landings, copied by the base recipe (owner: "Try both in parallel now")

- **Step 1, tune the teacher** (`python -m sdc.phase47.e4_tune`): 7 settings, each driven 4 times over 150 practice runs (50 practice loops × 3 starts, never used for scoring). A setting replaces the current teacher only if it beats it by more than 3 a pass.

  | Setting | Finished per pass (of 150) | Mean | Median lap |
  |---|---|---|---|
  | Current (doubles at the general 6.8 m/s target) | 129, 130, 135, 136 | 132.5 | 30.8 s |
  | **Doubles at 7.4 m/s** | 138, 136, 134, 140 | **137.0** | 30.8 s |
  | Doubles at 8.0 m/s | 128, 135, 125, 129 | 129.3 | 30.7 s |
  | Doubles at 8.6 m/s | 115, 121, 123, 119 | 119.5 | 30.7 s |
  | Faster steering response (k_rate 0.5) | 133, 136, 133, 136 | 134.5 | 31.4 s |
  | Doubles at 8.0 + k_rate 0.5 | 123, 127, 121, 126 | 124.3 | 31.1 s |
  | Each jump at its design speed | 136, 133, 134, 133 | 134.0 | 31.0 s |

  - Chosen: doubles at 7.4 m/s, a little faster than before. Going faster still loses runs.
- **Step 2, check it on the selection runs** (`python -m sdc.phase47.where e4teacher`; training only starts if it beats the current teacher):

  | Driver | Finished per pass (of 150) | Mean | Doubles landed | Double take-off vs design speed | Median lap difference on shared runs |
  |---|---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | 91.6% | −1.31 m/s | — |
  | Teacher | 139, 136, 141, 127, 142 | 137 | 94.6% | −1.14 m/s | — |
  | **Re-tuned teacher** | 136, 144, 142, 142, 144 | **142** | 97.0% | −0.61 m/s | −0.10 s against the teacher, −0.08 s against the base car |

  - The re-tuned teacher finishes 5 more a pass than the old one and is no slower. Landing failures (rolled over or off the lane) fell from 26 to 20 over 5 passes, and "other" failures from 37 to 19.
- **Step 3, train the students:** base recipe with the re-tuned teacher (`bc_loop --truggy --rounds 16 --expo 0.4 --teacher-json runs/p47-e4-teacher.json`, 2 seeds averaged). Final validation: 53/60 and 49/60. Scored with `python -m sdc.phase47.where e4`.
- **Prediction:** none given (owner asked to run both overnight).
- **Result:**

  | Driver | Finished per pass (of 150) | Mean | Failures (5 passes): corner / landing / other | Doubles landed | Corner entry speed | Median lap difference on shared runs |
  |---|---|---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | 35 / 16 / 47 | 91.6% | 4.59 m/s | — |
  | Teacher | 139, 136, 141, 127, 142 | 137 | 2 / 26 / 37 | 94.6% | 4.12 m/s | — |
  | **E4 car** | 120, 127, 125, 125, 123 | **124** | 47 / 40 / 43 | 94.4% | 4.75 m/s | −0.50 s against the base car |

  - **Fewer finishes than the base car: 124 a pass against 130.** Not a success: the goal needs more finishes with no slower lap, or the same finishes with a faster lap.
  - **Faster:** on runs both finished, its laps are 0.5 s quicker than the base car's and the teacher's.
  - It lands doubles as often as the teacher (94.4%), but fails more at landings overall (40 against 16), mostly by rolling over (28 against 12), and enters corners faster than the base car (4.75 m/s against 4.59). Corner failures rose from 35 to 47.
- **Explanation (owner):** "I don't know why 3b and 4 aren't helping." The owner asked to step back: research what would really move the needle toward a good racer, rather than chase small gains (see "Stepping back" below).

### After E4 — A lower speed limit for the E4 students (owner: "Sure")

- **Question:** the E4 students are 0.5 s a lap faster than the base car but finish 6 fewer a pass. Does a lower speed limit trade that time back for finishes? If it got back to 130 with a faster median lap, it would meet the goal's second half.
- **Setup:** the E4 students with forward throttle scaled to 95% and to 90% (braking and steering unchanged), no retraining, scored on the 150 selection runs (`python -m sdc.phase47.where e4cap`).
- **Prediction (owner):** "I predict it won't help."
- **Result:** _running_
