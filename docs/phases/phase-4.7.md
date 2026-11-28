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
- **Result:**

  | Driver | Finished per pass (of 150) | Mean | Corner entry speed | Median lap difference vs base car (shared runs) |
  |---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 130 | 4.59 m/s | — |
  | E4 car | 120, 127, 125, 125, 123 | 124 | 4.75 m/s | −0.50 s |
  | E4 car, throttle 95% | 113, 118, 125, 127, 127 | 122 | 4.73 m/s | −0.32 s |
  | E4 car, throttle 90% | 107, 126, 121, 120, 127 | 120 | 4.70 m/s | −0.06 s |

  - It didn't help: less throttle gave back the lap time but no finishes (122 and 120, within noise of 124 or slightly worse).
  - Corner entry speed barely moved (4.75 → 4.70 m/s): the extra corner speed doesn't come from too much throttle, so a throttle limit can't remove it.
  - **Prediction vs. result:** predicted it won't help; it didn't.
- **Explanation (owner):** _pending_

## Stepping back (owner: "Take a step back… Are we putting too many restrictions in place? Are we stuck in a narrow paradigm?")

The owner asked for research on what would really move the needle toward a good racer, instead of chasing small gains. Up to a day of the agent's own work before reporting back.

### What the literature does (`docs/research/beyond-the-teacher-literature.md`)

- Every system in the review that beat its baseline driver (GT Sophy, Swift drone racing, an F1TENTH car that beat expert human drivers, legged robots on rough terrain) trained by RL on a **dense progress reward** with **privileged information** (true state, track ahead).
  - Where the deployed car had only sensors, that privileged policy was then distilled into it, or trained it through a critic that sees the true state.
  - None capped itself by copying a hand-written driver. A copied teacher is a ceiling.
- **Budget:** those runs used about 10^8 steps with 80k–150k steps per update. Every RL run in this project so far: 10^6 steps, 2,048 steps per update, one core.

### Speeding up the sim

- Profiling showed a third of the sim's time went to one small numpy call: the cross product in each tire's slip speed, run 40 times per control step.
- Written out by hand, it gives bit-identical results and about 1.4× the speed.

### Headroom: how fast can a driver that knows the track go? (`python -m sdc.phase47.headroom`)

- **Setup:** the re-tuned teacher (E4) driven more aggressively on 50 practice loops × 3 starts, gate range, tiny action noise, 2 passes each. Jump approach speeds were left unchanged.
  - "Corners" = the cornering and braking limit the teacher plans with (default 2.5 m/s²).
  - "Cruise" = the speed away from jumps (default 7.4 m/s).

  | Setting | Finished per pass (of 150) | Median lap | Lap vs teacher (shared runs) |
  |---|---|---|---|
  | Teacher (E4) | 135, 134 | 30.8 s | — |
  | Cruise 9 | 134, 135 | 30.4 s | −0.4 s |
  | Cruise 11 | 135, 132 | 30.3 s | −0.4 s |
  | **Corners 3.5** | 135, 131 | **28.2 s** | **−2.6 s** |
  | Corners 4.5 | 127, 125 | 26.9 s | −3.9 s |
  | Corners 6 | 108, 105 | 26.3 s | −4.7 s |
  | Corners 3.5, cruise 9 | 127, 133 | 27.4 s | −3.4 s |
  | Corners 4.5, cruise 11 | 94, 92 | 25.7 s | −5.2 s |
  | Corners 6, cruise 13 | 55, 46 | 24.9 s | −5.4 s |

- **What it shows:**
  - The teacher, and so every car copied from it, corners far below the tires' limit.
  - Cornering at 3.5 m/s² is 2.6 s a lap faster (8%) at about the same finishes (133 against 134.5 a pass).
  - Every car in Phases 4.6–4.7 has lapped in 30.5–31 s. All of them were copying this caution.
  - Cruise speed matters little: the straights are short.
- **What follows:** beating this teacher on time is easy for a driver that knows the track. Doing it from sensors alone is the hard part, because sharp corners are where the sensor car already fails (E3 analysis).

### Pilot: an RL racer with privileged information (`python -m sdc.phase47.racer`)

- **Question:** given the true state, the track ahead and the physics, and a reward for progress, does RL find driving better than the best hand-coded teacher?
- **Setup:**
  - **Inputs:** true state, curvature and height 30 m ahead, the next jump, grip, mass and motor.
  - **Reward:** metres of progress, −50 for a crash, +50 for a finish.
  - **Start:** a copy (4 DAgger rounds) of the teacher with 3.5 m/s² corners.
  - **Training:** the critic first, then PPO with the owner's GAE and losses on 128 cars across 4 processes, 32k steps per update, 181 practice loops, training-range physics. 9 h.
  - **Scoring:** every 3M steps on the 150 selection runs, 2 noisy passes.
- **Prediction:** none asked (agent's research step, owner away).
- **Result:** stopped at 9M steps (3.6 h). PPO finds speed the teacher can't, but it spends finishes to get it. The teacher's own trade-off on the same 150 selection runs is measured by making it corner harder (`runs/p47-frontier-select.json`).

  | Driver | Finished per pass | Median lap |
  |---|---|---|
  | Teacher, corners 3.5 (the pilot's start) | 136 | 28.0 s |
  | Teacher, corners 4.0 | 130 | 27.3 s |
  | Teacher, corners 4.5 | 127.5 | 27.0 s |
  | Teacher, corners 5.0 | 112.5 | 26.6 s |
  | RL, copy of the teacher (0 steps) | 132.5 | 28.1 s |
  | RL, 3M steps | 129 | 27.2 s |
  | RL, 6M steps | 124.5 | 26.0 s |
  | RL, 9M steps | 113 | 25.0 s |

  - At 6M steps RL is 0.6 s faster than the teacher at corners 5.0 and finishes 12 more runs per pass. At 9M it finishes as often as that teacher and is 1.6 s faster. No teacher setting gets near 25 s at any finish rate (headroom table above).
  - So the frontier moved outward. But the reward let RL pick a point on it the goal doesn't want: a crash cost 50 m of progress, about 7 s of driving.
- **Crash price 200** (from the 6M checkpoint, `runs/p47-racer-s1`): 3M steps gave 130.5 at 25.9 s; 6M gave 120.5 at 25.3 s. Finishes came back at first, then the driver drifted toward speed again.
- **Crash price 500, horizon about 20 s (γ 0.999)** (from s1's 3M checkpoint, `runs/p47-racer-s2`): 3M steps gave 125.5 at 25.4 s; 6M gave 121.5 at 25.1 s. Same drift. Stopped.
- **Best driver, 5 passes** (s1 at 3M): 133, 137, 131, 135, 128, mean **132.8** at **25.9 s**. Against the teacher (137 at 30.8 s): 4 fewer finishes per pass, about 5 s faster. Not a success under the goal.
- **Where it loses** (`python -m sdc.phase47.by_grip`, 1 pass):
  - Not on low grip. On runs with friction 0.55–0.7, below anything seen in training, RL finished 20/25 and the teacher 22/25.
  - Failures look like the teacher's, mostly off the lane (RL 9 off the lane, 3 rolled; teacher 10 and 4).
- **Reading (agent):** speed pays a little on every step. A crash is rare and partly chaotic, so its cost reaches the learner as a weak, noisy signal, and a bigger price or a longer horizon didn't fix that.
- **Next (running):** a dense warning instead of only the rare crash. A small cost on every step the car is more than 1.0 m off the centre line (the lane is 1.5 m each side). `--set margin=0.5 --set crash_pen=200`, from s1 at 3M (`runs/p47-racer-s3`).
  - 3M steps: 129, 131, mean 130.0 at 25.5 s. Finishes flat, laps a little faster. Every reward tried so far (crash 50, 200, 500 with long horizon, lane margin) settles near 130 finishes per pass. Still running.

### Owner's answers after the RL pilot

- **Which car would you race?** "The faster one. The slow ones probably wins more but it isn't as exciting. And crashes are common in rc racing."
- **What limits finishes?** "Crashing is a matter of luck and random chance at some point. Better driving can't prevent it."
- **What survives on sensors?** "We would lose some route planning. But I'm thinking that driving is more dependent on ability to react rather than plan. It never learned or needed a racing line. I think it'll be a little less reliable on the jumps"

### New goal (owner, after the RL pilot)

- **Owner:** "Let's do total time for a 20 lap race. Crashes cost 7 seconds."
- **Scoring:** race time = 20 × median lap + 20 × crash rate × 7 s, from the 5 noisy passes of the 150 selection runs (each run counts as one lap). Lower is better; success is a sensors-only car with a lower race time than the teacher.
- **What it means (agent):** 1 s per lap is worth 20 s, about 3 crashes. A car can crash on 14 more laps in 100 for each second it gains per lap and still break even.

  | Driver | Crashes per lap | Race |
  |---|---|---|
  | Base car | 0.133 | 633 s |
  | Teacher | 0.087 | 628 s |
  | Re-tuned teacher | 0.053 | 622 s |
  | E4 students | 0.173 | 628 s |
  | Teacher, corners 5.0 | 0.250 | 567 s |
  | Best RL racer (privileged) | 0.115 | 534 s |
  | RL racer, 9M steps (privileged) | 0.247 | 535 s |

### E5 — Copy the RL racer onto sensors

- **Setup:** the base recipe (`bc_loop --truggy --rounds 16 --expo 0.4`, 2 seeds averaged), with the privileged RL racer as the teacher (`--teacher-racer runs/p47-racer-s1/ckpt/3014656.pt`; `bash sdc/phase47/e5_runs.sh`).
  - The racer was trained without steering expo. On the expo car it sees its last steering as the servo angle it asked for, and its steering goes through the expo inverse, as the hand-coded teacher's does (`racer.ExpoDriver`).
  - Check, 1 pass on the selection runs with expo: 135 finished at 25.8 s (without expo, 5 passes: 132.8 at 25.9 s).
  - Scoring: 5 noisy passes of the 150 selection runs, racer and students (`python -m sdc.phase47.where e5`).
- **Prediction (owner):** "a little less reliable on the jumps"; driving depends more on reacting than planning, so most of the speed should carry over.
- **Result:** a success under the owner's race rule. The sensors-only E5 car beats the teacher by 53 s over 20 laps. It keeps nearly all of the racer's speed, but crashes on 4 laps in 10. Scored over 5 passes × 150 selection runs; laps are paired against the teacher (`runs/p47-e5-where.json`).

  | Driver | Finished per pass | Crashes per lap | Median lap | Lap vs teacher | 20-lap race |
  |---|---|---|---|---|---|
  | Base car | 127, 133, 135, 126, 131 | 0.131 | 30.7 s | −0.06 s | 633 s |
  | Teacher | 139, 136, 141, 127, 142 | 0.087 | 30.8 s | — | 628 s |
  | Re-tuned teacher | 136, 144, 142, 142, 144 | 0.056 | 30.5 s | −0.10 s | 618 s |
  | E4 students | — (mean 124) | 0.173 | 30.4 s | −0.49 s | 631 s |
  | RL racer (privileged) | 132, 136, 134, 130, 138 | 0.107 | 25.8 s | −4.92 s | 531 s |
  | **E5 students** | 97, 89, 87, 89, 99 | 0.385 | 26.1 s | −4.78 s | **575 s** |

  - **Speed carried over** almost whole: −4.78 s against the racer's −4.92 s.
  - **Reliability did not.** E5 finished 92 per pass against the racer's 134. Its 289 failures (5 passes):
    - 198 in corners (off the lane)
    - 48 elsewhere off the lane
    - 38 on landings
  - **Jumps:** E5 landed 94.3% of all jumps (racer 96.3%, teacher 97.5%). Doubles were *better* (98.8%; racer 94.8%), drops worse (87.2%; racer 95.5%), tables about the same (97.9%; racer 98.3%).
  - **Corner entry:** E5 enters corners at 6.12 m/s, faster than the racer it copies (5.77). E4's students did the same against their teacher (4.75 against 4.12).
  - Validation (60 runs, start 0 and others): 40/60 and 31/60, against E4's 53 and 49.
  - Owner's prediction: speed survives ("driving is more dependent on ability to react rather than plan"): **right**. "A little less reliable on the jumps": **partly**. It is less reliable, but by far more than a little, and mostly in corners rather than on jumps.
- **Explanation (owner):**
  1. "The sensor car doesn't know to brake in time because it can't see the corner coming. It's luck that it is still able to make corners, either by drastic maneuvers or because the teacher left some on the table. It also must learn to brake randomly after a long period of relative straight and acceleration. Or just to arbitrarily got the brakes sometimes."
  2. "The sensor driver can't see through the corner, so it doesn't know how sharp or long the corner is."
  3. "I like the idea of skipping the teacher, but we've always found that transfer learning from the teacher works better. Maybe a bigger network and more training would work? But yes having the teacher match the students knowledge send likely to help so that the student has clear reasons for actions."

### After E5 — Checking the owner's explanation against the traces (`python -m sdc.phase47.corners`)

Every sharp turn (|curvature| > 0.15 /m) each driver reached, from where.py's 0.1 s traces (5 passes × 150 selection runs). "Brake" means a command below −0.2.

| | Teacher | Base | E4 | RL racer | E5 |
|---|---|---|---|---|---|
| Turns reached | 4728 | 4618 | 4510 | 4633 | 3902 |
| Speed 8 m before (m/s) | 7.11 | 6.27 | 6.46 | 7.52 | 7.37 |
| Speed at turn start (m/s) | 4.13 | 4.62 | 4.78 | 5.79 | 6.16 |
| Slowest in turn (m/s) | 3.35 | 3.47 | 3.52 | 4.57 | 4.77 |
| Braking starts (m before turn) | 0.1 | 2.3 | 2.7 | 2.6 | 1.8 |
| Turns with no braking | 90% | 25% | 22% | 5% | 16% |
| Entry speed, mild turns (peak ≤ 0.3 /m) | 4.15 | 4.61 | 4.77 | 5.83 | 6.11 |
| Entry speed, sharp turns (peak > 0.3 /m) | 3.99 | 4.68 | 4.86 | 5.61 | 6.39 |
| Run ended in a mild / sharp turn | 0.0% / 0.0% | 0.6% / 1.3% | 1.1% / 0.5% | 0.5% / 0.2% | 4.9% / 4.4% |
| Largest steering in turn (median) | 0.44 | 0.60 | 0.62 | 0.30 | 0.46 |
| Braking with no sharp turn in the next 15 m | 0% | 7.6% | 7.0% | 2.5% | 7.7% |
| Braking with a sharp turn within 4 m | 97% | 87% | 87% | 96% | 90% |

- **"Doesn't know to brake in time":** supported. E5 starts braking later than the racer (1.8 m before the turn against 2.6 m), skips braking three times as often (16% of turns against 5%), and enters 0.4 m/s faster.
- **"Drastic maneuvers":** supported. E5 steers about 50% harder in turns (0.46 against 0.30).
- **"Brake randomly / arbitrarily":** supported, but small. 7.7% of E5's braking has no sharp turn in the next 15 m, against the racer's 2.5%. The base car and E4 do the same (7–8%), so every sensor car does it. Most braking is still near turns.
- **"Doesn't know how sharp the corner is":** supported for speed. Every driver that sees the track enters sharp turns slower than mild ones (racer −0.22 m/s, teacher −0.16). Every sensor car enters them *faster* (E5 +0.28, E4 +0.09, base +0.07), so they don't adjust to sharpness at all. But E5 doesn't crash more in sharp turns (4.4%) than in mild ones (4.9%): its corner crashes come from too much speed everywhere, not mainly in the sharp ones.
- **The racer's braking:** 96% of it happens with the turn already within 4 m, inside the sensors' range. It brakes late and hard, knowing what's coming.

### E6a — A teacher that sees only as far as the sensors

- **Question:** how much of the racer's speed comes from seeing the track ahead? A teacher that sees only what the student can would give the student "clear reasons for actions" (owner).
- **Setup:** the pilot's recipe (copy of the teacher, then PPO, crash cost 200, 9 h, `runs/p47-e6a`). The driver sees the track only 4 m ahead (`--set preview=4.0`): curvature at 0, 2 and 4 m, heights to 4 m, and the next jump only within 4 m. Beyond that the track reads as flat and straight. It keeps the true state and the physics.
- **Prediction (owner):** "a little slower and a little safer. It's only fast now because it doesn't know to prepare for corners (ie it's reckless)"
- **Result:** seeing 4 m instead of 30 m cost almost nothing. Training ran like the pilot's first run, eval for eval (2 passes each):

  | Steps | E6a (sees 4 m) | Pilot s0 (sees 30 m) |
  |---|---|---|
  | 0 (copy of the teacher) | 136.5 at 28.1 s | 132.5 at 28.1 s |
  | 3M | 126.0 at 27.1 s | 129.0 at 27.2 s |
  | 6M | 123.0 at 26.0 s | 124.5 at 26.0 s |
  | 9M | 114.0 at 25.2 s | 113.0 at 25.0 s |

  - 5 passes, 6M checkpoint: 124.0 finished per pass (124, 123, 121, 126, 126) at 25.87 s, race **541.7 s**. The best full-sight racer: 134 at 25.82 s, race 531.3 s. Same speed, 10 more crashes in 1,000 laps, 10 s slower over 20 laps.
  - Training was stopped after the 9M eval: like the full-sight runs, it was drifting toward speed and losing finishes.
- **Next:** E6, copy E6a onto sensors (below).

### E6 — Copy the 4 m racer onto sensors

- **Setup:** as E5, with E6a's 6M checkpoint as the teacher (`bash sdc/phase47/e6_runs.sh`; `bc_loop --teacher-racer runs/p47-e6a/ckpt/6029312.pt --teacher-preview 4.0`), then `where e6`.
- **Prediction (owner):** not given before the result came in.
- **Result:** worse than E5. A teacher that sees only what the sensors see did not make it easier to copy.

  | Car | Finished per pass | Crashes per lap | Median lap | Race |
  |---|---|---|---|---|
  | Teacher (hand-coded) | 137 | 0.087 | 30.78 s | 627.7 s |
  | E5 (copies the 30 m racer) | 92.2 | 0.385 | 26.06 s | 575.1 s |
  | E6 (copies the 4 m racer) | 78.8 (77, 83, 73, 81, 80) | 0.475 | 26.35 s | 593.5 s |

  - Failures (5 passes): corners off the lane 251 (E5 198), other off the lane 54 (48), landings 50 (38).
  - Students at the end of training (60 runs): E6 29 and 34 finished; E5 40 and 31.
  - Still beats the teacher's race by 34 s, but E5 remains the best sensor car.
- **Explanation (owner),** asked why copying loses crashes but hardly any speed: "Is there something in the setup? Like what you do now impacts how likely you are to crash a few moves ahead? I still haven't determined if we should be working towards a clean racing line that plans ahead or a scrappy racer that is good at reacting. I think we've tried both and neither has helped. Maybe we found the happy medium? I find that hard to believe."

### E7 — Train the sensor car itself, with a coach that sees the whole track

- **Question:** most of the loss is in the copying (racers crash on 11–17% of laps, their students on 39–48%). Can RL on the student's own driving win it back?
- **Setup:** `python -m sdc.phase47.e7 --out runs/p47-e7` (9 h).
  - Actor: E5's seed-0 student (GRU, sensors only), trained by PPO on 64-step sequences from its own memory.
  - Critic: a separate network on the racer's privileged inputs, started from the racer's critic and warmed for 1M steps before the actor changes. It is used only in training.
  - Reward as the racer (progress, crash 200, finish 50). Student env (truggy, expo 0.4), E2's train loops. Scored every 3M steps on the 150 selection runs (2 passes); best checkpoint then scored over 5 passes.
- **Prediction (owner):** "Crash more but go faster. Won't beat e5."
- **Result:** stopped on the old car, before its first score, when the owner switched to the full-size truggy (next section). Rerun there as E7 below.

## The full-size truggy (`SDC_REAL=truggy2`; `docs/phases/air-control.md`, `docs/phases/truggy-size.md`)

The owner saw too little air control in renders, and a chassis riding about 4 cm too high. Fixes: wheel spin inertia that couples to the body in the air (`wheels`), no rollover end while airborne (`groundroll`), and a full-size 1/8 truggy body, wheels, springs and motor (`size8`). Owner: "A, go with the full-size truggy." Everything after this section runs on it.

- **Teacher retune** (`python -m sdc.phase47.air_teacher`): the old gains still work best; the double-jump speed goes from 7.4 to 7.8 m/s.
- **RL racer** (`runs/p48-racer`, 9 h, crash price 200), 2-pass race estimates: 585 s at start, 553 at 9M, 538 at 15M, 560 at 18M, 535 at the end (19.7M, chosen).
- **Sensor cars** (`runs/p48-e5-s0/s1`): the E5 recipe copying the chosen racer.

Race over 5 noisy passes of the 150 selection runs (`python -m sdc.phase47.where p48 ...`, `runs/p48-race.json`):

| Driver | Finished | Crash rate | Median lap | Race |
|---|---|---|---|---|
| Teacher | 672/750 | 10.4% | 28.41 s | 582.8 s |
| RL racer (privileged) | 531/750 | 29.2% | 24.96 s | 540.1 s |
| Sensor cars (E5 recipe) | 335/750 | 55.3% | 25.48 s | 587.1 s |
| E7 (RL on the sensor car, below) | 362/750 | 51.7% | 24.56 s | **563.6 s** |

The sensor cars lose to the teacher by 4.3 s. They lap only 0.5 s slower than the racer but crash almost twice as often (55% against 29%). On the old car the gap was 39–48% against 11–17%.

### E7 on the full-size truggy (`sdc/phase47/p48_runs.sh e7`)

Same setup as E7 above: actor `runs/p48-e5-s0`, critic from the chosen racer, 9 h.

- 2-pass race estimates: 600 s at start, 601 at 3M, 588 at 6M, 591 at 9M, 579 at 12M, 571 at 15M, 568 at 18M, 561 at 21.1M (chosen; `runs/p48-e7`).
- **Result:** over 5 noisy passes, 362/750 finished (crash rate 51.7%), median lap 24.56 s, **race 563.6 s**: 19.2 s ahead of the teacher (582.8 s) and 23.5 s ahead of the E5-recipe cars (587.1 s). **A sensors-only car beats the teacher.**
- It wins on pace: it laps 3.85 s faster than the teacher and as fast as the RL racer (24.96 s), but still crashes on half its laps (teacher 10%, racer 29%).
- **Caveat: grip.** Each run draws its own grip (gate range, friction 0.55–1.25), so the 563.6 s averages over grip. Split by grip third (same formula), E7 wins at mid and high grip but loses badly at low grip:

  | Grip (friction) | Teacher | RL racer | E7 |
  |---|---|---|---|
  | 0.55–0.79 | 593 s (crash 18%) | 565 s (38%) | 620 s (80%) |
  | 0.79–0.96 | 582 s (9%) | 531 s (30%) | 552 s (43%) |
  | 0.96–1.25 | 574 s (4%) | 521 s (20%) | 525 s (32%) |

  The replay session's full 20-lap races (one grip for the whole race) agree: at grip 0.58–0.70, E7 crashed 38–65 times per race, the teacher 0–14. The formula also counts at most one crash per lap; in full races E7 sometimes crashed 3 times in one lap.
- Weights on the `car-weights` branch: `runs/p48-e7` (and `runs/p48-e5-s0/s1`). Run with `SDC_REAL=truggy2`.

### Owner's answers after E7 (2026-11-28)

- "it's being less safe because crashes don't cost much time. If it can crash on half the laps and still beat the teacher, maybe 7 seconds isn't long enough for the corner marshall."
- "I'm up for trying the gate after we address the traction issue. We could tweak all day but it appears we finally succeeded in using RL to improve the driving."
- "Probably win on average is sufficient, but we should think more about of there's a way to adapt to the conditions."
- "is the car still estimating traction? Or does it just drive for some average condition. It might not be responding to grip because it doesn't sense it at all"

What the code says about grip:
- **Sensor car (E5, E7 actor):** no grip input. It gets the IMU, gyro, front and rear wheel speeds, rays and its last action. Grip could only be inferred from slip (wheel speed against what the IMU says the car is doing), through the GRU's memory. Nothing in training asks for that.
- **RL racer and E7's critic:** see the true friction (`priv_obs` ends with friction, mass, motor). So the racer can drive faster on high grip and slower on low grip, and the student copies the result without the cause.
- **Teacher:** doesn't know grip either (`teacher.py`: "A_LAT is one fixed number, not the car's true grip"). It survives low grip because its corner budget (3.5 m/s²) is cautious for every grip.

### E7b — E7 with a grip head (option B), against a control (owner, 2026-11-28: "We already tried C and it didn't help. A seems similar, so let's try B.")

- **Physics change first (owner: "any ground contact from the chassis give an immediate stop. That's not realistic - momentum means it will slide"):** the chassis box caught the heightfield's prism walls and had tyre grip (1.0), so a slam stopped the car from 7 m/s in 40 ms. New switch `skid` (in `truggy2`): the box collides through three rounded capsules with plastic-on-dirt friction 0.4 (scaled with the episode's grip). A 5 m/s slam now keeps about 3 m/s and slides on. All earlier full-size scores were on the old chassis (`runs/old-chassis/`).
- **Setup:** two E7 runs from `runs/p48-e5-s0`, 12 h, 2 workers each, new chassis. `runs/p48-e7b`: `grip_coef=1.0`, a head on the actor's memory predicts the episode's true grip and its squared error joins the loss. `runs/p48-e7c`: control, the same head trained on a detached memory (a probe: does the memory already know grip?).
- **Prediction (owner):** "Intuitively, grip sensing is important and will help a lot. If E7b doesn't win over the control, I think it means we are still missing something. So I hope it does win, but I feel like we've tried similar enough things in the past that I'm not convinced it'll help."
- **Teacher on the new chassis** (5 passes): 671/750 finished, median lap 27.50 s, **race 564.7 s** (was 582.8 s on the old chassis): the new bar.
- **Result at 3M steps (2 noisy passes):**

| Run | Finished /150 | Lap | Race estimate | Grip error (MSE) |
|---|---|---|---|---|
| Both at step 0 | 59.5 / 56.5 | 24.22 / 24.54 s | ~569 s | — |
| E7b (grip head trained) | 34.0 | 24.95 s | ~608 s | 0.040 |
| E7c (control, probe only) | 53.0 | 24.30 s | ~577 s | 0.147 (still falling) |
| E7c at 6M | 58.5 | 23.46 s | ~555 s | 0.052 |
| E7c at 9M | 65.5 | 24.08 s | ~560 s | 0.034 (below the 0.041 average guess) |
| E7c at 12M | 72.5 | 23.72 s | ~547 s | 0.025 |

Owner, on how a driver senses grip: "I get grip info throughout. Moreso at the start and less later. But I know how the car should handle from previous driving and pretty quickly realize when it fits that and when it doesn't."

Correction (owner caught it): the truggy is 4WD, not rear-drive. One motor drives all four wheels through limited-slip diffs (`car.py`: `drive`, `lsd`), so front and rear wheel speeds give almost no slip contrast.

Owner, on testing grip: "I do kind of goose the throttle early. So I'll start with a slower lap but add some throttle kicks to see how it responds while still leaving room to recover. Later in the race I tune that understanding based on how it slides when driven at the limit and on feedback from going too hard and crashing."

4WD check (`truggy2`, full throttle from rest, flat ground): the one `motor` actuator puts the same torque on all four wheels (0.91 N·m each at 0.1 s, 0.48 at 1.2 s). At 1.2 s the car does 5.9 m/s while the wheels turn at 8.1–8.8 m/s: full throttle spins all four. Simplification: the motor gives each wheel equal torque whatever the others do. In a real open diff a wheel in the air takes almost no torque and its partner loses drive too; here the grounded wheels keep theirs (closer to a heavily oiled diff, as racing truggies run).

Correction after research ([truggy diffs](../research/truggy-diffs.md)): equal torque to every wheel, with motor speed = mean wheel speed, is exactly three ideal open diffs, so the claim above was wrong. In a real car a lifted wheel starves the others only through the motor: back-EMF (modelled) and rotor inertia (missing). Stock 1/8 truggy oils are about 10k front, 10k centre, 5k rear (Tekno 20k/20k/10k, Associated 10k/10k/5k). Estimated wheel-referred damping: axle diffs about 0.002 N·m·s/rad (nearly open; our 0.002/0.001 match), centre about 0.04 (ours 0.03), stiffer because it sits behind the 4.4:1 ring and pinion. New opt-in switch `rotor`: the motor's rotor as a body about the long axis, 14:1 to the mean wheel speed (1.5e-3 kg·m² at the wheels, replacing `heavy`'s per-wheel drivetrain armature), centre damping 0.04. Check: the ratio holds at 14.00; same acceleration (5.86 m/s at 1.2 s against 5.85); the wheels spin up more slowly off the line (4.4 against 5.6 m/s at 0.2 s). Not in `truggy2` yet, so running experiments stay comparable.

Owner, on the test method: "I prefer the throttle test of you think it could work. Earlier we measured acceleration, but when I do this test the throttle is just meant to make the wheels slide but my feedback comes from how the car responds along with how easy it was to make them slide (because I have a reference for how hard it should be). Maybe that give us a method - drive the car on a few known surfaces (or roughly known) as part of training and build that expectation into memories. Or just simulate that from knowing torque and friction coefficients"

Owner, on surfaces: "grip requires a grip map since it isn't uniform through the track. I'm not sure we ever fully settled that". In the Phase 4+ jump sim, grip is one number per run (`dr="gate"`: friction 0.55–1.25); Phase 3.5's dirt patches were not carried over.

Straight-line tests on `truggy2` (flat, one run per grip). Brake: full brake from 7 m/s. Throttle: full throttle for 0.3 s from 3 m/s; wheelspin = wheel speed − car speed at the end.

| Grip | Brake decel | Throttle accel | Wheelspin after kick |
|---|---|---|---|
| 0.55 | 4.3 m/s² | 4.0 m/s² | 3.9 m/s |
| 0.75 | 6.0 | 6.1 | 1.2 |
| 0.90 | 7.0 | 6.6 | 0.5 |
| 1.05 | 7.1 | 6.8 | 0.2 |
| 1.25 | 7.2 | 6.9 | 0.2 |

Both tests read grip up to about 1.0, then saturate: the brake (capped at 0.65 of motor torque) and the motor can't break the tyres loose any more. Wheelspin is the clearest reading (20× from 0.55 to 0.9), the owner's "how easy it was to make them slide".

  Guessing the average grip every time gives an MSE of 0.041 (the variance of grip drawn evenly from 0.55–1.25). E7b reached 0.040, so its memory learned no grip at all; only the head's bias moved. Making the memory carry that target cost driving: finishes fell from 59.5 to 34. E7b was stopped at 3M to give E7c the CPU. E7c's probe is still above the average-guess level, so it shows no grip in the memory either.

### Owner on racing lines (2026-11-28)

"Watching the replay I see two behaviors that mean I could race better than this car. First, it seems to aim for the center line. Sometimes it makes big corrections to get there. I think that's what the teacher does and it makes sense because it maximizes buffer of track space. But a human driver would aim for more of a racing line - enter corners wide, turn early, hit the apex, etc. Rarely do I actually aim for the the center line. I almost always want to be near an edge. Perhaps related, I see the car turning late instead of early. Like it's well into the corner and goes oh crap I'd better turn! My hope was that the teacher would teach it to get around the track and use its controls. And then RL would get it finding these tricks for how to really race. I don't think we've gotten there. Although I did say that it finally looks more like racing. I think that's because there are corners where I see it slide and counter steer. And it appears to be attacking the track a little more."

Measured on the old-chassis traces (first 300 runs each; lane is 3 m wide; "inside" = offset toward the corner's centre where |curvature| > 0.1/m; steering lag = the shift that best lines steering up with the track's curvature at the car):

| Driver | Mean distance from centre line | Inside offset in corners | Steering vs curvature |
|---|---|---|---|
| Teacher | 0.07 m | −0.03 m | 0.1 s early |
| RL racer | 0.39 m | −0.02 m | on time |
| E7 | 0.35 m | −0.09 m | 0.1 s late |

The teacher holds the centre line. RL moves off it, but not into a racing line: no inside bias in corners (slightly outside). E7 turns 0.1 s after the curve starts.

Owner, on how far ahead they look: "I think I setup for the next corner immediately after leaving the previous one. But sensors can't see well for a long distance. 6m might be all we can hope for"

Owner, on surfaces (2026-11-28): "On a real track, grip is mostly uniform apart from two things - areas that are bumpy and areas where cars have either made the dirt smooth or loose. We added some bumps already, but maybe need more right surface through out to be realistic? And we don't model track wear. Any ideas to add that? Or is it too complex and we should just cheat with a little grip noise rate than a map"

Ideas offered: a per-tyre grip field (groove +10% on the common line, loose −15% off it and outside corners; each tyre's grip is already set every step for `slide`), braking bumps before corners and after landings, track wear over laps (later). Owner, on where the groove comes from: "The RL racer's line, since it's actually faster and where cars will race. But the ideal racing line plus some noise would work, too."
