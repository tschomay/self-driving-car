# Phase 4.6 — A 3D course with turns

- **Status:** in progress
- **Goal:** a 3D course with turns, jumps, and rough ground, driven with Phase 4.5's realistic car: parts-list sensors, randomized physics, the owner's GRU and filter, and a map the car builds itself on an out-lap.
- **Learn:** steering and pitch control at the same time, lateral load and body roll in turns, a teacher the student can follow on a harder course, mapping a course that turns.
- **Gate (owner, 2026-11-16):** at least 80% of runs finished (48 of 60) on fresh test loops 46000–46019, 3 starts each, with parts-list sensors and gate-range physics. A run is a crawl out-lap plus one timed lap. It is finished if the timed lap is completed (no rollover, off the lane, or stuck) within 1.5× the teacher's time. The jump landing rate is reported, not gated.
- **Code:** `sdc/jumps/` (sim), `sdc/phase46/` (training, evaluation, reports).
- **How to reproduce the gate:** to be filled in at the gate.

## Phase-entry decisions

Owner, 2026-11-16: "Agree on all but lap completion can come from a finish line sensor."

1. **Course shape:** a closed loop. Phase 1's spline track, shrunk (about 120–150 m, 3 m lane) and laid on 3D ground, with 3 jumps on its straights. The car crawls one lap to build its map, then races the same loop.
2. **Gate course:** jumps on straights, rough ground and whoops, flat and banked turns. Jumps near turns and off-camber turns are a stretch experiment after the gate.
3. **Finished:** as in the gate above (timed lap completed within 1.5× the teacher's time).
4. **Learner-owned code:** a dead-reckoning estimator. Heading from the gyro's yaw rate plus distance from motor speed gives a 2D position, which draws the map of the turns.
5. **Sensors for turns:** no camera for the gate. First rely on the out-lap map; add cheap range sensors (about $30) if E1 shows the map isn't enough.
6. **Finish-line sensor (owner's addition):** the car senses the finish line itself. Race timing systems make the car passive (a transponder that the track reads), so the car needs something extra: modelled as an onboard receiver that pulses when it passes a beacon at the line. It counts laps and can reset the dead-reckoning drift once per lap.
7. **Compute:** container CPU, $0; Modal only if a run passes about an hour.

## Plan

0. **Build (agent):**
   - Loop courses: the spline centreline on a 2D heightfield with banks, jumps on straights, whoops, rough ground, banked turns.
   - Lane scoring along the centreline; the finish-line sensor; lap timing.
   - A teacher for turns (steers along the centreline, slows for corners, 4.5's jump speeds and air control), and a crawl driver that can follow a turning lane on sensors only.
1. **E1 — What breaks on turns?** Phase 4.5's gate car (trained on straight lanes) and the new teacher on loop courses. How far ahead do the parts-list sensors see a turn at race speed?
2. **E2 — Your dead reckoning.** How far does the estimated position drift over a crawl lap and a race lap, with and without the finish-line reset? Is the map of turns it draws good enough to plan corner speeds?
3. **E3 — Copy the teacher on loops.** The owner's GRU and filter, with and without the map of turns as inputs.
4. **E4 — Close the gap.** Depends on E3: extra range sensors, a teacher tuned to what the student can sense, or fine-tuning.
5. **Gate** on test loops 46000–46019.
6. **Stretch (after the gate):** jumps near turns and off-camber turns.

## The loop courses (built 2026-11-16)

Code: `sdc/jumps/loop.py` (courses), `sdc/jumps/render_loop.py` (GIFs), `sdc/phase46/` (teacher, experiments), options on `car.py` and `env.py` (`loop=True`). Tests: `tests/test_loop.py`. The straight courses are unchanged (same generator output, bit for bit).

- **Shape:** a random polygon with 4–6 rounded corners, so it has true straights joined by turns of 3–10 m radius. Phase 1's spline generator was tried first in spirit, but shrunk to this size it left no straights long enough for a jump. A spare straight gets a chicane (an S-bend), so about 9 loops in 10 turn both ways.
- **Size:** laps of 130–175 m (planned: 120–150 m). Three jumps each need about 20 m of straight; shorter laps left no room. The lane is 3 m between banks, as before.
- **On the lane:** three of Phase 4.5's roller-course jumps (double, table, drop), one per straight, the first after a 10 m run-up from the line; turns flat (about a third) or banked up to 12°; one 6–9 m stretch of whoops (3–9 cm bumps); a few millimetres of roughness elsewhere.
- **Track coordinates:** position is measured along the centreline (progress, the reward) and sideways from it (off the lane beyond 1.5 m). Jump scoring, the reward and Phase 4.5's air control work unchanged.
- **A run:** the car starts 1.5 m past the start/finish line and the run finishes when it crosses the line again: one lap, at most 60 s. A crawl out-lap is a separate run before it, as in Phase 4.5.
- **Car:** Phase 4.5's realistic car and parts-list sensors, unchanged (training and gate ranges too).
- **Teacher** (`sdc/phase46/teacher.py`): Phase 4.5's steady expert (7.4 m/s into every jump, pitch control in the air) plus turns. It steers by the curve just ahead plus the old lane-centring feedback, and slows so that every corner is taken at no more than 3.5 m/s² sideways, braking at 3 m/s². It reads the true state, like every teacher so far. It drives full laps; its scores are part of E1.
- **Sim speed:** about 600 steps/s per core. MuJoCo's own ray casting took 3 ms per sensor read on the large heightfield (20× slower than on the straight lane), so the range sensors now march the height map directly (agrees with MuJoCo's rays to about 1 cm).
- **Seeds:** validation loops 10000–10019, test loops 46000–46019 with physics seed 46 (gate only), practice loops from 100000.
- **Finish-line sensor:** the env reports the line crossing (the lap ends there). The crawl and the owner's dead reckoning will use it in E2.

## Experiments

### E1 — What breaks on turns?

- **Prediction (owner, before the run):** "Teacher 80%, car 45%. Going off the track ends runs" (laps finished of 60 at gate-range physics; the Phase 4.5 car's runs mostly end off the lane).
- **Setup:**
  - Drivers: the loop teacher (reads the true state), and Phase 4.5's car unchanged: the owner's GRU and filter, copied from the steady expert on straight roller lanes (E5 seed 0, rebuilt from the logged command; 84% on straight lanes, as logged).
  - 20 validation loops × 3 starts = 60 laps each, at nominal and gate-range physics, one car at a time.
  - Sensor check: the car stands on the centreline every 0.25 m before a turn. A sensor "sees" the turn when any reading differs by more than 0.25 m from the same stretch laid out straight (no noise). 40 turns that follow at least 11 m of clear straight.
  - `python -m sdc.phase46.e1`, about 4 min on 4 cores.
  - The teacher's scores were first seen during the build (tuning), before the prediction; the Phase 4.5 car and the sensor check were run after it.
- **Result:** [report](https://claude.ai/artifact/FwYEQ9uqiBm77HzCtBZtG1).

  | Driver | Physics | Laps finished | Median distance | Endings | Jump 1 landed |
  |---|---|---|---|---|---|
  | Teacher | nominal | 60/60 | 100% | | 60/60 |
  | Teacher | gate range | **56/60 (93%)** | 100% | 2 off the lane, 2 rolled over | |
  | Phase 4.5 car | nominal | 0/60 | 22% of a lap | 59 off the lane, 1 rolled over | 56/60 |
  | Phase 4.5 car | gate range | **0/60** | 23% of a lap | 58 off the lane, 2 rolled over | 53/60 |

  - The Phase 4.5 car still lands jump 1, then leaves the lane at the first turn in 50 of 60 runs (gate range). Eight more ended in the second turn and two in the third. The furthest run covered 35% of a lap.
  - The teacher laps in about 30 s and lands 88% of jumps at gate range (98% nominal). Its four failures: two rollovers landing jump 1, two off the lane (one with grip 0.56).
  - Sensors: they first register a turn a median 0.75 m before it begins (furthest 3.25 m); that is 0.1 s at 7.4 m/s. The 8×8 ToF sees it first; the side sensors never do before the turn begins. The teacher starts slowing a median 2.25 m before. The sensors saw the turn before that point in 6 of 40 turns.
  - Prediction vs. result: teacher 80% predicted, 93% measured. Car 45% predicted, 0% measured. "Going off the track ends runs": yes, 58 of 60.
- **Explanation (owner, after the run):** "Phase 4.5 learned that minimal steering input is important. Now it has to be aggressive about corners or it doesn't turn enough."
- **Agent's check (first turn, 10 validation loops, gate range):** the Phase 4.5 car does steer, harder than the teacher (peak command 0.41 of full lock vs 0.27). But it arrives at 6.5 m/s where the teacher is down to 4.8 m/s. At 6.5 m/s a 4 m radius needs about 10 m/s² sideways, more than the tires can give at any steering angle. It never learned to brake for a turn, and its sensors show the turn only 0.1 s ahead.
- **Takeaway:** on turns the missing skill is speed, not steering: the car has to slow down before it can see why. The knowledge of what is coming has to arrive some other way: the out-lap map (E2).

### E2 — Your dead reckoning

- **Owner's estimator (`sdc/phase46/my_deadreckoning.py`, 2026-11-16, "Agree" to the recommended set):**
  1. Heading from the gyro's turn rate, minus its bias measured at standstill.
  2. Tilt-corrected: the gyro is projected onto the vertical with the attitude filter's pitch and roll.
  3. Distance from motor speed, holding the last ground speed in free fall.
  4. At the finish line, the closing error is spread evenly back along the lap, so the mapped loop closes.
  - Agent's details: only the horizontal part of the distance counts (× cos pitch); the map's curvature is the heading change per metre, smoothed over 1 m.
- **Sim support:** `sdc/phase46/laps.py` records a lap's sensor readings with the truth alongside.
- **Crawl driver, not solved yet (2026-11-16):** Phase 4.5's sensor-only crawler finishes about 9 of 12 practice loops; a rewrite (`laps.LoopCrawler`) finishes 10 of 24. Causes found:
  - The two side sensors can't tell being off-centre from pointing the wrong way.
  - In a banked turn or under body roll, the low side's beam hits the ground and the high side's beam passes over the 0.35 m bank, so both go blind.
  - Hard steering rolls the body and makes it worse; kickers stall a 2 m/s crawl unless it speeds up on the climb.
- **Owner's decision (2026-11-16):** "We should do both 1 and 2. If it can't tell where it is for the out lap, it definitely can't while racing."
  1. A person drives the out-lap by remote while the car records. In the sim a slow hand-coded driver that reads the true position stands in (`teacher.OutLapDriver`). The timed lap stays autonomous.
  2. Two angled range sensors join the parts list (VL53L1X, about $10–20): 45° forward-left and forward-right, so the car can see its heading relative to the banks. Env option `angled=True`, two more inputs.
- **Prediction (owner, before the run):** "I think distance error with be worse and hurt more. I think we'll be plus/minus 20% on the slow lap and 40% racing." (Read as: position off by about 20% of the distance driven on the slow lap and 40% on the race lap, mostly from distance error, not heading drift.)

- **Setup:**
  - Slow lap: the remote-driven stand-in at 2 m/s (87 s). Race lap: the teacher (30 s, three jumps). 20 validation loops × 3 starts, gate-range physics, parts-list sensors.
  - Each lap's sensor log is replayed through your estimator and through its alternatives, one choice changed at a time. Scored against the true pose when the car is back at the line, before the finish-line correction.
  - `python -m sdc.phase46.e2`, about 8 min on 4 cores.
- **Result:** [report](https://claude.ai/artifact/Jzqry6UvE4pND4adhFSfj1).

  | Lap | Position off at the line | Share of distance | Heading off | Distance off | From heading alone | From distance alone |
  |---|---|---|---|---|---|---|
  | Slow (51 of 60 finished) | **8.3 m** | 5.5% | 16° | +3.2% | 7.8 m | 1.2 m |
  | Race (56 of 60 finished) | **3.3 m** | 2.0% | 3.6° | +4.2% | 2.2 m | 1.9 m |

  - Heading hurts more than distance, and the slow lap is worse than the race lap: it lasts three times as long, and the sim's gyro bias keeps wandering after the standstill calibration.
  - Your choices against the alternatives (position off, slow / race): steering-model heading 19 m / 26 m; blend 19 m / 25 m; no tilt correction 8.3 m / 4.0 m (race heading 6.7° instead of 3.6°); raw distance 8.6 m / 3.5 m (race distance +7.6% instead of +4.2%).
  - The map after your finish-line correction: mean position error 3.9 m → 2.3 m (slow), 2.6 m → 1.6 m (race). Mapped corner radii are within about 9% (0.2 m/s of corner speed). A mapped turn sits a median 2.9 m from the true one, mostly late (+2.4 m), because the odometer reads long; 1 in 10 is off by more than 8 m.
  - **Check, gyro drift model:** the sim's gyro bias wanders 0.05°/s per √s, set in Phase 4.5 when runs lasted 12 s. With a tenth of that, the slow lap is off by 3.1 m (4.2°) and the race lap by 2.4 m. The agent's recollection of the ICM-42688 datasheet is that real drift over a minute is well below the modelled value; not verified.
  - **Recorder bug found and fixed before any result was reported:** the lap recorder took the standstill gyro calibration after the run had ended and the car had been reset, so two of every three laps (the offset starts) used another run's calibration and drifted up to 85°.
  - The out-lap stand-in failed 9 of 60 slow laps at gate range, on 4 loops: stuck on a kicker (4), off the lane (4), rolled over (1). It needs fixing before E3, since a failed out-lap is a failed gate run.
  - Prediction vs. result: about 20% (slow) and 40% (race) predicted; 5.5% and 2.0% measured. "Distance error will hurt more": no, heading does, and the slow lap is the worse one.
- **Explanation (owner, after the run):** "I assumed that where you are on the track left to right would have a lot of impact on distance, but it seems that wasn't an issue for it and the gyro drift won out. Error for both is pretty low, but to be useful tolerance is even lower."
- **Takeaway:** dead reckoning's error grows with time through the heading, not with distance: a slow lap is the hard case. A few percent sounds small, but the car has to brake within about 2 m of the right place, so the map needs landmarks to re-sync on, not just a closed loop.
- **Owner's decision (2026-11-17):** "Yes check the fact sheet."
  - Found: no official bias-instability figure for the ICM-42688-P. Zero-rate drift with temperature is quoted as ±0.005 °/s per °C in a TDK comparison table and ±0.06 in distributor copies of the datasheet (they conflict). The BNO086's gyro-only heading is quoted at 0.5° per minute of dynamic drift.
  - So real heading drift is roughly 0.5–2° per minute, perhaps 9° per minute in the worst case. The sim's 0.05°/s per √s gave about 13° per minute.
  - Changed: `GYRO_WALK` is now 0.01°/s per √s (about 3° per minute; 5° at the gate range's double noise), for training and the gate alike. Phase 4.5's 12 s runs see about 1° or less either way, but its gate no longer reproduces bit for bit.
- **E2 again with the lowered drift** (`runs/p46-e2-new`): slow lap off by 3.8 m (2.2%, heading 4.5°), race lap 2.5 m (1.5%, heading 3.3°). Distance is unchanged (+3.2% / +4.2%), so heading and distance now matter about equally (slow: 2.4 m vs 1.2 m; race: 1.7 m vs 1.9 m).
- **Out-lap stand-in, after E2 (2026-11-17):** it now carries 3.5 m/s over jumps, doesn't brake on the way down, steers gently on a jump, and rolls back for a second run if it stalls on a face. Slow laps finished: 60 of 60 at training-range physics, 54 of 60 on validation and 58 of 60 on practice loops at gate range (was 51 of 60).
  - What still fails: doubles whose two rollers sit close together make a pit. At low grip (about 0.58) and extra weight the car noses into it, stops, and can't climb out: a 31° face needs a grip of 0.6 from a standstill.
  - Going faster made it worse (4.5 m/s: 31 of 60; 5.5 m/s: 26 of 60): the car lands in the valley and slides.
  - Open: each failed out-lap is a failed gate run, so about 1 run in 10 is lost before the timed lap starts.

### E3 — Copy the teacher on loops, with and without the map

- **Built for it (2026-11-17):**
  - `sdc/phase46/maps.py`: one out-lap per run (the stand-in drives, the car records), turned by your dead reckoning into a map: the track's curvature every 0.25 m along the car's own odometer, plus the odometer reading at each kicker.
  - Env option `turn_map`: your dead reckoning runs during the timed lap as the car's odometer, and the policy gets 9 more inputs: the mapped curvature 0, 1, 2, 3, 4.5, 6, 8, 10 and 13 m ahead.
  - **Odometer error comes from the jumps** (from the E2 logs): of about 6–7 m gained per lap, 5 m is in the air and the half second after landing; flat ground and turns read about 1% long. Two fixes:
    - In your dead reckoning (agent's detail, flagged to the owner): in the air, distance uses the pitch at takeoff, not the current pitch. The car keeps the horizontal speed it left the lip with.
    - Re-sync at kickers: when the filter's pitch climbs through 12° on the ground, the odometer snaps to the nearest mapped kicker within 8 m.
  - **The teacher with the map** (plans corner speeds and steering feed-forward from the car's own map, not the true track; validation, gate range, 60 laps): 44 finished with the teacher's 3 m/s² braking, **51 with 2 m/s²** (brakes earlier, which forgives map error). Six runs have no map because their out-lap failed; 51 of the 54 mapped runs finished. With the true track the teacher finishes 56.
- **Setup:**
  - Student: your GRU and filter inputs, parts-list sensors plus the two angled range sensors. Arm A: sensors only, copying the true-track teacher. Arm B: plus the 9 map inputs, copying the map-following teacher (2 m/s² braking).
  - DAgger copying as in Phase 4.5, 8 rounds of 60k steps, on the 291 practice loops whose out-lap finished, training-range physics. No fine-tuning.
  - Scored on the validation loops at gate range, 3 starts (60 laps), each with its own out-lap map. A run whose out-lap failed races without a map.
  - Seeds 0–1. `bash sdc/phase46/e3_runs.sh`.
- **Prediction (owner, before the run):** "40" (laps finished of 60; one number, not split by arm).
- **Result (validation, gate range, 60 laps per seed):** [report](https://claude.ai/artifact/Ruxyg3iBiHdvR6LDKsFAUk).

  | Student | Laps finished, seed 0 / 1 | Off the lane | Rolled / stuck | Jumps landed |
  |---|---|---|---|---|
  | A: sensors only | **27 / 40** | 29 / 15 | 4 / 5 | 72% / 77% |
  | B: + out-lap map | **34 / 39** | 22 / 15 | 4 / 6 | 71% / 76% |
  | Map-following teacher | 51 | | | |
  | True-track teacher | 56 | | | |
  | Phase 4.5 car (E1) | 0 | | | |

  - Every finished lap was within the gate's time limit (about 30 s, the teacher's pace).
  - The map made no clear difference: 36.5 vs 33.5 laps on average, inside the 13-lap gap between A's two seeds.
  - B finishes none of the 6 runs whose out-lap failed (raced without a map); A finishes 2 and 4 of them.
  - Where runs end (not finished, seed 0 / 1): A in a turn 20 / 10, on a jump 7 / 7, on a straight 6 / 3; B in a turn 15 / 9, on a jump 3 / 4, on a straight 8 / 8.
  - Both arms were still improving at the last round (finished laps on the 20-lap check rose in the last rounds; imitation error still falling).
  - Sensors only does far better than Phase 4.5's car (0 of 60). What changed: turns in training and the two 45° sensors. The experiment doesn't separate the two.
  - Bug fixed after the run: the per-round check timed laps against the wrong teacher laps. It changed nothing (every finished lap was well inside the limit); the final scores were always matched correctly.
  - Prediction vs. result: 40 predicted; 27–40 measured (mean 35).
- **Explanation (owner, after the run):** "Sensors allow it to react to the actual situation. The map is only an estimate of where it is. It seems the better sensors are sufficient for it to respond."
- **Takeaway:** with sensors that show its heading to the banks, the car reacts well enough to turns without knowing them in advance; an approximate map adds little. Both arms were still improving.
- **Owner's decision (2026-11-17):** "Let's increase training time." (E4)

### E4 — Train longer

- **Setup:** E3 exactly, with 16 copying rounds of 60k steps instead of 8 (twice the data). Both arms (sensors only; + out-lap map), seeds 0–1. `bash sdc/phase46/e4_runs.sh`, about 2.2 h on 4 cores.
- **Prediction (owner, before the run):** "I think it helps everything only a little. 2-3 more completions" (each arm finishes 2–3 more of 60 laps than in E3; the gap between the arms barely changes).
- **Result:** [report](https://claude.ai/artifact/HpNLjh5sHMcUsAWw3t5tv3). Validation loops 10000–10019 × 3 starts, gate-range physics, finished laps of 60 (seed 0 / 1):

  | Arm | E3 (8 rounds) | E4 (16 rounds) | Change (mean) |
  |---|---|---|---|
  | A: sensors only | 27 / 40 | **46 / 39** | +9 |
  | B: + out-lap map | 34 / 39 | **41 / 40** | +4 |

  - Endings in E4 (seed 0 / 1): A off the lane 10 / 15, rolled over 3 / 5, stuck 1 / 1; B off the lane 12 / 14, rolled over 6 / 6, stuck 1 / 0.
  - Best run 46 / 60; the gate needs 48. Every finished lap was inside the time limit.
  - The gain is mostly A seed 0 (27 → 46); A seed 1 didn't move (40 → 39). The 20-lap practice check bounced in the last rounds (B: 17 → 13), so seed noise is still large.
  - The arms swapped order: sensors only is now ahead (42.5 vs 40.5), still inside seed noise.
  - Prediction vs. result: +2–3 per arm predicted; +9 (A) and +4 (B) measured.
- **Explanation (owner, after the run):** "Training longer helped. It's a bigger network and harder task, so that seems reasonable. The difference between seeds is a bit concerning. It seems it isn't really converging on a great way to drive."
- **Takeaway:** more copied data still pays (the gain beat the prediction), but two seeds of the same recipe land 7 laps apart, so the student hasn't settled on one reliable way to drive. Leaving the lane is still the main failure.
- **Owner's decision (2026-11-17):** a teacher that drives with margin (E5), and averaging the trained students in parallel (E6). Not chosen: training longer again ("the training curves appear to mostly saturate").

### E5 — Teacher with margin

- **Setup:** E4's sensors-only arm with a more cautious teacher: corner grip budget 2.5 m/s² instead of 3.5 (about 15% slower through turns) and braking 2.5 m/s² instead of 3.0. Everything else as E4 (16 rounds of 60k steps), seeds 0–2 (E4 had 0–1). Laps still timed against the standard true-state teacher. `bash sdc/phase46/e5_runs.sh`, about 2.2 h. (A container restart killed the first launch at round 3; restarted from scratch 2026-11-18.)
- **The cautious teacher itself (gate range, validation):** 55 / 60 laps, median 32.5 s (standard teacher: 56 / 60, 30.1 s; limit 1.5× = about 45 s).
- **Prediction (owner, before the run):** "47" (laps finished of 60).
- **Result:** [report](https://claude.ai/artifact/2voQWyTwR2z6giBGy2NMEW). Laps finished of 60, validation loops, gate-range physics:

  | Teacher | Seed 0 | Seed 1 | Seed 2 | Mean |
  |---|---|---|---|---|
  | Standard (E4, sensors only) | 46 | 39 | – | 42.5 |
  | **With margin (E5)** | **43** | **47** | **49** | **46.3** |

  - Endings (seed 0 / 1 / 2): off the lane 11 / 8 / 8, rolled over 3 / 3 / 0, stuck 2 / 2 / 3, time limit 1 / 0 / 0.
  - Median lap about 33 s (E4: 30.4 s); every finished lap inside the limit (about 45 s).
  - Seed 2 reaches 49 / 60 on validation, above the gate's 48; the gate is judged on the test loops.
  - Seeds still spread 6 laps (43–49); the 20-lap check per round was still noisy at the end.
  - Prediction vs. result: 47 predicted; 43–49 measured (mean 46.3).
- **Explanation (owner, after the run):** "It seems cautious is good. Better to complete than crash"
- **Takeaway:** a student copies its teacher imperfectly, so a teacher at the grip limit leaves no room for the student's errors. Copying a teacher with margin trades about 2.5 s of lap time for fewer crashes, and the time limit has plenty of room for it.

### E6 — Average the students

- **Setup:** no training. Two cars driven by the average of two E4 students' throttle and steering, each student keeping its own memory: sensors-only seeds 0 + 1, and map seeds 0 + 1. Same 60 validation runs as E4. `python -m sdc.phase46.ensemble`, about 20 min each, run beside E5.
- **Prediction (owner, before the run):** "No, I don't think driving strategies can be averaged at a step by step basis. But maybe if it isn't planning ahead much or if the styles don't conflict" (the average won't beat the better student).
- **Result:** [report](https://claude.ai/artifact/GafL33A8bLyYhYvaZ4iRwp). Laps finished of 60, validation loops, gate-range physics:

  | Arm | Seed 0 alone | Seed 1 alone | Average of the two |
  |---|---|---|---|
  | A: sensors only | 46 | 39 | **46** |
  | B: + out-lap map | 41 | 40 | **45** |

  - Run by run, sensors only: the average finished 1 of the 7 runs both students failed and 18 of 19 runs only one student finished, but failed 5 of 34 runs both students finished. Map arm: 5 of 13, 8 of 9, and failed 2 of 38.
  - So averaging rescues most runs where the students disagree, but sometimes breaks a run both students handled: two good lines averaged can be a bad one.
  - Median lap 30.2–30.5 s, the same as the students alone.
  - Prediction vs. result: "won't beat the better student" predicted; it tied the better student in A (46 vs 46) and beat it in B (45 vs 41).
- **Explanation (owner, after the run):** "It seems to reduce the impact of bad decisions, especially when one of the two made a better decision."
- **Takeaway:** the two students' mistakes are mostly in different places, so averaging their throttle and steering cancels many of them; it costs a few runs where both were right in different ways. A cheap way to reduce the seed-to-seed variation.
- **Owner's decision (2026-11-18):** E7, average the three E5 students on validation; then the gate with the better of that average and E5 seed 2 alone.

### E7 — Average the cautious students

- **Setup:** no training. One car driven by the mean throttle and steering of E5 seeds 0, 1 and 2, each with its own memory. Same 60 validation runs. `python -m sdc.phase46.ensemble --runs runs/p46-e5-s0 runs/p46-e5-s1 runs/p46-e5-s2 --out runs/p46-e7.json --procs 4`.
- **Prediction (owner, before the run):** "I think we could do 49 or 50" (laps finished of 60).
- **Result:** [report](https://claude.ai/artifact/3HFiNiA65FFQkL7Ra4vkRm). **50 / 60** laps finished (students alone: 43, 47, 49). Endings: 7 off the lane, 3 rolled over. Median lap 32.8 s.
  - Run by run (students that finished alone → average finished): all 3 → 29 of 33; 2 → 15 of 16; 1 → 4 of 8; none → 2 of 3.
  - It mostly follows the majority, and still breaks 4 runs every student handled.
  - Prediction vs. result: 49–50 predicted; 50 measured.
- **Explanation (owner, after the run):** "They must balance each other."
- **Takeaway:** students copying one deterministic teacher make mostly different, idiosyncratic mistakes (variance, not different driving styles), so the majority usually outvotes a mistake. The few runs averaging breaks may be places where the students' good lines differ.
- **Owner's question:** is averaging common practice, is there a standard alternative, or does it point to a design issue such as multiple success modes? (Answered in chat: ensembling is standard; alternatives are distilling the ensemble into one network or averaging one network's weights over training; the evidence points to variance rather than multiple modes.)

## Gate check

- **Run (2026-11-18):** `python -m sdc.phase46.gate`, then `python -m sdc.phase46.gate_report`. [Report](https://claude.ai/artifact/EooxqvSc7QgK56VHJy1M8g).
- **Car:** E7's average of the three E5 students (owner's decision: the better of that and E5 seed 2 on validation, 50 vs 49).
- **Leakage review:** test loops 46000–46019 and physics seed 46 appear only in the seed lists and in `tests/test_loop.py`, which builds loop 46001 to check the generator's geometry rules (no driving, no selection). Nothing was trained, tuned or chosen on them.
- **Compute:** three GRU networks of about 83k parameters each (actor and critic; the car only needs the actor), roughly 0.5 million multiply-adds per step at 50 steps a second. Small for any Raspberry Pi–class board.
- **Result:** **39 / 60 finished (needs 48). Gate failed.**
  - Timed lap only: 40 / 60. One out-lap stuck (loop 46012, start +0.3) on a run the car then finished.
  - Endings: 9 off the lane, 7 stuck, 4 rolled over. Every finished lap within the time limit (median 32.9 s).
  - All 7 stuck runs stop just past the lip of a double, short of the landing. This car had none stuck on validation.
  - The test loops were harder for the teacher too: 48 / 60 here against 56 on validation (same mix of jumps: 25 doubles, 20 drops, 15 tables vs 27 / 18 / 15). The car finished 36 of the teacher's 48 (75%; validation 89%) and 4 the teacher failed.
  - The test set wasn't used to dig further, so it stays fresh for a retry.
- **Verdict:** gate failed. Phase 4.6 stays in progress.

## Handoff to next phase
