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
  - Decision pending (owner): who drives the out-lap.

## Gate check

## Handoff to next phase
