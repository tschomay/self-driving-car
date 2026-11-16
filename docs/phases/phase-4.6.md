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

## Gate check

## Handoff to next phase
