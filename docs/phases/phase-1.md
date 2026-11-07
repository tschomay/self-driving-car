# Phase 1 — 2D top-down racer with ray sensors

- **Status:** in progress (started 2026-11-07)
- **Goal / gate:** our own minimal vectorized sim: kinematic bicycle-model physics with perfect traction, procedural closed tracks, 7–15 ray-cast "lidar" distances plus speed as the observation, and an MLP policy. **Gate:** the RL policy completes ≥ 95% of laps on 20 held-out tracks with mean lap time ≤ 1.1× the pure-pursuit baseline.
- **Learn:** reward shaping (progress vs. speed vs. penalties), reward hacking, episode design, PPO hyperparameters, overfitting to one track vs. generalizing.
- **Code:** `sdc/racer/` (sim, tracks, pure-pursuit baseline, lap evaluation, rendering, SB3 adapter), `sdc/phase1/train.py` (training + held-out lap eval + GIFs), `sdc/phase1/report.py` (report page), `tests/test_racer.py`
- **Setup:** same as Phase 0: `uv venv .venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`
- **How to reproduce the gate:** _to be added_
- **Phase-entry decisions:** none deferred to Phase 1 in the roadmap. Phase 0's previous gate: passed (2026-11-07).

### The racer (what the agent sees and controls)

- **Track:** a closed loop 6 m wide, about 160–380 m long, with corners as tight as 6 m radius. Each track is generated from a seed. Training uses seeds < 10,000. The **20 held-out tracks** are seeds 10,000–10,019 and are never trained on.
- **Car physics:** perfect grip (no sliding), but sideways acceleration in a turn is capped at 8 m/s² (about 0.8 g). Top speed is 25 m/s (90 km/h). It accelerates at 6 m/s² and brakes at 12 m/s². So a tight corner forces you to brake: at most about 7 m/s through a 6 m radius.
- **Control:** 10 decisions per second, each two numbers in [−1, 1]: steering (as a fraction of what's allowed at the current speed) and throttle (+ accelerate, − brake).
- **Observation (12 numbers):** 11 ray distances to the road edge, every 18° from right (−90°) to left (+90°), each up to 40 m, plus speed. The car doesn't see the track map, its position, or upcoming corners beyond its rays.
- **Episode (training):** start at a random point on a random training track, standing still. Ends when the car's center goes more than 2.1 m from the centerline (off the road), or after 100 s.
- **Reward (default, "progress"):** metres gained along the centerline per step ÷ 2.5. That's 1.0 per step at top speed, and negative when going backwards. Leaving the road gives no extra penalty by default; it just ends the stream of reward. Other modes, for experiments: `survival` (+1 per step on the road) and `center` (+1 per step, less the farther from the centerline).
- **Lap evaluation (the gate metric):** one attempt per held-out track, from the start line at a standstill, with the greedy (no-noise) policy. Each attempt ends finished (lap time), off the road, or timed out at 120 s.
- **Baseline: pure pursuit.** A hand-coded driver that reads the centerline and its curvature directly. It's privileged information the RL car doesn't get. It steers toward a point on the centerline 4 m + 0.5 s × speed ahead. It plans speed so it can brake down to each corner's limit (95% of the grip) in time. It finishes 20/20 held-out tracks, with a mean lap time of **20.3 s**. That's close to the ideal speed profile along the centerline. Gate threshold: ≤ 22.4 s mean lap time and ≥ 19/20 finished.

### Sim design notes

- First version: wobbly-circle tracks, 8 m wide, corners ≥ 10 m radius, 20 m/s top speed. A pilot run showed it was nearly flat-out driving with no braking needed. I tightened it before any experiment: spline tracks with 8–14 control points, 6 m wide, corners ≥ 6 m radius, 25 m/s top speed. With that, the cornering limit is below top speed on about two-thirds of the centerline.
- I used pilot runs to size the step budgets. Their results are held back until the owner has predicted.
- Speed: about 13k sim steps/s per core with 8–32 cars batched. SB3 PPO trains at about 3.5k steps/s on 1 thread.

## Experiment plan

Each experiment is predict → run → explain. Runs are 1–5 min on 1 thread, 4 in parallel, so no checkpointing yet.

| # | Question | Setup | Concept |
|---|---|---|---|
| E1 | How well does out-of-the-box PPO drive tracks it has never seen, against a hand-coded driver? | SB3 PPO (8 cars, 256 steps per update, else defaults), progress reward, fresh random training track every episode, 300k steps, 3 seeds; pure pursuit as the yardstick | MDP design (observation, action, reward, termination), reading lap metrics |
| E2 | What does the car do when the reward isn't exactly what you want? | Same as E1, reward = survival vs. center vs. progress (and progress with a crash penalty) | Reward hacking and reward shaping |
| E3 | Does training on one track teach driving, or memorize that track? | Train on 1 track vs. 10 tracks vs. a fresh track every episode; evaluate on its own tracks and on the held-out 20 | Overfitting vs. generalization |
| E4 | How far ahead must the agent look to brake for corners? | γ ∈ {0.9, 0.97, 0.99, 0.995} (horizon about 1, 3, 10, 20 s) | Discounting and braking horizon (follows Phase 0's E4) |
| E5 | Can your own PPO match the library's? | Owner-written PPO (learner-owned, scaffolded piece by piece) on the E1 setup, against the SB3 result | How PPO works inside: rollouts, advantages (GAE), clipped objective, value loss |

Then the gate check. Plan: run it with the owner's own PPO, and report SB3 alongside.

## Experiments

### E1 — Out-of-the-box PPO vs. a hand-coded driver on unseen tracks

- **Prediction (owner, before the run):** _pending_
- **Setup:** `python -m sdc.phase1.train --preset pursuit --out runs/p1-pursuit`; `python -m sdc.phase1.train --seed S --steps 300000 --out runs/p1-e1-sS --label "PPO base"` for S = 0, 1, 2 (1 torch thread each, in parallel).
- **Result:** _pending_
- **Explanation (owner, after the run):** _pending_
- **Takeaway:** _pending_

## Gate check

_pending_

## Handoff to next phase

_pending_
