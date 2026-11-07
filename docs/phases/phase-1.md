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

Then the gate check.

**Decision (owner, 2026-11-07, after E1):** the Phase 1 gate counts only for the owner's own PPO. SB3 already meets the gate numbers (E1), so the sim and gate stay unchanged, and E2–E4 run quickly on SB3. E5 (own PPO) is the real hurdle. The gate report shows SB3 alongside for reference.

## Experiments

### E1 — Out-of-the-box PPO vs. a hand-coded driver on unseen tracks

- **Prediction (owner, before the run):** (1) Most of them, maybe 18. (2) Slower. It's not optimizing for speed, just completion.
- **Setup:** `python -m sdc.phase1.train --preset pursuit --out runs/p1-pursuit`; `python -m sdc.phase1.train --seed S --steps 300000 --out runs/p1-e1-sS --label "PPO base"` for S = 0, 1, 2 (1 torch thread each, in parallel).
- **Result:** [report](https://claude.ai/artifact/1T3tKT67hUY1p9NWprraNA). Held-out completion: **95%, 100%, 100%** (seeds 0, 1, 2). Mean lap time: **19.3, 19.1, 19.4 s**, i.e. **0.96×, 0.94×, 0.95×** pure pursuit (20.3 s). **All 3 seeds pass the gate numbers**, about 85 s of training each. First 100% held-out eval: 75k steps (seed 0), 50k (seeds 1, 2). Seed 0 was at 100% at 300k steps and dropped to 95% three updates later: it left the road on track 10000, 41% into the lap. Why it's faster than pure pursuit: it drives a racing line. In corners it sits 1.2 m inside the centerline on average, against pure pursuit's 0.7 m. Mean speed 14.2 m/s against 13.5. It's at full throttle 35% of the time and braking 35%. The training policy (with noise) still left the road in 8–10% of episodes at the end.
- **Owner's question (asked with the prediction):** why aren't the current throttle, brake and steering part of the input? Answer: in this sim they carry no information about the future. Steering and throttle are set fresh each step and have no lag or rate limit. The car's whole state is its position and heading (seen through the rays) plus speed (observed). The last action would matter if actuators lagged or were rate-limited, which is Phase 3 (actuation latency). Then it becomes part of the state and goes in the observation.
- **Explanation (owner, after the run):** Oh, so it is optimizing for speed.
- **Takeaway:** The reward defines the goal, including the parts you didn't spell out. "Metres per step" already means "go fast," so PPO found a racing line on its own and beat a map-reading driver by about 5%. With rays plus speed and perfect grip, library PPO meets the Phase 1 gate numbers in under 2 minutes of CPU. Learned policies can also lose a track between two nearby updates.

### E2 — Reward hacking: survival, center, crash penalty

- **Prediction (owner, before the run):** Survival tips the scale towards safe driving over fast driving, so I expect it to be a little slower. Center forces middle of road rather than the fastest line, so again slower. Crashes are already penalized because they stop progress; adding the crash penalty might make learning faster.
- **Setup:** E1 setup with `--env crash_penalty=10`, `--env reward='"survival"'` or `--env reward='"center"'`, seeds 0–2, 300k steps each, 4 in parallel on 1 thread each. The progress arm reuses the E1 runs.
- **Result:** [report](https://claude.ai/artifact/DniTgbXm6p2EWCJw2iRvqR). **Survival: 0/20 laps (all 3 seeds). Center: 0/20 laps (all 3 seeds).** The car learned to park: mean speed at the end of training 0.01–0.12 m/s, almost every training episode lasted the full 1,000 steps, and all 120 held-out attempts timed out before reaching 1% of the lap. Standing still earns the maximum return under survival (+1 × 1,000) and nearly the maximum under center. **Crash penalty: no faster.** First ≥ 95% held-out eval at 50k, 50k, 75k steps (plain progress: 75k, 50k, 50k). Early on it was more cautious: mean training speed around 50k steps was 8.5, 7.3, 4.0 m/s, against 9.4, 8.8, 9.2 for plain progress. It caught up by 150k steps (about 14 m/s). Final 95%, 100%, 100% at 0.95–0.96× pure pursuit, the same as plain progress.
- **Explanation (owner, after the run):** Nothing in those two rewards progress. And for the crash penalty: once the car learns to drive well, it isn't going to crash anyway, so it reaches the same end result.
- **Takeaway:** An agent optimizes exactly what you reward, not what you meant. "Don't crash" and "stay centered" both have a trivial optimum: don't move. Penalties shape the early, clumsy phase of learning (here: more caution, slower to speed up), but once the policy rarely triggers them they stop mattering. Reward what you actually want (progress), and add penalties only for behavior the main reward fails to discourage.

### E3 — One track vs. many

- **Prediction (owner, before the run):** One track might not be enough variation (e.g., it might never have a right turn), so that might not work. 10 tracks will probably do well enough, but not as good as E1.
- **Setup:** E1 setup with `--env tracks='[0]'` (1 track) or `--env tracks='[0,1,...,9]'` (10 tracks), seeds 0–2, 300k steps. The fresh-track arm reuses the E1 runs. Each policy also gets one lap per own training track.
- **Result:** [report](https://claude.ai/artifact/BXZQg1M1HpqMaD4wHL6aRg).
  - **1 track:** 100% on its own track (all seeds) and the fastest there: 20.3–20.5 s, against 21.0–21.2 s for the E1 policies and 22.3 s for pure pursuit. **Held-out: 55%, 65%, 70%.**
  - Track 0 runs clockwise (mostly right-hand corners; left-hand bends on only 4% of it, none tighter than 17 m). On the 9 counter-clockwise held-out tracks, 16 of 27 attempts failed; on the 11 clockwise ones, 6 of 33.
  - Overfitting over time: seeds 1 and 2 hit 100% held-out at 75k steps, then slid to 65–70% by 300k while staying at 100% on track 0.
  - Some failures were timeouts: on unfamiliar corners the car crawled or stopped.
  - **10 tracks:** 100% held-out (all seeds) at 0.97× pure pursuit (E1: 0.94–0.96×). On its own 10 tracks it averaged 21.1–21.3 s, slower than the E1 policies on those same tracks (20.6–20.9 s).
- **Explanation (owner, after the run):** Over fitting. And not enough exposure to new situations to learn new tricks.
- **Takeaway:** The policy learns the situations it's trained on, nothing more. One track gives a specialist: fastest at home, lost on mirror-image corners, and getting worse elsewhere the longer it trains (overfitting). Variety in training is what makes a driver generalize, and more variety (fresh tracks) beat 10 fixed tracks even on those 10. Judge generalization on held-out tracks, never on the training ones.

### E4 — Planning horizon (γ)

- **Prediction (owner, before the run):** .9 will fly off the track. .97 will finish some, maybe enough? The others will be fine.
- **Setup:** E1 setup with `--set gamma=G`, G ∈ {0.9, 0.97, 0.995}, seeds 0–2, 300k steps. The γ 0.99 arm reuses E1.
- **Result:** [report](https://claude.ai/artifact/5VnQn7MxxP9NV5DMurPZpR).
  - **γ 0.9: 90%, 90%, 80% held-out (gate fail).** All 8 failures were crashes at 7–16 m/s in 6–8 m corners: it braked too late.
  - **Training crash rate at the end: 64–93% of episodes for γ 0.9** (6–15% for the others). The noise-free policy mostly survives, but it has no safety margin: a crash more than about 1 s away barely counts.
  - **γ 0.97: 100%, 100%, 95%. γ 0.99: 95%, 100%, 100%. γ 0.995: 100%, 95%, 95%.** All pass, at 0.94–0.96× pure pursuit.
  - γ 0.995 drove slightly slower (13.8–14.0 m/s mean held-out speed vs. 14.0–14.3) and was no better.
  - Held-out track 10000 (6.9 m corner at 40% of the lap) caused every failure at γ ≥ 0.97 (4 of 4) and 3 of γ 0.9's 8.
- **Explanation (owner, after the run):** _pending_
- **Takeaway:** _pending_

## Gate check

_pending_

## Handoff to next phase

_pending_
