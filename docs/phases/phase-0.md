# Phase 0 — Environment and loop

- **Status:** in progress
- **Goal / gate:** a reproducible Python env and a published training report, built on PPO on a toy task. **Gate:** a single command trains PPO to solve `CartPole-v1` (avg return ≥ 475) in under 10 min, and a report page with curve + GIF is viewable on your phone.
- **Learn:** agent/env loop, returns, policy gradient intuition, what a learning curve tells you.
- **Code:** `sdc/phase0/train.py` (training + held-out eval + GIF), `sdc/report.py` (report page), `sdc/phase0/gate.py` (gate), `tests/test_smoke.py`
- **Setup:** `uv venv .venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt` (CPU-only torch, ~1 min)
- **How to reproduce the gate:** `python -m sdc.phase0.gate`
- **Phase-entry decisions:** none deferred to Phase 0 in the roadmap.

### Terms used in this phase

- **Return:** the sum of rewards in one episode. CartPole gives +1 per step the pole stays up, capped at 500 steps, so return = how many steps it survived.
- **Held-out evaluation:** episodes reset with seeds ≥ 10,000, which the training envs never use. Uses the greedy (deterministic) policy. "Solved" = mean held-out return ≥ 475 over 100 episodes.
- **Training curve vs. eval curve:** training episodes use the stochastic, exploring policy, so the training curve sits below what the greedy policy can do.

## Experiment plan

Each experiment is predict → run → explain. Runs are short (1–5 min on 4 CPUs), so no checkpointing is needed in this phase.

| # | Question | Setup | Concept |
|---|---|---|---|
| E1 | What return does a random policy get, and how quickly does out-of-the-box PPO leave it behind? | Random policy (100 eps) vs. SB3-default PPO, 1 env, 100k steps, seed 0 | The agent/env loop, what "return" means, reading a learning curve |
| E2 | Is one run trustworthy? | E1's PPO, 5 seeds | RL variance; why a single curve can mislead |
| E3 | What does the learning rate do to a policy-gradient learner? | Default PPO, lr ∈ {3e-5, 3e-4, 3e-3, 3e-2}, 3 seeds each | Step size in policy gradient; too small = slow, too big = collapse |
| E4 | What does the discount factor γ change? | Default PPO, γ ∈ {0.5, 0.9, 0.99}, 3 seeds each | Discounted return and effective horizon (~1/(1−γ) steps) |
| E5 | Why do the published "tuned" settings solve it faster? | Default (1 env) vs. RL-Zoo tuned (8 envs, batch 256, 20 epochs, linear lr decay), 3 seeds, wall clock to first solved eval | Parallel envs, sample reuse, steps vs. seconds; feeds the gate |

Then the gate check.

## Experiments

### E1 — Random policy vs. out-of-the-box PPO

- **Prediction (owner, before the run):** Random policy averages about 110 steps. PPO reaches ≥ 475 within 100k steps, late in training, with a smooth curve.
- **Setup:** `python -m sdc.phase0.train --preset random --out runs/e1-random`; `python -m sdc.phase0.train --preset default --steps 100000 --seed 0 --out runs/e1-ppo --label "PPO default"`
- **Result:** _pending_
- **Explanation (owner, after the run):** (1) From the GIF it appears the pole counts as fallen once it reaches a few degrees of tilt, not fully horizontal. That's part of it, but mostly I knew nothing about the physics and guessed that random motion would average out to doing nothing, and that it probably takes about 1 second for a broom to fall. (2) It's more effective than I expected!
- **Takeaway:** Random pushes don't cancel out. Each one adds velocity, so the errors pile up like a random walk, and the episode ends at 12°. Even out-of-the-box PPO solves CartPole in about 25k steps (under 30 s on CPU).

### E2 — Is one run trustworthy? (seed variance)

- **Prediction (owner, before the run):** All 5 seeds solve it. They solve at different points, but since this seems fairly easy to solve, maybe not a lot of spread.
- **Setup:** `python -m sdc.phase0.train --preset default --steps 100000 --seed S --out runs/e2-s$S --label "PPO default"` for S = 0..4, run in parallel (1 torch thread each).
- **Result:** [report](https://claude.ai/artifact/Ko9EenCyMWQxSJLbFP5gSg). All 5 seeds: **500.0** held-out mean (5/5 solved), about 130 s each running 5 in parallel. First held-out eval ≥ 475: 15k steps (seeds 1, 3), 20k (seeds 0, 2, 4). 20-episode training average first ≥ 475 between 27.6k and 35.1k steps. Untrained greedy scores varied widely (416, 9, 9, 146, 9) but the runs converged within about 5k steps of each other. Seed 0 here solved at 20k vs. 25k in E1: E2 used 1 torch thread per job, and different thread counts change floating-point summation order, so the same seed is only bit-identical on the same setup.
- **Explanation (owner, after the run):** I suspect there isn't much variation in successful strategies, so all starting seeds eventually converge to the same place. But the order of the building blocks that get you there can change, and that order shapes each seed's curve. With a harder task, I expect more building blocks and more possible solutions, which means more variance.
- **Takeaway:** On an easy task with essentially one good strategy, seeds converge tightly despite very different starts. Expect a wider spread on harder tasks, so judge configs over several seeds. Runs now pin 1 torch thread so a seed reproduces exactly.

### E3 — What does the learning rate do?

- **Prediction (owner, before the run):** All but the biggest solve it. The smallest grows very slowly; the biggest is erratic.
- **Setup:** `python -m sdc.phase0.train --preset default --steps 100000 --seed S --set learning_rate=LR --out runs/e3-lrLR-sS --label "lr LR"` for LR ∈ {3e-5, 3e-4, 3e-3, 3e-2}, S ∈ {0, 1, 2}; 1 torch thread each, 4 in parallel.
- **Result:** [report](https://claude.ai/artifact/N1p8Mc4qhYFmJDRibGUKCF). Solved at the end (held-out mean ≥ 475): **3e-5: 1/3** (499, 455, 441; 20-episode training average peaked at 276–378); **3e-4: 3/3** (first solved eval 15k–20k); **3e-3: 3/3**, fastest (first solved eval 5k–15k); **3e-2: 0/3** (471, 9, 9). At 3e-2, seeds 1 and 2 reached about 500 on the 10k-step eval, then collapsed to 9–10 (always pushing one way) by 20k and never recovered; seed 0 swung between 20 and 425.
- **Explanation (owner, after the run):** (1) Once it overshoots, the weights are too big and it never even tries pushing the other direction. (2) Bigger is faster until you stop exploring.
- **Takeaway:** The learning rate trades speed against stability. Too small crawls. Too big can jump to a near-deterministic policy that stops exploring: every episode then looks the same (about 9 steps), so there is nothing left to learn from and the collapse is permanent. The default is a safe middle, not the optimum for every task.

### E3b — Does the policy need a neural network at all? (added at owner's request)

- **Prediction (owner, before the run):** _pending_
- **Setup:** default PPO with a linear actor (logistic-regression policy: 4 inputs → push-right probability, no hidden layers) and the default 64×64 critic: `--set policy_kwargs='{"net_arch": {"pi": [], "vf": [64, 64]}}'`, 3 seeds, 100k steps.
- **Result:** _pending_
- **Explanation (owner, after the run):** _pending_
- **Takeaway:** _pending_

## Gate check

- **Run:** _pending_
- **Result vs. threshold:** _pending_
- **Verdict:** _pending_

## Handoff to next phase

_Pending._
