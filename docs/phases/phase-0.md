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

- **Prediction (owner, before the run):** _pending_
- **Setup:** `python -m sdc.phase0.train --preset random --out runs/e1-random`; `python -m sdc.phase0.train --preset default --steps 100000 --seed 0 --out runs/e1-ppo --label "PPO default"`
- **Result:** _pending_
- **Explanation (owner, after the run):** _pending_
- **Takeaway:** _pending_

## Gate check

- **Run:** _pending_
- **Result vs. threshold:** _pending_
- **Verdict:** _pending_

## Handoff to next phase

_Pending._
