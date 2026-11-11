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

1. **E1 — Pick for speed, not just safety.** Score every checkpoint for both reliability and lap time; choose the fastest with ≥ 90% validation gate range instead of the most reliable. How much speed did early stopping throw away?
2. **E2 — Train longer.** 2–3M steps with the lap bonus: does it keep getting faster without the late reliability slide?
3. **E3 — Reward speed directly.** A per-step time cost or a lap-time bonus: the speed vs. reliability trade-off as a reward.
4. **E4 — Train with the grip estimate.** Fine-tune with the estimator in the loop, so the car learns to trust the estimate.
5. **E5 — Track memory (only if still short).** A grip map built on an out-lap; the out-lap rule gets decided first.

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

## Gate check

## Handoff to next phase
