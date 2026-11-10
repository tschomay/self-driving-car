# Phase 3: why the RL car saws the steering (investigation, 2026-11-10)

A background investigation requested by the owner. Subject: E8 (l) seed 0 unless noted, evaluated on the validation tracks only (the test tracks were not touched). No repo files were changed. Scripts and outputs were kept in the session scratchpad.

**Short answer:** the deterministic policy really does saw, and nothing in training pushes it to stop, because sawing costs almost nothing. It isn't a lack of foresight: a smooth line can be learned from the existing 40 m rays. The slowness comes from the throttle (caution), not from the steering.

## 1. Sawing is nearly free (measured)

The policy's steering was low-pass filtered at evaluation (exponential smoothing, α = weight on the new command):

| α | Steering change/step | Lap | Slip > 4° | Gate range | Train range |
|---|---|---|---|---|---|
| 1.0 (as trained) | 0.458 | 35.35 s | 70% | 73% | 95% |
| 0.5 | 0.252 | 35.45 s | 64% | 66% | 96% |
| 0.25 | 0.115 | 35.06 s | 34% | 73% | 95% |
| 0.15 | 0.072 | 35.00 s | 13% | 51% | 79% |

- Down to α = 0.25, completion and lap time are unchanged and sliding halves. Heavy lag (α = 0.15) hurts low-grip laps.
- The servo and tire relaxation filter it: the wheel moves 0.33 per step while the command moves 0.46.
- The body barely weaves: mean |lateral| 0.53 m, 95th percentile 1.07 m.
- The steering is close to white noise: lag-1 autocorrelation 0.31 (hand-coded: 0.99), direction flips on 61% of steps.

## 2. Exploration noise never shrinks, and the policy learns to fight it (measured)

- The learned steering noise (std, pre-tanh) is still 0.53 at the end of training, and 0.4–0.66 in every Phase 3 run.
- The deterministic policy has strong negative feedback on fast-changing inputs. Sensitivity of the steering command to: last steering command −0.74, gyro −2.9, sideways acceleration −0.9.
- Feedback of −0.74 × the last command alternates sign every step: that is the sawing. It plausibly cancels the injected noise, which the policy sees through its last-command input.
- Causal check (one seed, 300k steps each):

  | 300k steps | Steering change | Sensitivity to last command | Slip > 4° | Held-out | Gate range | Lap |
  |---|---|---|---|---|---|---|
  | baseline | 0.431 | −0.45 | 72% | 100% | 98% | 42.9 s |
  | initial std 0.2 (`log_std_init=-1.6`) | **0.057** | +0.12 | 47% | 65% | 49% | 35.9 s |

  Starting with low noise removes the sawing (the std stays about 0.17), but exploration suffers: less reliable at 300k.
- A car blind to its last command (E3 c) still saws 0.28–0.45, so that input is one route to sawing, not the only one.

## 3. Sensor noise (owner's hypothesis): amplifies, doesn't cause (measured)

- Validation laps run at zero noise and still saw (0.458).
- Steering change grows with evaluation noise: noise 0 / 0.02 / 0.04 → 0.458 / 0.556 / 0.660; laps 35.4 / 35.7 / 36.5 s.
- E5 (f), trained with no noise at all, saws 0.350.

## 4. Foresight (owner's hypothesis): ruled out for steering (measured)

The hand-coded driver's steering was cloned into a 64×64 network (the policy's size) from the RL car's observation, minus the last commands, with no curvature input. It predicts the hand-coded steering with R² 0.988. In closed loop:
- cloned steering + hand-coded throttle: 19/20, 25.3 s, steering change 0.02
- cloned steering + the RL policy's own throttle: 20/20, 34.3 s (RL alone 35.35 s), gate range 74%, steering change 0.013

So the 40 m rays contain enough lookahead for a clean line. The 20–40% slowness is throttle caution: mean throttle 0.15, full throttle on 1.2% of steps.

## 5. Physics review issues: not the cause at nominal grip (measured)

At nominal grip, axles spin on 0.7% of steps and lock on 0.1%, and the car is below 3 m/s on 1.8%. These may matter at grip × mass < 0.72. Sawing is about the same across E3–E8: γ 0.99 or 0.997, squash on or off, realism 1 or 2 (0.26–0.64).

## Aside

The 300k-step baseline, with its learning rate decayed to 0 by 300k, scored 98% gate range (validation; one seed, 100 laps) at 42.9 s laps. E8 (l)'s training curves show the same pattern: 80–90% gate range around 200–400k steps, falling to 60–75% as laps get faster. Reliability is being traded for speed.

## Proposed experiments

1. **Lower or decaying exploration noise** (e.g. std 0.5 → 0.1 over training). The most direct fix, with no reward change.
2. A small steering-rate penalty (reward shaping; the owner prefers to avoid it).
3. A low-pass filter in the action path during training, or a higher control rate with a filter.
4. For speed: work on the throttle caution and the reliability-vs-speed trade-off (training length, horizon), not on lookahead.
