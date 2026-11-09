# Phase 3 — Realistic dynamics and robustness

- **Status:** in progress (started 2026-11-09)
- **Goal / gate:** dynamic bicycle model with tire slip, throttle/brake, low-grip "dirt" patches, actuation latency, sensor noise, and **domain randomization** of friction, mass, and latency. **Gate:** on held-out tracks, with physics parameters drawn from a range *wider* than training, completion is ≥ 85% and the robustness curve is published. **Go/no-go:** after this phase, decide whether to go 3D (cost, interest, time).
- **Learn:** vehicle dynamics, why perfect-traction policies fail, domain randomization as sim-to-real insurance, robustness evaluation; drifting may emerge on its own.
- **Code:** `sdc/racer/dynamics.py` (dynamic car, dirt, delay, noise, physics ranges, hand-coded driver, Phase 1 policy adapter), `sdc/phase3/robust.py` (robustness sweeps, gate laps), `sdc/phase3/train.py` (SB3 PPO on the dynamic car), `tests/test_dynamics.py`
- **How to reproduce the gate:** to be written (`python -m sdc.phase3.gate`)
- **Previous gate:** Phase 2 passed (2026-11-09).

## Phase-entry decisions (owner, 2026-11-09)

The roadmap defers no decision to Phase 3 entry; these came from Phase 2's handoff and the phase setup. The owner accepted all four recommendations.

1. **Rays, not pixels.** Phase 3's experiments and gate run on the ray observation: runs take minutes, and the lessons stay about dynamics. Cloning into the owner's CNN at the end is optional.
2. **SB3 for experiments, the owner's PPO for the gate**, as in Phase 1.
3. **Physics ranges:**

   | | Train range | Gate range (wider) |
   |---|---|---|
   | Tire grip | 0.7–1.1× | 0.5–1.3× |
   | Mass | ±20% | ±35% |
   | Control delay | 0–60 ms | 0–120 ms |
   | Sensor noise | ≤ 2% | ≤ 4% |

   Dirt patches on every track. Gate: ≥ 85% lap completion on held-out tracks 10000–10019 with physics drawn from the gate range (5 draws per track, 100 laps), plus a published robustness curve (completion vs. grip, delay, mass, noise).
4. **Compute: this container's CPU, $0.** Modal only if a run needs more than about 2 h.

## The dynamic car (what changed since Phase 1)

Same tracks, rays, progress reward, episode rules and lap evaluation as Phase 1. What's new:

- **Tires that slip.** Each axle's tires push sideways only when they slip: the slip angle is the angle between where a wheel points and where it's actually moving. The force grows with slip, peaks at about 6–9°, then drops to about 70% as the tire slides. The peak is grip × the load on that axle. Nominal grip is 0.85 g, matching Phase 1's 8 m/s² cornering limit.
- **Friction circle.** Throttle and brakes use the same grip as cornering. A tire that's braking hard has little left for turning. Full brake (12 m/s² of force) is more than the tires can hold, so the wheels lock and can't steer. Full throttle on low grip spins the wheels.
- **Steering is now a wheel angle.** In Phase 1, steering was "a fraction of what perfect grip allows at this speed", so the car could never overdo it. Now steer = wheel angle up to 30° at any speed. Turning hard at speed slides or spins the car.
- **Drive:** all-wheel drive. Power-limited (100 kW): full push up to about 14 m/s, weaker above. No reverse. Below about 3 m/s the model blends into Phase 1's no-slip model (slip is ill-defined at walking pace).
- **Dirt patches:** 2–4 stretches of 10–30 m per track at 65% grip, fixed per track (drawn brown in GIFs). The car can't see them; it can only feel them.
- **Delay:** the wheels get each command *delay* seconds after the policy chooses it (simulated in 10 ms substeps).
- **Sensor noise:** random noise added every step to the rays, speed and gyro (std as a fraction of each sensor's full scale).
- **Observation (15 numbers):** 11 rays, forward speed, the last steering and throttle commands (because with delay, the commands still in flight are part of the state), and yaw rate from a gyro (how fast the car is rotating). It isn't told the grip, mass, delay or where the dirt is.
- **Hand-coded driver (yardstick):** pure pursuit as before, but it now *knows* the track, this episode's grip and mass, and where the dirt is. It slows its corner speeds and caps its throttle and brake to stay inside the friction circle. It doesn't know the delay or noise. It finishes 19/20 held-out tracks at nominal physics (one dirt corner beats it), mean lap 25.3 s.
- **Speed:** about 3.1k sim steps/s per core (10 substeps per step).

### Sim design notes

- First version: rear-biased drive (30/70), no power limit, brakes biased to the front. Everything spun or locked: full throttle asked the rear tires for more force than they hold, so they had nothing left for cornering. The hand-coded driver crashed every track at every grip. I changed it to all-wheel drive with a power limit and an even brake split. The friction-circle lesson stays: full brake still locks the wheels.
- Dirt at 50% grip made even the all-knowing hand-coded driver crash everywhere, so it went up to 65%.
- Pilot runs sized the step budgets. Their results are held back until the owner predicts.

## Experiment plan

Each experiment is predict → run → explain. SB3 PPO, Phase 1 settings, 4 runs in parallel on 1 thread each.

| # | Question | Setup | Concept |
|---|---|---|---|
| E1 | Does a perfect-grip driver survive real tires? | The Phase 1 ray policy, unchanged, in the dynamic car at nominal grip. Then PPO retrained in the dynamic car. Hand-coded driver as the yardstick. | Vehicle dynamics: slip, the friction circle, why perfect-traction policies fail |
| E2 | Trained at one grip, how does it do at others? | PPO trained at nominal physics vs. trained with grip randomized 0.7–1.1×; robustness curve over grip 0.4–1.3× | Domain randomization; overfitting to the physics |
| E3 | What does control delay do, and does knowing your last command help? | Train with 0 delay vs. 0–60 ms delay, with and without the last-command inputs; test at 0–150 ms | Latency, partial observability, the Markov property |
| E4 | What does full randomization cost and buy? | Nominal vs. full train-range randomization (grip, mass, delay, noise); all four robustness curves, plus nominal lap time | The robustness vs. performance trade-off; sensor noise |
| E5 | Can your own PPO drive the randomized car? | The owner's PPO (Phase 1) with the 15-number observation and E4's randomization: the gate candidate | Reusing the owner's PPO under harder dynamics |

Then the gate check, and the go/no-go on Phase 4.

## Experiments

### E1 — Does a perfect-grip driver survive real tires?

- **Prediction (owner, before the run):** "It will fail all tracks. Too aggressive so it will slide off the track or spin"
- **Setup:** (a) the Phase 1 SB3 ray policy (Phase 1 E1 seed 1, retrained here: 100% held-out on the Phase 1 car, 0.94× pure pursuit), unchanged, through `dynamics.phase1_policy` (rays + speed in, steering converted to a wheel angle the Phase 1 way). (b) SB3 PPO retrained in the dynamic car: Phase 1 settings (8 cars, 256 steps per update), nominal physics, dirt on, 15-number observation, 500k steps, seeds 0–2, about 6 min each on 1 thread. Yardstick: the hand-coded driver (knows track, grip, dirt). Evaluation: held-out tracks 10000–10019 at nominal physics with dirt, plus 100 gate-range laps and robustness sweeps. Pilot runs (1M steps, held back from the owner) sized the budget: the held-out curve peaked around 200–400k.
- **Result:** [report](https://claude.ai/artifact/4H6sjxrJgKhT2MEz9GgdgN)

  | | Held-out, nominal | Gate range (100 laps) | Lap time vs. hand-coded |
  |---|---|---|---|
  | hand-coded (knows grip) | 19/20 | 94% | 1.00× (25.3 s) |
  | Phase 1 policy, unchanged | **0/20** | 0% | – |
  | PPO retrained, seeds 0/1/2 | 85% / 80% / 75% | 35% / 45% / 44% | 1.13× / 1.13× / 1.17× |

  - Phase 1 policy: 0/20 at every point of every sweep, even at 1.3× grip. 18 attempts leave the road 12–50% into the lap; 2 spin, stop and time out. Mechanism (checked step by step): full steering lock plus full brake into corners. Full brake exceeds what the tires hold, so the wheels lock and can't steer; the fronts slide 20–60° past their 6–9° peak and the car ploughs straight off.
  - Retrained: the held-out curve swings 50–100% between checks every 50k steps (best checkpoints 95–100%). Tracks 10003 and 10005 fail in all 3 seeds. Below 0.8× grip it collapses.
- **Explanation (owner, after the run):** "Even with extra grip, the grip budget is shared, full brake can lock the wheels, and sliding wheels have less friction."
- **Takeaway:** a policy trained where grip is free learns habits (full lock plus full brake) that break the moment grip becomes a shared budget. Retraining in the new car fixes nominal driving (75–85%), but not other grips.
- **Owner's request (2026-11-09):** GIFs now show steering and pedal as gauges under the map instead of numbers, and leave skid marks that get darker and thicker the more an axle slides (from 4°, full at 20°). The E1 report was republished with the new GIFs.

### E2 — Trained at one grip, how does it do at others?

- **Prediction (owner, before the run):** "It might struggle with learning to drive. The rules based on what it measures aren't fixed anymore. It has to learn to react to the situation and determine what impact its reaction has. I think it can learn with enough training but it will cost some speed"
- **Setup:** E1's retrained runs (grip fixed at 1.0×) vs. new runs with grip drawn uniformly from 0.7–1.1× each episode (mass, delay, noise nominal). Same settings as E1: SB3 PPO, 500k steps, seeds 0–2, about 5.5 min each. Robustness curve: held-out tracks with dirt at grip 0.4–1.3×, the other knobs nominal.
- **Result:** [report](https://claude.ai/artifact/6GkRXA1c65jnERAaZyH7P7)

  | Grip | 0.6× | 0.7× | 0.8× | 0.9× | 1.0× | 1.1× | 1.3× |
  |---|---|---|---|---|---|---|---|
  | fixed 1.0× (3-seed mean) | 2% | 22% | 45% | 70% | 80% | 87% | 85% |
  | randomized 0.7–1.1× | 7% | 32% | 63% | 72% | 78% | 82% | 82% |

  - Both 0% at 0.4–0.5×. Train-range laps (100): fixed 51–62%, randomized 60–61%. Gate range: 35–45% vs. 42–45%.
  - Cost: about 4% slower at nominal (29.7 s vs. 28.0 s mean lap; 1.18× vs. 1.14× the hand-coded driver). Same nominal completion.
  - Learning speed no slower (at 150k steps: 45–90% vs. 30–75%).
  - Side effect: more robust to heavy mass (1.35×: 95% vs. 82%), though mass wasn't randomized.
- **Explanation (owner, after the run):** "It doesn't know the situation until it's too late to react appropriately"
- **Takeaway:** randomization only teaches caution as far as the car can sense the condition. Grip is felt only once the tires slide, so at low grip that's already too late in a corner. It buys a little robustness for about 4% speed.

### E3 — What does control delay do, and does knowing your last command help?

- **Prediction (owner, before the run):** "The delay risks that it won't be able to respond in time. But it also forces it to learn how to plan ahead. I think it could help. And seeing the last command might teach it how the car responds. I'm not confident, but I think b might help while c would just react late"
- **Setup:** (a) E1's runs (trained without delay, 15 inputs). (b) Delay drawn 0–60 ms per episode, 15 inputs. (c) Same delay, 13 inputs: last steering and pedal commands removed (`obs_keep`). Nominal grip and mass, dirt on, SB3 PPO, 500k steps, seeds 0–2; 6 runs on 4 CPUs, about 9.5 min each. Test: held-out tracks at 0–150 ms delay.
- **Result:** [report](https://claude.ai/artifact/KLGDbAQtYAzaV9XGnyc8h1)

  | Delay | 0 | 40 ms | 60 ms | 80 ms | 100 ms | 120 ms | 150 ms |
  |---|---|---|---|---|---|---|---|
  | (a) no delay in training | 80% | 83% | 78% | 67% | 67% | 58% | 35% |
  | (b) delay, sees last command | 90% | 85% | 95% | 92% | 77% | 70% | 47% |
  | (c) delay, blind to last command | 77% | 80% | 78% | 75% | 72% | 62% | 58% |

  - (b) also did better at physics it never trained on: grip 1.0× 90% vs. 80%, mass 0.65× 65% vs. 47%, mass 1.35× 98% vs. 82%. Cost: about 4% lap time.
  - Uncertainty: about ±10 points per seed, about ±5 for a 3-seed mean.
  - Recurring: held-out completion often peaks mid-training and falls by the end (e.g. (c) seed 0: 95% → 65%).
  - Gate range: (a) 35–45%, (b) 44–49%, (c) 41–50%.
- **Explanation (owner, after the run):** _pending_
- **Takeaway:** _pending_

## Gate check

## Handoff to next phase
