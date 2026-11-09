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

## Gate check

## Handoff to next phase
