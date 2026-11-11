# Phase 3 — Realistic dynamics and robustness

- **Status:** in progress, gate failed (70% vs. 85%, 2026-11-10); started 2026-11-09
- **Goal / gate:** dynamic bicycle model with tire slip, throttle/brake, low-grip "dirt" patches, actuation latency, sensor noise, and **domain randomization** of friction, mass, and latency. **Gate:** on held-out tracks, with physics parameters drawn from a range *wider* than training, completion is ≥ 85% and the robustness curve is published. **Go/no-go:** after this phase, decide whether to go 3D (cost, interest, time).
- **Learn:** vehicle dynamics, why perfect-traction policies fail, domain randomization as sim-to-real insurance, robustness evaluation; drifting may emerge on its own.
- **Code:** `sdc/racer/dynamics.py` (dynamic car, dirt, delay, noise, physics ranges, hand-coded driver, Phase 1 policy adapter), `sdc/phase3/robust.py` (robustness sweeps, gate laps), `sdc/phase3/train.py` (SB3 PPO on the dynamic car), `tests/test_dynamics.py`
- **How to reproduce the gate:** `python -m sdc.phase3.gate` (trains the owner's PPO, seed 0, 1M steps, then evaluates once on the test set; about 15 min)
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
| E4 | Does the car drive smoothly once the physics stops rewarding twitchiness? (added after E3, from the owner's observation) | E3's recipe vs. learning fixes (squashed actions, decaying learning rate) vs. a more realistic car (servo, sliding wheels, tire relaxation) vs. both | Sim fidelity vs. learning fixes; smooth driving and the friction peak |
| E5 | Can it drive smoothly once it can feel the slide? (added after E4, from the owner's explanation) | E4 (d) vs. + sideways velocity vs. + accelerometer vs. both | What the car can sense limits what it can learn |
| E6 | What does full randomization cost and buy? | Nominal vs. full train-range randomization (grip, mass, delay, noise); all four robustness curves, plus nominal lap time | The robustness vs. performance trade-off; sensor noise |
| E7 | Can the car estimate grip from a short memory? (added after E6; owner chose option A of three: teach it to estimate grip / change the gate ranges / run the gate as is) | E6 (h) vs. + the last 1 s of own-motion readings (history), at 500k and 1M steps | System identification: adapting to conditions instead of being cautious about them |
| E8 | Does making crashes costly make it finish more? (added after E7; owner chose both fixes, memory dropped) | E6 recipe at 1M steps vs. crash penalty 100 vs. horizon γ = 0.997 vs. both | Aligning the reward with the goal (finished laps) |
| E9 | Can your own PPO drive the randomized car? | The owner's PPO (Phase 1) with E8's winning recipe and E6's randomization: the gate candidate | Reusing the owner's PPO under harder dynamics |

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
- **Explanation (owner, after the run):** "It needs that feedback loop - here's what I was doing to get me here and how the car responded"
- **Takeaway:** with delay, the commands still in flight are part of the state. A car that can see them (b) learns to drive through delay, and even drives better without it. A car that can't (c) gains nothing from practising with delay.
- **Owner's observation (2026-11-09):** the GIFs show twitchy driving: lots of skids, steering flipping between full left and full right. Measured on held-out laps (seed 0): mean steering change per step 0.26 (a) and 0.64 (b) vs. 0.016 for the hand-coded driver; at full lock 51% / 37% of the time; sliding more than 4° on 72% / 79% of steps vs. 11%. The owner asked for a decaying learning rate (adopted) and ideas for smoother driving.

### E4 — Does the car drive smoothly once the physics stops rewarding twitchiness?

- **Why:** the owner saw twitchy driving in E3's GIFs (skids, steering flipping full left to full right) and asked for smoother driving without a reward penalty: "it can drive however it wants", but smooth should be faster and use traction better. The owner suspected the sim or the learning. Gaps found:
  - Sim: locked or spinning wheels kept full front-to-back grip; steering was instant; tire force was instant.
  - Learning: actions clipped at ±1 (switch-like); constant learning rate.
  - Fixes, as `realism=2` (`sdc/racer/dynamics.py`, the default from now on; E1–E3 ran at `realism=1`):
    - steering servo, 0.4 s lock to lock (the owner noted a loaded servo is slower than the 0.2 s unloaded spec)
    - a locked or spinning wheel slides at 75% of peak grip with none left for turning
    - tire relaxation length 0.5 m
  - Learning fixes: tanh-squashed actions (`squash`) and a learning rate decaying linearly to 0 (owner asked for this).
- **Prediction (owner, before the run):** "d will be smoothest and fastest, but there's a confounder that the front/back traction change means it's actually harder to drive laps, so it might measure slower"
- **Setup:** all with E3 (b)'s recipe: delay 0–60 ms, 15 inputs, nominal grip, dirt. (a) E3 (b) itself. (b) learning fixes, old car. (c) new car, old learning. (d) both. 500k steps, seeds 0–2; 9 runs on 4 CPUs, about 14 min each. Smoothness measured over the 20 held-out laps (`train.motion`).
- **Result:** [report](https://claude.ai/artifact/JBm7NGR7owAYGesuVz5R1n)

  | 3-seed mean | Steering change/step | Full lock | Slip > 8° | Slip > 20° | Mean lap | Gate range |
  |---|---|---|---|---|---|---|
  | hand-coded | 0.02 | 0% | 1% | 0% | 25.3 s | 94% |
  | (a) E3 recipe | 0.44 | 30% | 61% | 30% | 29.2 s | 44–49% |
  | (b) learning fixes | 0.41 | 10% | 50% | 21% | 29.7 s | 40–48% |
  | (c) new car | 0.33 | 17% | 39% | 7% | 29.9 s | 44–51% |
  | (d) both | 0.34 | 7% | 45% | 7% | 32.0 s | 51–61% |

  - Not faster: the hand-coded driver takes 25.3 s in both cars, so the new physics doesn't slow a smooth driver. RL cars average 8–9 m/s vs. 11, with full throttle on 2–6% of steps vs. 21%: they drive slowly and still slide, from steering.
  - (d) is the most robust: delay 87–93% up to 120 ms, 72% at 150 ms (others 45–50%); best gate range.
  - The new car fixed most of the training wobble: (c) and (d) stay at 85–100% held-out from 150k steps on.
  - Agent's read: the car can't sense sideways sliding (gyro only, no sideways accelerometer); training may also be too short.
- **Explanation (owner, after the run):** "it doesn't know that it's going sideways. Instead of speed, it should see velocity and compare that vector with the direction it's facing"
- **Takeaway:** sim fidelity mattered more than learning tricks: once twitching stopped paying, big slides and training wobble dropped and robustness rose. But the car still can't sense a slide, so it drives slowly and slides anyway. Next: give it a way to sense sideways motion.

- **Correction (2026-11-09, found in E5):** the table's "slip > 8°" column is not "past the peak". At nominal grip the tires' sideways force peaks at about 9° (lower at lower grip) and falls off gently after it. E5 uses slip > 12° as "clearly past the peak".

### E5 — Can it drive smoothly once it can feel the slide?

- **Owner's question before the run:** with the wheels sawing back and forth, isn't forward acceleration lost to vectoring too, not just traction? Checked: yes, it's in the model. Turned front wheels push partly sideways, and slipping tires drag. Two seconds of full throttle from 10 m/s: 20.5 m/s straight, 18.5 m/s sawing ±1 (plus 4 m/s sideways). Missing from the model: rolling resistance and air drag (top speed is a hard 25 m/s cap), so coasting and scrubbing cost little.
- **Prediction (owner, before the run):** "Both probably help but I think the accelerometer is sufficient"
- **Setup:** E4 (d)'s recipe (realism 2, squashed actions, LR decay, delay 0–60 ms) plus new sensors (`sensors` option in `dynamics.py`): (e) sideways velocity in the car's frame (with forward speed: the velocity vector); (f) accelerometer, forward and sideways (cheap IMU); (g) both. 500k steps, seeds 0–2, 9 runs on 4 CPUs, about 15 min each. Compared with E4 (d).
- **Result:** [report](https://claude.ai/artifact/Peia655v6S9mS9WbW3fZ9F)

  | 3-seed mean | Slip > 12° | Slip > 20° | Mean speed | Mean lap | Held-out /20 per seed | Gate range per seed |
  |---|---|---|---|---|---|---|
  | hand-coded | 0% | 0% | 11.0 m/s | 25.3 s | 19 | 94% |
  | (d) no new sensor | 29% | 7% | 8.5 | 32.0 s | 19 / 20 / 17 | 54 / 61 / 51% |
  | (e) + velocity | 35% | 9% | 8.6 | 32.0 s | 20 / 18 / 20 | 56 / 48 / 59% |
  | (f) + accelerometer | 27% | 6% | 8.6 | 31.6 s | 19 / 20 / 19 | 47 / 52 / 49% |
  | (g) + both | 32% | 8% | 8.8 | 31.0 s | 20 / 20 / 20 | 56 / 49 / 57% |

  - No change in sliding or speed. Slightly more reliable at nominal (both sensors: 20/20 in every seed); the steadiest training curves yet. Both sensors: better with sensor noise (87% at 6% noise vs. 70–77%).
  - Agent's read: the car can now feel the slide, but nothing pushes it to use that. Sliding past the peak costs little here (gentle fall-off, no drag), and the reward is only progress. Its real weakness is caution: full throttle on 2–3% of steps vs. 21%.
  - Gate-range weak spots: grip 0.4–0.6× (0–20%) and mass 0.65× (about 65%), neither ever trained on.
- **Explanation (owner, after the run):** "Is it some of the variance we've added? It learns to be cautious because it never knows if it'll hit mud (I don't remember if we're still using sensor noise etc as well)"
- **Check (agent):** E5 trained at nominal grip and mass, no sensor noise, delay 0–60 ms, and invisible dirt on every track. The hand-coded driver tests the dirt idea:
  - knowing where the dirt is: 19/20, 25.3 s
  - blind to dirt (assumes full grip everywhere): 2/20 (crashes on dirt corners)
  - assuming every corner might be dirt (on a dirt-free track): 20/20, 29.1 s
  
  The RL car's 31–32 s sits near the "assume dirt everywhere" driver, so unseen dirt plausibly explains most of its caution.
- **Takeaway:** a sense only changes behavior if the training makes it pay. Here the dominant risk (invisible dirt) is one no sensor reveals in time, so the car drives every corner as if it might be dirt.

### E6 — What does full randomization cost and buy?

- **Prediction (owner, before the run):** "I think it will get us close to gate range. grip and mass are really related, but one of those curves will move the most. Only costs a little time at normal physics, and maybe not even that" (owner chose the accelerometer only, no velocity sensor, from here on)
- **Setup:** E5 (f) (accelerometer, delay 0–60 ms only) vs. (h) full train-range randomization: grip 0.7–1.1×, mass 0.8–1.2×, delay 0–60 ms, noise 0–2%. Realism 2, squashed actions, LR decay, 500k steps, seeds 0–2, about 6.5 min each.
- **Result:** [report](https://claude.ai/artifact/XFCGQQ14trJC94pC31bh6H)
  - Gate range: 47–52% → **50–72%** (51 / 72 / 50%). Train range 66–73% → 74–92%. Held-out at nominal: 20/20 in all 3 seeds.
  - Curves (delay-only vs. full): noise 6%: 70% → 93%; grip 0.7×: 33% → 63%; grip 0.6×: 20% → 35%; mass 0.65×: 65% → 75%; delay 150 ms: 72% → 62%.
  - Cost: about 6% slower at nominal (33.5 s vs. 31.6 s).
  - Gate-lap failures by grip (300 laps): 0.5–0.7×: 9% finished; 0.7–0.9×: 46%; 0.9–1.1×: 92%; 1.1–1.3×: 93%. Light cars (0.65–0.8×): 33%.
  - Implication: a car perfect at grip ≥ 0.7× and hopeless below would score about 75%. The gate needs most laps at 0.5–0.7× grip to finish. The hand-coded driver does that (95%, about 35 s laps) because it knows the grip.
- **Explanation (owner, after the run):** "It must not be learning grip as a variable to adjust to, but rather as noise to drive conservatively enough that it's ok in the range of what it has seen."
- **Takeaway:** domain randomization without a way to identify the condition buys robustness only inside the trained range, by being cautious enough for its worst case. To handle grip it never saw, the car has to estimate grip as it drives and adapt to it.

### E7 — Can the car estimate grip from a short memory?

- **Prediction (owner, before the run):** "Yes to both." (memory lifts the low-grip laps, and it needs the extra training)
- **Setup:** E6 (h) recipe (full train-range randomization, accelerometer, realism 2, squashed actions, LR decay) vs. + `history=10`: the previous 9 steps' own-motion readings (speed, commands, gyro, accelerometer; 71 inputs). (i) 500k steps, (j) 1M steps; seeds 0–2; 6 runs on 4 CPUs, 9 and 14 min.
- **Result:** [report](https://claude.ai/artifact/1bWtSinzs1rmjRgyddyM6n)

  | | Gate range per seed | Gate laps at grip 0.5–0.7× | Train range | Held-out nominal | Mean lap |
  |---|---|---|---|---|---|
  | (h) no memory, 500k | 51 / 72 / 50% | 9% | 74–92% | 20 / 20 / 20 | 33.5 s |
  | (i) memory, 500k | 43 / 57 / 39% | 2% | 69–80% | 20 / 20 / 18 | 33.3 s |
  | (j) memory, 1M | 37 / 45 / 45% | 2% | 36–57% | 14 / 18 / 18 | 29.3 s |

  - Memory made it worse, and more training made it faster (fastest randomized car) but riskier.
  - Agent's read: the reward is metres gained; a crash costs only the reward it would have earned next (about 10 s at γ = 0.99), with no penalty. The gate counts finished laps. Longer training finds faster-but-riskier trades. Memory adds 54 mostly noisy inputs, and nothing in the reward asks for grip estimation.
- **Explanation (owner, after the run):** "Reward is distant per step. Crashing later isn't punished now so it pays to take risks once you know how to drive the car"
- **Takeaway:** the reward (metres per step) and the gate (finished laps) disagree about risk. Once the car can drive, more training spends that skill on speed, not reliability. Memory didn't help within 1M steps.

### E8 — Does making crashes costly make it finish more?

- **Prediction (owner, before the run):** "K makes it slow and careful, l runs the risk of losing sight of what's in front of you and struggling to drive at all. Hopefully m means you can weight each a little less and get all the benefit. Maybe a little faster if it actually works because the twitchiness could also come from not looking far enough ahead (it does in real life)"
- **Setup:** E6 recipe (full train-range randomization, accelerometer, no memory, realism 2, squashed actions, LR decay), 1M steps, seeds 0–2: (h′) no change; (k) `crash_penalty=100` (about 10 s of top-speed reward); (l) γ = 0.997 (about 30 s horizon); (m) both. 12 runs launched at once; one, (k) seed 0, was killed for running out of memory (about 3 GB per run; 15 GB container) and rerun alone (same seed and thread count, so the same result). From now on: at most 4 runs at once.
- **Result:** [report](https://claude.ai/artifact/PiKxMurtiPYnCcZ4f5GBoa)

  | | Gate range per seed | Train range | Held-out /20 | Mean lap | Gate laps at grip 0.5–0.7× |
  |---|---|---|---|---|---|
  | (h′) no change | 36 / 47 / 45% | 45–69% | 20 / 20 / 11 | 29.7 s | 0% |
  | (k) crash penalty | 23 / 0 / 77% | 0–97% | 9 / 0 / 20 | 36.8 s | 16% |
  | (l) γ 0.997 | **73 / 72 / 75%** | 94–97% | 20 / 20 / 20 | 35.8 s | 30% |
  | (m) both | 2 / 6 / 58% | 0–72% | 0 / 0 / 15 | 46.3 s | 16% |

  - (l): consistent across seeds; delay, mass and noise curves flat at 90–100% across the gate range. Remaining gap is grip below 0.7×.
  - Crash penalty: 2 of 3 seeds in both (k) and (m) learned to barely move (0.1–3.8 m/s in training): never crash, never pay. Phase 1 E2's parking hack again. The one (k) seed that drove is the best single car (77% gate range, 97% train range).
- **Explanation (owner, after the run):** "Crash penalty is stronger than drive reward. Maybe could be tuned but it's likely not the right approach. Looking ahead just means you don't set yourself up for failure."
- **Takeaway:** a longer horizon aligns the reward with finishing without adding a term the car can game. A crash penalty big enough to matter makes standing still the safest policy.

### Decision: a separate test set for the gate (owner, 2026-11-10)

The owner asked whether testing on the held-out tracks in every experiment is like tuning on the test set. It is. Tracks 10000–10019 and the fixed 100 gate-range physics draws scored every experiment since Phase 1, and recipes were chosen partly by them. The network never trains on them, but the recipe choices did see them. Phases 1 and 2 ran their gates on the same tracks.

From now on:
- Tracks 10000–10019 and physics seed 12345 are the **validation set** that experiments report on.
- The gate uses a fresh **test set**: tracks 20000–20019 (`track.TEST_SEEDS`) with physics seed 54321 (`robust.TEST_PHYSICS_SEED`), looked at once.
- The gate report shows both numbers; the gap measures overfitting by choice.

The threshold and physics ranges are unchanged.

### E9 — Can your own PPO drive the randomized car?

- **Prediction (owner, before the run):** "I expect mine to be slightly better but not pass. I'm interested in pursuing better traction handling if that's the case"
- **Setup:** `sdc/phase3/my_ppo.py`: the owner's Phase 1 PPO (advantages, textbook clipped loss, value loss imported unchanged) with E8 (l)'s recipe: full train-range randomization, accelerometer, realism 2, squashed actions, LR decay, γ 0.997, 1M steps, seeds 0–2 (3 runs in parallel, about 11 min each). Compared with E8 (l), SB3 on the same recipe. Seed 0 was fixed as the gate policy before the run.
- **Result (validation set):** [report](https://claude.ai/artifact/7YNoj4TU3b4UvceKaS9e7J)

  | | Gate range per seed | Mean | Held-out /20 | Train range | Grip 0.5–0.7× laps | Mean lap |
  |---|---|---|---|---|---|---|
  | SB3 (E8 l) | 73 / 72 / 75% | 73% | 20 / 20 / 20 | 94–97% | 30% | 35.8 s |
  | owner's PPO | 66 / 73 / 80% | 73% | 20 / 20 / 20 | 93–97% | 27% | 35.9 s |

  - Same average, wider spread. Seed 2 (80%) is the best run so far; seed 0, the gate policy, is the lowest of the six.
- **Explanation (owner, after the run):** "Needs more training to converge?"
- **Agent's note:** each seed's validation gate-range score swung between 55% and 95% from one 50k-step check to the next, right to the end, and the learning rate decays to zero by 1M. So each seed's final policy is partly a matter of where the swings stopped. The 100-lap score itself has about ±4.5 points of measurement noise.
- **Takeaway:** the owner's PPO matches SB3 on the hardest recipe so far (73% mean). Seed-to-seed spread is large, so one seed is a noisy measure, which matters for a gate decided by one seed.

## Gate check

- **Run:** 2026-11-10, `python -m sdc.phase3.gate --run runs/p3-e9-s0` (the gate policy is E9 seed 0, trained by the same code and settings `python -m sdc.phase3.gate` uses), commit 936435f. Owner's PPO, full train-range randomization, accelerometer, γ 0.997, 1M steps, seed 0 (fixed before E9's results were seen), 1 torch thread, cloud container (4 CPU, no GPU). [Gate report](https://claude.ai/artifact/6bd3xNLDQNKNFxDipMP1bv)
- **Result vs. threshold:**
  - Test set (tracks 20000–20019, never used before; gate-range physics, 5 draws per track, physics seed 54321): **70/100 = 70%** vs. ≥ 85%: **fail**.
  - Same policy on the validation set: 66%. The 4-point gap is within the noise of 100 laps, so there's no sign that choosing recipes on the validation set flattered the numbers.
  - By grip in the draw: 0.5–0.7×: 15% (26 laps); 0.7–0.9×: 73% (22); 0.9–1.1×: 97% (29); 1.1–1.3×: 96% (23). Light cars (mass < 0.8×): 55%. Test tracks at nominal: 20/20. The hand-coded driver (knows the grip) on the same 100 laps: 93%.
  - Robustness curves (test tracks, one knob at a time): delay, mass and noise stay at 90–100% across the gate range; grip falls from 90% at 0.7× to 35% at 0.6× and 5% at 0.5×. Published in the gate report.
- **Verdict:** **FAIL.** Everything but low grip is handled; grip below the training range (0.5–0.7×) is not. The owner chose to continue on traction handling.

## After the gate: traction handling (owner chose to continue, 2026-11-10)

Plan: a two-step version of Rapid Motor Adaptation (RMA). E10, oracle: give the policy the true grip under the car (sim only) and check that it can extrapolate below its 0.7× training grip. E11, estimator: learn to predict grip from the last 1 s of readings, supervised by the sim's true grip, and feed the estimate to the policy. The gate isn't rerun until a candidate is ready; a rerun uses the same test set and threshold.

### E10 — With perfect grip knowledge, can it drive grip it never trained on?

- **Prediction (owner, before the run):** "the task is still extrapolation. However, acceleration is linear in friction and all the cars controls set acceleration, so I think that should be ok to extrapolate."
- **Setup:** E8 (l)'s recipe (full train-range randomization, grip 0.7–1.1×; accelerometer; γ 0.997; 1M steps; SB3) plus the oracle `grip` sensor (true grip under the car / nominal, including dirt). Seeds 0–2, 3 runs in parallel. Compared with E8 (l).
- **Result (validation set):** [report](https://claude.ai/artifact/WbYJa9x5WDjTfEmEMywM29)

  | | Gate range per seed | Grip 0.5× sweep | Grip 0.4× sweep | Held-out /20 | Mean lap (nominal) |
  |---|---|---|---|---|---|
  | E8 (l), no oracle | 73 / 72 / 75% | 13% | 2% | 20 / 20 / 20 | 35.8 s |
  | oracle grip | **83 / 81 / 88%** | **93%** | **68%** | 20 / 18 / 20 | 33.3 s |

  - It extrapolates, and it adapts its speed to the grip (seed 0's mean lap: 48.4 s at 0.4×, 38.5 s at 0.7×, 32.7 s at 1.0×, 28.8 s at 1.3×). Without the oracle the car laps 35–36 s at every grip.
  - Remaining gate-lap failures by grip: 0.5–0.6×: 62%, 0.6–0.7×: 77%, 0.7–0.9×: 81%, 0.9–1.1×: 96%, 1.1–1.3×: 91%. Light cars (mass < 0.8×): 57%, consistent with the review's grip cliff.
  - Smoothness unchanged (steering change about 0.34 per step).
- **Explanation (owner, after the run):** "Knowing grip is a huge advantage on something that's difficult learn or infer. The fact that it's linear in acceleration means whatever you learn about it can be extrapolated. We can't keep the oracle, but we can add something to estimate traction."
- **Takeaway:** with the grip as an input, the policy learns how to scale its driving to grip, and that scaling extrapolates far beyond the training range. Without it, randomization only teaches one cautious speed. An estimate of the grip is worth pursuing.
- **Also launched (owner's request):** two background reviews: (1) a code review of the physics, controls and sensors for correctness and real-world sense; (2) an investigation of why the car saws the steering instead of holding a clean line (foresight/planning among the hypotheses).

### Sim review (2026-11-10)

A background code review of the physics, controls and sensors: [phase3-sim-review.md](phase3-sim-review.md). The core equations are right. Found:
- Two bugs: the low-speed blend erases slides and spins; a locked front tire pushes along the wheel.
- A grip cliff (verified): fixed drive force vs. grip-scaled tire peak, so full throttle on dirt or light cars spins both axles and leaves zero cornering force.
- Sawing is nearly free: no drag or rolling resistance; the speed cap hides losses.
- Smaller realism gaps.

The owner added a hypothesis for the twitchiness investigation: sensor noise makes the car think it's off its line.

### Twitchiness investigation (2026-11-10)

Full write-up: [phase3-twitch-investigation.md](phase3-twitch-investigation.md).
- The deterministic policy saws, but it costs almost nothing (smoothing it at evaluation halves sliding with no lap-time or completion change).
- Root cause: PPO's exploration noise never shrinks (std about 0.5 to the end), and the policy learns negative feedback on its own last command to fight it.
- Low initial noise removed the sawing (0.43 → 0.06 per step).
- Sensor noise amplifies sawing but doesn't cause it.
- Foresight is ruled out for steering: a clone of the hand-coded driver's steering from the same rays drives a clean line.
- The slowness is throttle caution.
- Aside: one 300k-step run scored 98% validation gate range; longer training trades reliability for speed.

**Decision (owner, 2026-11-10):** bundle all fixes and continue with them in place:
- realism 3: the review's physics fixes
- an exploration-noise schedule (no reward penalty)
- E11, a learned grip estimator

### Realism 3 and the noise schedule (built 2026-11-10)

`dynamics.py` `realism=3` (`_substep3`); realism 1 and 2 are kept unchanged for reproducing E1–E10:
- A sliding tire (locked by the brake, or spinning under drive) pushes with 75% of peak against its slip velocity, so a spinning wheel still corners.
- Cornering stiffness scales with axle load.
- Aero drag (CdA 0.8 m²) and rolling resistance (0.015).
- Relaxation length 0.3 m.
- The low-speed blend uses total speed and doesn't depend on the substep.
- The accelerometer reads the acceleration the car actually gets.

Checks against the review:
- A low-speed slide survives (sideways speed 8.0 m/s kept, was 0).
- Full left + full brake no longer turns right.
- Full throttle on dirt keeps cornering (yaw 0.31 rad/s, was 0.06).
- Sawing on a straight now costs 6% of distance (was 1.3%).
- Coasting from 20 m/s loses 0.3 m/s per second.
- Hand-coded driver: 95–100% held-out at nominal, 0.5× grip, 0.65× mass and 120 ms delay; 97% validation gate range.

Not changed: the servo position isn't observed; the off-track test uses the centre of mass; the speed sensor is ideal.

Exploration-noise schedule (`std_cap=[start, end]`, SB3 and the owner's PPO): the policy's noise std (before tanh) is capped, and the cap is lowered linearly over training.

### E11 — The fixed car with calmer exploration

- **Prediction (owner, before the run):** "Smoother steering but not fixed. Better gate scores, lap times similar"
- **Setup:** E8 (l)'s recipe (full train-range randomization, accelerometer, γ 0.997, 1M steps, SB3) on realism 3 with `std_cap=[0.5, 0.1]`. (o) no grip information; (p) + oracle grip. Seeds 0–2, at most 4 runs at once, about 11 min each. Compared with E8 (l) and E10 (realism 2, free noise).
- **Result (validation set):** [report](https://claude.ai/artifact/EYjRn3NgFUByWfiYvJkJzf)

  | | Steering change/step | Slip > 4° | Mean lap | Gate range per seed | Held-out /20 |
  |---|---|---|---|---|---|
  | E8 (l), no grip info | 0.46 | 68% | 35.8 s | 73 / 72 / 75% | 20 / 20 / 20 |
  | **(o)** fixed car, calm noise | **0.17** | 40% | **30.0 s** | 57 / 55 / 60% | 20 / 19 / 20 |
  | E10, oracle | 0.34 | 74% | 33.3 s | 83 / 81 / 88% | 20 / 18 / 20 |
  | **(p)** fixed car, calm noise, oracle | **0.16** | 47% | **28.7 s** | 52 / 58 / 61% | 17 / 17 / 16 |

  - Smoother and faster, but less reliable. (p)'s validation gate range peaked at 85–95% around 300–600k steps in every seed, then fell to 35–50% by 1M as laps got faster. (o) peaked at 75–80%.
  - Agent's read: the reliability-for-speed trade (E7, twitchiness aside) is stronger now. Less late-training noise plus drag that rewards carrying speed push the car toward faster, riskier driving. We keep the final policy; choosing a checkpoint by the validation gate range would have scored much higher, and is legitimate now that the gate has a separate test set.
- **Explanation (owner, after the run):** "The longer it trains, the more nuance it can pick up. So here we see it learn to drive first, then attempt to drive fast. First, it picks up the broad skill of driving, but then it moves into more of the finer reward of driving fast at the cost of driving at all."
- **Takeaway:** the fixes made it smoother and faster, and showed that the final policy is the wrong one to keep: reliability peaks mid-training.
- **Decision (owner, 2026-11-10):** early stopping. Pick the checkpoint with the best validation gate range ("tune to the behavior we want rather than the end behavior").
- **Owner's idea (2026-11-10), for discussion:** change the training paradigm to match how a driver learns a track:
  - drive a few slow sighting laps to learn the surface and how hard to push
  - train on physics ranges much wider than reality, so reality is a subset (interpolation, not extrapolation), with exposure keeping it from being cautious everywhere
  - later, learn a map of where the low-grip spots are
  
  The concern: constant probing (slamming the brakes every second) isn't wanted.
- **Decisions (owner, 2026-11-10), after discussing the idea:**
  - the gate protocol gets an out-lap: one lap to learn the conditions, then the counted lap
  - track memory (a map of low-grip spots) is deferred to a later phase
  - plan: E12 early stopping; E13 grip estimator (1 s window vs. a whole-drive running estimate, with and without the out-lap); then decide on much wider training ranges and the gate rerun

### E12 — Early stopping

- **Prediction (owner, before the run):** "From what you said earlier, I'd expect around 300k and lap times to be a little over 30 seconds"
- **Setup:** E11's runs repeated exactly (same seeds; the final policies reproduce E11's numbers), with `select=true`: a checkpoint every 100k steps; the one with the best validation gate range (100 laps) is kept. (o′) no grip info, (p′) oracle grip. 6 runs, 4 then 2 in parallel.
- **Owner's question during the run:** adjust the rewards instead (e.g. a decay)? Agent's options:
  - an even longer horizon (γ 0.999)
  - a "lose the lap" crash cost (take back the progress earned since the last start-line crossing; parking has nothing to lose, so it can't be gamed the way E8's fixed penalty was)
  - a lap-completion bonus
  - a reward weight schedule
  
  Recommended: keep early stopping and test "lose the lap" next. **Owner's decision:** test options 2 and 3 ("lose the lap" and the lap-completion bonus).
- **Result (validation set):** [report](https://claude.ai/artifact/2Gubj9FYKiHrtc587oZeRq)

  | | Chosen checkpoint | Gate range, chosen | Gate range, final | Mean lap, chosen | Mean lap, final |
  |---|---|---|---|---|---|
  | (o′) no grip info | 200k / 500k / 600k | 80 / 72 / 86% | 57 / 55 / 60% | 34.5 s | 30.0 s |
  | (p′) oracle grip | 300k / 500k / 100k | **92 / 93 / 90%** | 52 / 58 / 61% | 34.1 s | 28.7 s |

  - The chosen validation numbers are flattered: each is the best of 10 on that same score. The test set is the honest check.
  - Oracle checkpoints: gate laps at grip 0.5–0.6×: 90%; light cars: 86%; 20/20 nominal in every seed. Without grip info the weak spot is still low grip (0.5–0.6×: 40%).
- **Explanation (owner, after the run):** "The Oracle car must be getting more confident"
- **Takeaway:** the best behavior appears mid-training, and how much of it survives to the end depends on how confident the car gets. The oracle car becomes confident fastest, so it has the most to lose by the end. Early stopping keeps the reliable version, at about 4–5 s per lap.

### E13 — Estimating the grip

- **Prediction (owner, before the run):** "Whole drive does better, out-lap helps, both below oracle"
- **Setup:** `sdc/phase3/grip_estimator.py`. E12's best oracle checkpoints, frozen; the oracle grip input is replaced by a learned estimate:
  - (q) 1 s window: an MLP on the last 10 steps of speed, commands, gyro and accelerometer
  - (r) whole drive: a GRU reading the drive from the standing start
  
  Both are trained by regression on the true surface grip, with 4 rounds × 60k samples. Round 1 drives on the true grip; later rounds drive on the estimate (DAgger). Seeds 0–2, one per checkpoint. Each is evaluated without and with an out-lap (`evaluate.run_laps(outlap=True)`: an out-lap then the counted lap; finishing needs both).
- **Result (validation set):** [report](https://claude.ai/artifact/BjVi1QDJP5bNhtnLsXaAUH)

  | Grip input | Gate range per seed | Mean | With out-lap | Counted lap, out-lap survived | Estimate error |
  |---|---|---|---|---|---|
  | true grip (oracle) | 92 / 93 / 90% | 92% | – | – | 0 |
  | (q) 1 s window | 91 / 92 / 86% | 90% | 85 / 87 / 76% | 92% | 0.06 |
  | (r) whole drive | 91 / 92 / 82% | 88% | 91 / 83 / 75% | 95% | 0.06 |
  | none (E12 o′, its own best checkpoints) | 80 / 72 / 86% | 79% | – | – | – |

  - The estimators keep almost all of the oracle's benefit. Both settle within about 5 s (error 0.08–0.10 in the first 5 s, about 0.055 after); the whole-drive memory adds no accuracy.
  - The out-lap lowers the gate range, because finishing means surviving two laps: 30–37 of about 50 failures happen in the out-lap. The counted lap after a survived out-lap does better (92–95% vs. 88–90%), most for the whole-drive estimator. The policy doesn't drive its out-lap more carefully. The counted lap is 1.5–3 s faster (flying start).
- **Explanation (owner, after the run):** "E7 didn't tell it what to do with that memory."
- **Takeaway:** the same readings that E7's policy ignored give an accurate grip estimate within about 5 s once a network is trained to predict grip from them. An explicit target turns memory into information; RL alone didn't discover it in 1M steps. The out-lap helps the counted lap but costs crashes, because the policy drives it like any other lap.
- **Owner's question:** should the out-lap have a different reward? "I would drive cautiously with some bursts to get a feel for it."

Plan from here: E13 grip estimator (on E12's best oracle checkpoints); E14 reward options ("lose the lap" crash cost, lap-completion bonus); then decide on much wider training ranges and the gate rerun (with an out-lap).

## Handoff to next phase
