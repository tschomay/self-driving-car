# Traction review for Phase 4.6: tires, friction and weight transfer on the truggy physics

2026-11-18. A second background review, after [the physics review](physics-review-phase46.md). It covers the car in `real="truggy"` (Phase 4.5's "all" plus "esc", "lsd", "power" and "heavy"). The question: can a learner work this car like a real 1/8 truggy (Traxxas Revo, Losi 8IGHT-T)? That means trail-braking, using weight transfer to rotate into a corner, power oversteer, drifting and catching slides, on loose dirt and clay.

How the tests ran:
- `dr="nominal"` unless stated, plus six gate-range draws.
- Mostly flat ground (the loop terrain, flattened), plus the teacher (`LoopExpert`) on validation loops 10000–10019.
- Tire forces come from `mj_contactForce`, summed per tire and averaged over the 10 physics substeps of each 20 ms control step.
- No file under `sdc/` or `tests/` was edited. The scripts and the monkeypatched prototypes are in the session scratchpad (`traction/`: `h.py` harness, `variants.py` prototypes, `t1`–`t15` tests).

## Summary

| Item | Sim measurement (truggy, nominal μ = 1) | Realistic target | Verdict |
|---|---|---|---|
| Contact model | Pyramidal cone, condim 3, impratio 1, solref 0.02 / 1, default solimp; 2 ms implicitfast, Newton, 10 substeps | — | **Bug-level side effect** (next row) |
| Friction vs compass heading | The friction limit is a diamond in **world** x/y. A tire sliding at 45° to the world axes gets **0.71 μ**, along an axis 1.0 μ. At 22.5° the force is also 0.15 g off the slip direction. | The same in every direction | **Bug.** Grip depends on which way the track points. |
| Cornering limit | 0.77 g (pyramidal: the average over a circle's headings); **0.89 g** with an elliptic cone | 0.8–1.2 g on prepped clay, 0.55–0.7 on loose dirt | OK once the cone is fixed |
| Slip curves | Regularised Coulomb. Lateral force rises linearly with slip *speed*, is fully saturated by about 0.15–0.2 m/s of slip, then stays flat to 90°. No peak, no drop-off. Speed-dependent: 2° saturates at 10 m/s, 1 m/s needs 10°. | Peak at about 6–12° slip angle and 0.1–0.2 slip ratio. Prepped clay drops 15–30% past the peak; loose dirt saturates without a peak (exponential-like). | Defensible for loose dirt. Too forgiving for clay. |
| Longitudinal vs lateral | One isotropic μ (apart from the pyramid artefact) | Knobby tires: longitudinal a little above lateral | Acceptable |
| Combined slip | Pyramidal: a diamond (\|F\| falls to 0.72 μ when braking and cornering at once). Elliptic: a true friction circle (\|F\| = 1.00 μ). | Friction ellipse | Elliptic is right |
| Rolling resistance | condim 3, so the 0.005 rolling coefficient is **ignored**. Coast-down at 8 m/s is 0.09 g, all of it from the ESC's 10% drag brake, and falls to 0 with speed. | About 0.03–0.08 g on loose dirt, roughly speed-independent | Slightly low at low speed; minor |
| Wheelspin threshold | Full throttle from 1 m/s: 0.97 g, traction-limited, front wheels spinning (fronts unloaded to 1.5 N each). Above about 6 m/s at μ 1, power-limited (0.73 g, rears at 66% of grip). At μ 0.55: wheelspin at all speeds. | Traction-limited well past jump speed | OK at low speed; no wheelspin at speed on high grip |
| Braking | Full brake locks all four wheels: 1.0 g at μ 1, straight. **Front flip (endo) at μ ≥ 1.2**, and at μ ≥ 1.1 for light gate draws (mass × 0.75). | 0.7–0.9 g; a stoppie is possible, flipping over the nose on flat clay is rare | **Bug-ish cliff** in the top of the gate range |
| Longitudinal weight transfer | 1 g braking: front axle +10.2 N (m·a·h/L predicts +8.6). Rear down to 1 N; rear suspension at full droop (−2.9 of −3.0 cm). Pitch −7.9°. Acceleration 0.97 g: −9.3 N (−8.2 predicted), +7.5°. | m·a·h/L, with h/L = 0.35 | **Correct**, slightly more than static (pitch dynamics) |
| Lateral weight transfer | 0.74 g: outside minus inside 12.7 N (2·m·a·h/t predicts 13.4). Roll 6.4° (8.6°/g). | Same formula; truggies run anti-roll bars, about 3–6°/g | Correct; roll a bit soft |
| Load → grip | Linear: each tire's grip is μ·Fz, with no load sensitivity | μ falls about 10–20% per doubling of load | Simplified: lateral transfer costs no total grip, so roll-stiffness balance does nothing |
| Trail-brake rotation | Turn-in at 7 m/s, steer 0.6 (elliptic). Brake −0.2 / −0.4 for 0.4 s: yaw rate at 0.3 s goes 1.14 → 2.25 / 2.70 rad/s, β 16° / 25°. Front Fz 12.9 → 15.3 / 17.6 N, rear 11.4 → 8.4 / 6.6 N. Full brake locks the fronts (less rotation). Lift: mild tuck. Throttle: push. | Exactly this pattern | **Works.** Pyramidal: the same input gives β 15–61° depending on compass heading. |
| Power oversteer | μ 1: none (4WD push, β ≤ 2°). μ 0.7: β 8° (elliptic), 47° with the slip-drop prototype | Available on loose dirt with a throttle stab | Weak; needs the slip drop (or a rear torque bias) |
| Steady drift | Countersteer + throttle grid (81 controllers): at μ 1 none holds 12–45° of slip for 1 s (elliptic); at μ 0.7, 7 / 81 hold ≥ 1 s. The cause: all four wheels spin, each tire's force follows its wheel's heading, and the front pulls the car straight. | Sustained 4WD power slides on loose dirt, with throttle and countersteer | **Hard.** Better with the slip drop (12 / 81 ≥ 1 s at μ 1). |
| Friction randomisation | Tire and ground scaled together (contact μ = max = draw). Chassis not scaled (μ 1). Train 0.7–1.1, gate 0.55–1.25. Pyramidal makes the *effective* gate range 0.39–1.25. | Loose dirt about 0.55–0.7, prepped clay 0.9–1.2 | Ranges sensible; implementation as intended |
| Numerics | No energy gain (5 s slides: gains total 4 mJ of 66 J). Per-2 ms Fz chatter of ±2.2 N (36% of static load) from heightfield triangle crossings (2–7 contacts per tire, +40–90% one-step spikes); control-step means steady. No speed snag on flat. | — | OK; chatter only matters for IMU vibration |

**Bottom line.** Weight transfer is modelled correctly, and trail-braking does rotate the car the way it should. But the default **pyramidal friction cone makes grip depend on the car's compass heading** (1.0 μ along the world axes, 0.71 μ at 45°). So the same brake-and-turn input gives anything from a gentle 15° rotation to a 61° spin depending on which way the corner faces. That's the top fix: `cone="elliptic"`, a one-word change. Second, a full brake flips the car over its nose in the top ~10% of the gate friction range. Capping the ESC brake at 0.65 of motor torque removes the flip and keeps 0.9 g stops. Third (optional), a sliding-friction drop would give power oversteer and make drifting a balancing act. Drifting is hard on this 4WD regardless.

## Real reference values

| Quantity | Value | Source |
|---|---|---|
| μ, car tire on gravel (peak / sliding) | 0.60 / 0.55 | [hpwizard table](https://hpwizard.com/tire-friction-coefficient.html) (source of the numbers not given) |
| μ, dry earth road | 0.68 / 0.65; wet 0.55 / 0.40–0.50 | same |
| μ, 1:10 autonomous car on indoor surfaces (force gauge) | tile 0.69, acrylic 0.84, cardboard 1.02 | [arXiv 2509.15423](https://arxiv.org/html/2509.15423v3) |
| μ, 1/8 truggy on prepped clay | about 0.9–1.2 (estimate; RC racers corner above 1 g on high-bite clay). No measured source found. | — |
| Off-road slip curve shape | "exponential-like saturation" on off-road surfaces, against the sharp peak of radial tires on pavement; ploughing can push side force above the normal load at large slip angles | [Allen et al., SAE 970559](https://papers.sae.org/publications/technical-papers/content/970559/) (abstract) |
| Lateral force vs slip angle on soil, gravel, snow | Peak and shape depend on soil type and depth | [Coutermarsh & Shoop, SAE 2010-01-0374](https://saemobilus.sae.org/content/2010-01-0374) (abstract only) |
| MuJoCo friction mixing | Equal priority: "the element-wise maximum of each friction coefficient over the two geoms is used"; a higher-priority geom's friction wins | [MuJoCo modeling docs](https://mujoco.readthedocs.io/en/latest/modeling.html) |
| MuJoCo cones | Pyramidal is the default, an approximation of the elliptic cone; "The elliptic model is more principled and more consistent with physical intuition." Against slip: "use elliptic cones, large impratio, and the Newton algorithm" | [MuJoCo computation docs](https://mujoco.readthedocs.io/en/latest/computation/index.html), [modeling docs](https://mujoco.readthedocs.io/en/latest/modeling.html) |

The car (from the model): 2.48 kg, CG 0.104 m above the ground, centred. Wheelbase 0.30 m, track 0.28 m, so h/L = 0.35 and h/t = 0.37. Static load 6.1 N per tire. Suspension: 2.4 cm of droop and 5.6 cm of bump from static (2.4 cm sag).

## Details

### 1. Contact and friction model

- **Settings** (`car.py:169` and the geoms). `<option timestep="0.002" integrator="implicitfast"/>`, so everything else is at default:
  - cone pyramidal, impratio 1;
  - Newton solver, 100 iterations, tolerance 1e-8, no noslip;
  - every geom condim 3, solref (0.02, 1), solimp (0.9, 0.95, 0.001, 0.5, 2), margin and gap 0, priority 0.
  - The policy acts every 10 substeps (50 Hz).
- **Friction combination.** Tire and ground both carry `friction="1.0 0.005 0.0001"`. With equal priority MuJoCo takes the element-wise max, and `_apply_phys` (`env.py:157`) scales both by the draw, so the contact gets exactly the drawn μ (checked: tire = ground = 1.16 on a gate draw). That's what's intended. Two loose ends:
  - The **chassis isn't scaled**, so a bumper or belly scraping the ground slides at μ = max(1.0, draw). That is more than the tires on every low-grip draw. Minor (crashes and bank scrapes).
  - The second and third friction numbers do nothing: condim 3 has no torsional or rolling friction. So there is **no rolling resistance from the tires**.
- **Contacts.** A capsule tire on the heightfield gets 2–7 contacts, one per triangle prism it overlaps. The contact frame's tangent axes come from the normal alone, so on level ground they are the **world** x and y axes, whatever the car's heading.
- **Pyramidal anisotropy** (`t2_aniso.py`). The pyramid's edges are ±μ along each tangent axis, so the friction limit is the diamond |f_x| + |f_y| ≤ μ·f_n in world axes.

  | Sliding direction vs world axes | 0° | 15° | 30° | 45° | 60° | 75° | 90° |
  |---|---|---|---|---|---|---|---|
  | Sideways slide, deceleration (g) | 0.99 | 0.90 | 0.75 | 0.72 | 0.78 | 0.88 | 0.99 |
  | Elliptic cone | 1.03 | — | — | 1.01 | — | — | — |

  - A locked-wheel straight slide gives 1.01 / 0.82 / 0.72 g at 0 / 22.5 / 45°. At 22.5° there is also 0.15 g of force *across* the slide, so slides curve.
  - The loop courses point every way, so the learner meets a different car on each straight. A steady circle averages over all headings, giving a mean capacity of (2/π)·√2·ln(1+√2) = 0.79. That's why the earlier review measured "max 0.75–0.8 g at μ 1, grip a little low".
- **Elliptic cone** (`m.opt.cone = elliptic`, impratio 1): friction is isotropic within 1–3%. Steady cornering rises to 0.89 g at μ 1 and 0.64 g at μ 0.7 (pyramidal: 0.77 / 0.55). It costs about 12% more physics time per control step (1.49 against 1.33 ms, flat ground). impratio 10 changed nothing measurable here.
- **No slip curve.** MuJoCo's soft friction constraint behaves like Coulomb plus a creep regularisation (`t3_slip.py`, a rig that holds the chassis at a slip angle with free-rolling wheels):

  | Slip angle (5 m/s) | 0.5° | 1° | 2° | 3° | 5° | 10° | 20° | 45° | 90° |
  |---|---|---|---|---|---|---|---|---|---|
  | F_y / F_z (elliptic) | 0.31 | 0.59 | 0.97 | 0.98 | 1.00 | 1.00 | 1.01 | 1.04 | 1.02 |

  - The force builds with slip *speed* (about 0.15–0.2 m/s to saturate), not slip angle:
    - at 1 m/s, 2° gives 0.28 and 10° gives 1.0;
    - at 10 m/s, 2° already gives 0.98.
  - So at racing speed a tire goes from grip to full slide within 1–2°, with no warning band. At walking speed it is mushy.
  - Longitudinally: slip ratio ±0.05 gives 0.35 μ, ±0.1 gives 0.6, ±0.2 gives 0.9, and from 0.5 it is flat at 1.0.
  - Real loose dirt saturates without much of a peak (SAE 970559), so the plateau is defensible there. Prepped clay behaves more like pavement, with a peak and a 15–30% drop. The sim's tires never lose grip by sliding more, so a slide costs nothing to keep.

### 2. Traction per axle

- **Launch, μ 1, full throttle from 1 m/s:**
  - 0.97 g, with all four tires at the friction limit (|F|/F_z 0.96–1.0).
  - The front tires are unloaded to 1.4–1.5 N each and spin (wheel 4.8 m/s against car 3.4). The centre LSD (0.03) lets the fronts overspin rather than sending torque rearwards.
  - From 0.4 to 0.8 s (4–6 m/s), 0.73 g: the rears are at 66% of grip and the motor is the limit (back-EMF).
  - 0–6 m/s takes 0.68 s.
- **Launch, μ 0.55:** 0.53–0.54 g, all four spinning (wheel 9 m/s at 4 m/s ground speed). Wheelspin at any speed.
- **Wheelspin above 6–7 m/s at μ ≥ 1:** none. The motor (0.6 N·m per wheel, 16 m/s no-load) gives about 1 g of thrust at 7 m/s, falling linearly. So power oversteer at speed on high-grip draws isn't available. That's realistic for a 4WD on clay at that power, less so for a 6S electric truggy.
- **Braking:**
  - The full ESC brake is 2.4 N·m at the motor (44 N at the tires, 1.8 g of capacity), so it locks every wheel at any gate μ. 8 → 0 m/s averages 0.99 g at μ 1 (heading change 2°), 0.72 g at μ 0.7.
  - A learner has to modulate the brake to keep steering. That's realistic: a strong RC brake also locks on dirt.
- **Brake flip (`t13_endo.py`).** Full brake from 8 m/s on flat ground:

  | μ (nominal mass) | 0.9–1.15 | 1.2 | 1.25 |
  |---|---|---|---|
  | Minimum pitch | −9° | −51° (pyramidal) / −10° (elliptic) | **−87°, flips over the nose** |

  - At mass × 0.75 (within the gate range) the flip starts at μ 1.1.
  - Mechanism, from the traces:
    - the rear lifts at 0.08 s (rear F_z = 0) and the nose dives on the soft front springs;
    - the brake keeps acting on the airborne rear wheels, which the damped centre diff lets turn slowly backwards (−0.3 m/s);
    - the reaction torque on the chassis adds to the pitch, as it does in the air ("brake = nose down").
  - Lighter chassis means less pitch inertia against the same brake torque, hence the lower threshold.
  - **The teacher is not affected:** none of its 15 gate-range failures (60 runs: 20 loops × three physics seeds) involved braking. They are jump landings and low-μ run-offs. A learner that slams the brake on a high-grip draw will flip, though.
  - Prototype: capping the brake at 0.65 × motor torque removes it (−8° even at μ 1.25 with mass × 0.75) and still stops at 0.89 g at μ 1.

### 3. Weight transfer

Steady windows (flat, world heading 0, forces averaged per substep; `t4_wt.py`):

| Manoeuvre | a | F_z fl / fr / rl / rr (N) | Measured transfer | m·a·h/L or 2·m·a·h/t | Pitch / roll | Suspension (cm, + = bump) |
|---|---|---|---|---|---|---|
| Static | 0 | 6.1 / 6.1 / 6.1 / 6.1 | — | — | 0 | 0 (2.4 sag) |
| Full brake, μ 1 | −1.02 g | 11.2 / 11.2 / 1.0 / 0.9 | front +10.2 | +8.6 | −7.9° | front +1.2, **rear −2.9 (droop limit −3.0)** |
| Full brake, μ 0.55 | −0.57 g | 9.2 / 9.0 / 3.1 / 3.3 | front +6.0 | +4.8 | −5.2° | +0.6 / −2.1 |
| Full throttle, μ 1 | +0.97 g | 1.5 / 1.4 / 10.8 / 10.7 | front −9.3 | −8.2 | +7.5° | **front −2.8** / rear +1.1 |
| Corner 4 m/s | 0.72 g | 3.0 / 9.2 / 3.0 / 9.1 | outside − inside 12.3 | 13.0 | roll 6.3° | inside −2.4, outside +0.7 |
| Corner 5 m/s | 0.74 g | 2.8 / 9.2 / 3.0 / 9.4 | 12.7 | 13.4 | roll 6.4° | −2.5 / +0.7 |

- **Transfer is right.** Longitudinal transfer runs 10–25% above the static formula (pitch dynamics, unsprung mass). Lateral transfer is within 5%.
- **Front/rear split in roll:** equal (springs equal, no anti-roll bars). With slider suspension the roll centre is effectively at the ground, so all the transfer goes through the springs.
- **Suspension limits:**
  - At 1 g of braking the rear tops out at full droop and is carried by the droop stop. The rear tires are left with 1 N, almost no lateral grip, which is why braking rotates the car so strongly.
  - At 0.97 g of acceleration the fronts top out and are left with 1.5 N, which is why full throttle pushes (understeer).
  - The droop travel (2.4 cm) is short for a truggy, which typically has 30–40% of its travel in droop. That makes the load swing almost all-or-nothing at 1 g.
- **Roll:** 8.6°/g is soft against a truggy with bars (about 3–6°/g), but harmless: rollover needs t/2h = 1.35 g, and nothing flipped in steady cornering at μ 1.
- **More load, more grip:** yes, linearly (|F|/F_z ≈ μ on every saturated tire). There is no load sensitivity:
  - transferring load from the inside to the outside tire costs no total grip, so roll stiffness balance has no handling effect;
  - front/rear transfer still moves axle capacity (μ·F_z), which is what trail-braking needs.
- **Trail-brake** (`t6_trail.py`, 7 m/s, turn-in steer 0.6, μ 1, elliptic):

  | Programme | Yaw rate at 0.3 s | Max β | Heading after 1.2 s | F_z front / rear (N, 0.1–0.3 s) |
  |---|---|---|---|---|
  | Hold throttle 0.5 | 1.14 rad/s | 1° | 91° | 12.9 / 11.4 |
  | Lift (0) | 1.42 | 3° | 107° | 13.7 / 10.3 |
  | Brake −0.2 for 0.4 s, then 0.5 | 2.25 | 16° | 109° | 15.3 / 8.4 |
  | Brake −0.4 for 0.4 s | 2.70 | 25° | 125° | 17.6 / 6.6 |
  | Brake −1 for 0.4 s (fronts lock) | 2.17 | 19° | 113° | 21.2 / 2.4 |
  | Full throttle | 0.63 | 2° | 54° | 7.2 / 16.4 |

  - Brake-and-turn bites the front and loosens the rear. Lift tucks the nose a little. Throttle shifts the load back and pushes.
  - That's the real pattern. The learner can rotate into a corner and power out.
- **Heading dependence, pyramidal** (`t6b_heading.py`). Same programme, car pointing 0 / 15 / 30 / 45 / 60 / 90° in the world:
  - brake −0.4: max β **61 / 35 / 23 / 15 / 42 / 54°**, end speed 1.8–5.2 m/s;
  - elliptic: 20–25° and 4.2–4.6 m/s everywhere.
  - A policy cannot learn a consistent rotation technique when the response depends on an angle it can't sense.

### 4. Slip behaviour and drifting

- **Slip curves:** see section 1. Flat Coulomb past about 2° at 5 m/s, and the same past slip ratio 0.2–0.5. No peak, no drop.
- **Combined slip** (`t5_circle.py`, 6° slip angle plus a wheel slip ratio):

  | Slip ratio | 0 | −0.05 | −0.1 | −0.2 | −0.5 | −1 |
  |---|---|---|---|---|---|---|
  | Pyramidal F_y / F_x / \|F\| | 1.00 / 0.00 / 1.00 | 0.96 / −0.04 / 0.96 | 0.82 / −0.18 / 0.84 | 0.56 / −0.45 / **0.72** | 0.05 / −0.95 / 0.96 | 0.00 / −1.01 / 1.01 |
  | Elliptic F_y / F_x / \|F\| | 0.99 / −0.03 / 0.99 | 0.99 / −0.16 / 1.00 | 0.96 / −0.29 / 1.00 | 0.85 / −0.53 / 1.00 | 0.43 / −0.91 / 1.00 | 0.24 / −0.97 / 1.00 |

  - Elliptic: a proper friction circle. Braking eats lateral grip as on a real tire.
  - Pyramidal: a diamond, with an extra 28% grip loss part-way.
- **Steady drift** (`t7_drift.py`):
  - Initiation: 7 m/s, steer 1, brake −0.4 for 0.35 s (reliably 20–30° of slip).
  - Then 3 s of a controller: steer = s0 − K_s·(δ − 25°)/30°, throttle = t0 + K_t·(δ − 25°)/30°, over a grid of s0, K_s, t0, K_t (81 combinations).
  - Score: the longest spell with 12° < δ < 45°, speed > 2 m/s and yaw rate > 0.5 rad/s.

  | Variant | μ 1: ≥ 1 s / ≥ 2 s | μ 0.7: ≥ 1 s / ≥ 2 s |
  |---|---|---|
  | V0 pyramidal (heading 0) | 1 / 0 | 6 / 1 |
  | V1 elliptic | 0 / 0 | 7 / 1 |
  | V2 elliptic + slip drop | **12 / 1** | 11 / 0 |
  | V3 V2 + rolling drag | 12 / 4 | 11 / 1 |

- **What breaks it** (traces):
  - With the centre LSD and 50/50 torque, throttle spins all four wheels (wheel 9–10 m/s at 5–7 m/s).
  - A spinning tire's Coulomb force points along its slip velocity, which is almost along the wheel's own heading. The steered fronts then pull the car onto their heading, and δ decays from 24° to 1° in about 0.6 s while the car accelerates away.
  - Without throttle, the slide scrubs off speed and ends.
  - At μ 0.7 a 15–19° slide held for 2 s at 7 m/s: a realistic 4WD power slide.
  - So drifting a 4WD is *meant* to be hard, but on high grip nothing short of a spin holds an angle. Two things would help, both real truggy traits: a sliding-friction drop (V2), or a rear-biased torque split.
- **Catching slides** works, as the earlier review found: a 45° slide is caught in about 0.5 s.

### 5. Friction randomisation

- **Implementation:** one draw per episode scales tire and ground friction together (`env.py:157–158`). The max rule then gives the drawn value. That's as intended (the 2026-11-14 review fix).
- **Ranges** (`real.py:78`, `81`): train 0.7–1.1, gate 0.55–1.25.
  - Compared with real surfaces, 0.55 is loose or dusty dirt (car tires on gravel and dry earth: 0.55–0.68) and 1.1–1.25 is high-bite prepped clay. The nominal 1.0 is a good clay value.
  - The gate's wider band tests extrapolation, as in Phase 3. Both ranges are sensible.
- **Caveats:**
  - Under the pyramidal cone the *effective* μ also varies 0.71–1.0 with heading. So the train range really spans 0.50–1.1 and the gate 0.39–1.25 (0.39 is wet clay or ice). The elliptic fix makes the ranges mean what they say.
  - The top 10% of the gate range (μ > 1.18) is where the full-brake flip lives (section 2).
  - The chassis friction isn't scaled.
  - μ is one number for the whole lap. Real tracks vary: a groove and loose edges, a blue-groove line versus loose dust. A per-patch μ (say ±10% in 5 m cells) would teach probing grip, but that is a design choice, not a bug.

### 6. Numerics

- **Energy** (`t10_numerics.py`, energy flag on):
  - A 5 s spinning slide from 66 J: no step gains more than 1 mJ, and the gains total 4 mJ.
  - Free roll at 8 m/s, motor off: 8 → 7.26 m/s in 5 s; the gains total 27 mJ.
  - No energy gain.
- **Contact chatter on flat ground** (`t11_flatchatter.py`, 6 m/s):
  - The per-2 ms F_z on one tire has a standard deviation of 2.2–2.4 N (static 6.1 N). The control-step mean stays in 4.9–7.3 N, and the speed's standard deviation is 0.01 m/s.
  - Cause: the capsule crosses heightfield triangle edges (5 cm grid), so the contact count jumps between 2, 3 and 6. The 6-contact steps carry +40–90% load spikes, and one step had 0 contacts (a 0.03 mm hop).
  - Harmless for driving. It does inject vibration into the 500 Hz IMU samples at v / 5 cm (120 Hz at 6 m/s).
- **Rough ground** (teacher on loop 10000):
  - Off the jumps, 6.6% (nominal) to 8.2% (gate) of control steps have a tire averaging under 0.5 N for the step: wheels skipping over the roughness.
  - Off-jump peaks reach 6–7× static load per tire, and the chassis's 99th-percentile vertical acceleration is 1.0 g.
  - Plausible for rough ground with rigid tires. The contact (solref 0.02, 0.1–0.2 mm static penetration) is far stiffer than a real foam-insert tire (about 5–10 mm of deflection), so small bumps are passed straight to the wheel. Not prototyped.
- **Grid snagging:** none seen with capsule tires (speed steady on flat ground at 0, 30 and 45° to the grid). The earlier fix holds.

## Recommendations (ranked)

Prototype variants (`traction/variants.py`, monkeypatched onto `JumpCar`):
- V0: baseline.
- V1: elliptic cone.
- V2: V1 + tires at `priority=1` + sliding-friction drop (μ falls linearly by up to 25% as the patch slip speed goes from 0.3 to 1.5 m/s, updated every 2 ms in `_actuate`).
- V3: V2 + 0.04 rolling drag.
- B0.65: brake capped at 0.65 × motor torque.

Teacher numbers: `LoopExpert()` defaults on validation loops 10000–10019 × 3 starts. Gate figures are summed over phys_seed 0, 1 and 2 (180 runs).

| Variant | Nominal laps finished | Gate laps finished | Gate failures (rolled / off lane) | Lap time |
|---|---|---|---|---|
| V0 (repo) | 55 / 60 | 138 / 180 (77%) | 18 / 24 | 28.1–28.4 s |
| V1 | 54 / 60 | 135 / 180 (75%) | 25 / 20 | 28.1–28.2 s |
| V1 + B0.65 | 53 / 60 | 141 / 180 (78%) | 25 / 14 | 28.1–28.3 s |
| V2 | 54 / 60 | 130 / 180 (72%) | 10 / 40 | 28.1–28.3 s |
| V3 (phys_seed 0 only) | 58 / 60 | 47 / 60 | 5 / 6 (+2 stuck) | 28.3–28.7 s |

Lap outcomes are chaotic: a binomial standard deviation of about ±6 laps per 180. So V0, V1 and V1 + B0.65 are indistinguishable. V2 costs about 8 laps, mostly low-μ run-offs.

1. **Elliptic friction cone (bug fix).**
   - Where: `car.py:169`, `<option timestep=... integrator="implicitfast" cone="elliptic"/>`. impratio stays 1; 10 made no difference here.
   - To keep old runs reproducible, gate it on a physics switch (for example a new `"grip"` switch added to `TRUGGY_SWITCHES`, read in `model_xml`).
   - What changes for a learner:
     - grip is the same in every compass direction (0.71 → 1.0 μ on diagonals);
     - a true friction circle under combined slip;
     - steady grip 0.77 → 0.89 g;
     - the same trail-brake input gives the same rotation everywhere (β spread 15–61° → 20–25°).
   - Before / after on the harness: sideways slide at 45°, 0.72 → 1.01 g; full brake at a 45° heading, 0.73 → 0.99 g.
   - Risk to the teacher: none measurable (135 against 138 of 180 at gate; nominal 54 against 55). Compute +12% per physics step. Every prior number changes slightly, so re-baseline.
2. **Cap the ESC brake (flip cliff).**
   - Where: `env.py:463`, `u = -K_BRAKE * abs(u) * tanh(...)` with K_BRAKE = 0.65 (real ESCs have a brake-strength setting). Optionally randomise it 0.5–0.8 in `real.RANGES` as a setup knob.
   - Effect:
     - no front flip anywhere in the gate range (μ 1.25 at mass × 0.75: −7° against a flip);
     - full stop 0.89 g at μ 1 (was 0.99) and the same at low μ (traction-limited);
     - a full-brake turn-in becomes a usable trail-brake (yaw 2.86 rad/s, β 32°, instead of locking the fronts);
     - in the air, nose-down authority falls by about 35%.
   - Risk: low on the ground (teacher 141 / 180). The teacher's in-air PD loses some nose-down authority (re-check the landing rate; it was 150–157 landed, the same as the baseline).
   - Alternatives: cap the gate friction at about 1.15, or raise the droop travel (below).
3. **Sliding-friction drop (optional realism, for clay).**
   - Where: `car.py:119` tire geom `priority="1"` (so the tire's μ is used alone; set ground and chassis by the draw as now). Plus, in `env.py` `_actuate`, per step: `geom_friction[tires, 0] = μ_draw · (1 − 0.25·clip((v_slip − 0.3)/1.2, 0, 1))`, with v_slip the horizontal speed of the tire's bottom point (`mj_objectVelocity` of the wheel body).
   - Effect (V2):
     - power oversteer appears at μ 0.7 (full-throttle turn-in β 8° → 46°), consistent across headings;
     - a held drift becomes reachable at μ 1 (12 / 81 controllers ≥ 1 s, against 0);
     - a locked-wheel stop drops 0.99 → 0.78 g, so modulating beats locking, as on a real car;
     - a sliding front pushes more.
   - The 25% drop and 0.3–1.5 m/s band are guesses. Loose dirt barely drops, so randomise the drop 0–30% with μ, or tie it to μ (high-bite clay drops most).
   - Risk: medium. The teacher loses about 8 / 180 gate laps (more low-μ run-offs) and would need a_lat retuned down about 5%. Do it only once a learner is trying to drift.
4. **More droop, stiffer roll.**
   - Where: `car.py:111`, joint `range="-0.03 0.05"`, `springref="-0.03"`. Try range −0.045 to 0.05 with springref −0.045 (keeps the 2.4 cm sag, gives about 4 cm of droop), and an anti-roll term (a fixed tendon fl_susp − fr_susp and rl_susp − rr_susp with stiffness, about 100–150 N/m).
   - Effect: under 1 g of braking the rear keeps some load instead of topping out, so the brake rotation is progressive rather than all-or-nothing, and the flip margin grows. Roll drops from 8.6°/g to about 5°/g. Front/rear bar balance becomes a real handling knob only with load sensitivity (item 7).
   - Not prototyped. Risk: medium (landings and the ray tilt that bit Phase 4.5).
5. **Rolling resistance.**
   - The `0.005` in `friction="1.0 0.005 0.0001"` (`car.py:119`, `179`) is inert under condim 3: document it or delete it. condim 6 would make it active but costs solver time and makes rolling friction depend on heightfield contact count.
   - Instead, add a drag on each spin joint in `_actuate`: −C_rr·F_z·r·tanh(ω r / 0.2), C_rr about 0.04 on dirt (V3: coast-down 0.09 → 0.12 g, constant at low speed).
   - Risk: low (teacher nominal 58 / 60, gate 47 / 60 on seed 0, about equal). Small realism gain: the car slows when coasting at low speed.
6. **Scale the chassis friction with the draw.** `env.py:157`: `sel = np.append(self.tires, [self.ground, self.chassis])`, or use tire priority as in item 3. Low impact (scrapes and crashes), no risk.
7. **Load sensitivity (optional).** In the same `_actuate` hook: μ_tire = μ·(1 − 0.15·(F_z/F_z0 − 1)), from the previous step's contact force. It makes lateral transfer cost total grip, so roll stiffness and bars matter, and adds a little more trail-brake rotation. Not prototyped; low priority.
8. **Tire vertical compliance (optional).** Tire geom `solref="0.03 1"` or a wider solimp, to emulate a few mm of foam deflection and cut wheel skip on rough ground (6–8% of steps). Untested. It interacts with the 1 cm "on the ground" rule and with landing spikes.

**Suggested order:** 1 and 2 together (one physics switch, re-baseline the teacher: no retune needed going by these numbers). Then 6 (trivial). Then 3 when the drills reach drifting, with 4 and 5 alongside if the owner wants the car to feel like a setup-able truggy.

## Sources

- MuJoCo documentation: [Computation (friction cones)](https://mujoco.readthedocs.io/en/latest/computation/index.html), [Modeling (contact parameters, friction mixing, solver settings)](https://mujoco.readthedocs.io/en/latest/modeling.html)
- Tire friction tables: [hpwizard.com tire friction and rolling resistance](https://hpwizard.com/tire-friction-coefficient.html)
- 1:10-scale car friction: [Online Slip Detection and Friction Coefficient Estimation, arXiv 2509.15423](https://arxiv.org/html/2509.15423v3)
- Off-road tire curves: [Allen et al., A Vehicle Dynamics Tire Model for Both Pavement and Off-Road Conditions, SAE 970559](https://papers.sae.org/publications/technical-papers/content/970559/); [Coutermarsh & Shoop, Lateral Force versus Tire Angle Measurements for Various Field Conditions, SAE 2010-01-0374](https://saemobilus.sae.org/content/2010-01-0374)
- Not found: measured μ, slip curves or lateral g for RC tires on clay or dirt. Those targets are estimates and are marked as such.
