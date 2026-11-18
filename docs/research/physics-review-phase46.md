# Physics review for Phase 4.6: can the learner "work the car" like a real 1/8 truggy?

2026-11-18. A background review of the MuJoCo car (`sdc/jumps/car.py`, `env.py`, `real.py`) in the configuration Phase 4.6 trains with: `common.CAR` plus `angled=True` and `dr="gate"` (`real="all"`: wide tires, soft suspension, one motor through diffs, servo, battery, delay). No file under `sdc/` or `tests/` was edited. The tests ran on a flat 64 m heightfield (the loop terrain, flattened) at nominal physics unless stated. Scripts are in the session scratchpad (`physics/`). The prototype variants below monkeypatch the model; none of them is in the repo.

## Summary

| Behaviour | Sim (nominal, μ = 1) | Real 1/8 truggy (Froude-scaled to the sim's size) | Verdict |
|---|---|---|---|
| Scale | Wheelbase 0.30 m, track 0.28 m, tires 110 mm, 2.48 kg | 8IGHT-T / E-Revo: wheelbase 0.35–0.38 m, track 0.42–0.45 m, tires 140–160 mm, 4.4–5.1 kg | **A 1/10 car.** About 0.8× a 1/8 truggy in length. The mass is right for that size (4.5 kg × 0.8³ = 2.3 kg). The track is narrow: 0.28/0.30 = 0.93 of the wheelbase, against 1.1–1.3 on a truggy. |
| CG height and rollover threshold | CG 10.4 cm; t/2h = 1.35 g | 8IGHT-T about 2.2 g; Revo about 1.7 g (estimates) | Plausible, Revo-like. Never traction-rolls on flat ground (snap full lock to 7 m/s: 8–9° roll). |
| Launch / 0–8 m/s | 0.52 g for the first 0.5 s; 0–8 m/s in 2.4 s; **no wheelspin at μ ≥ 0.7** | Traction-limited (≈ 1 g) to well past jump speed; 0–30 mph in about 2 s (anecdotal) | **Too weak.** Peak power at the wheels is about 63 W (25 W/kg), against about 400 W/kg real (≈ 360 W/kg scaled). |
| Top speed | 11.7 m/s | 20–22 m/s (≈ 18–20 m/s scaled) | Low, but rarely binding on 3 m-radius loops. |
| Braking (full brake, straight line) | **Spins the car every time** (8 m/s: −95° to +30° of heading, 3 s, wheels driven backwards at 12–17 m/s) | 0.6–0.9 g, mostly straight (thick centre diff) | **Bug-level.** The brake is a reverse torque, so the rear wheels are spun backwards. |
| Steady cornering | Max 0.75–0.8 g (μ 1), 0.45 g (μ 0.55); strong limit understeer (yaw rate falls to 10–40% of kinematic, β ≈ 0) | 0.8–1.2 g on prepped clay, 0.5–0.8 g on loose dirt (estimate, no measured source found); understeer on power | Realistic balance; grip a little low. |
| Lift-off at full lock (7 m/s) | Spins 174° (neutral = full back-EMF braking) | Tucks in and rotates; a full spin only on slick ground | **Too strong** (neutral acts as a 100% drag brake). |
| Brake + steer + throttle entry (7 m/s) | Rotates (β up to 55°, 3.3 rad/s), then **bogs**: full throttle can't drive out (4.6 → 1.6 m/s; inside front wheel spins to 14 m/s) | Rotates 20–40° of slip, then powers out | **Half there.** The rotation comes from the brake bug. There's no power to hold a drift, and open diffs dump torque into the inside wheel. |
| Counter-steer catch | Works: β 45° caught in about 0.5 s | Works | Realistic. |
| Air: full throttle | +25° in 0.5 s from 7 m/s takeoff; **+11° from 10 m/s** (wheels already near the 12 m/s no-load limit) | Roughly ±60–120° over a 1 s jump (estimate from the wheel/body inertia ratio; the owner's 30–90° is the same order) | Weak, and it fades with speed. |
| Air: full brake | **−90° in 0.5 s** (−255°/s), because the wheels are driven into reverse | Stopping the wheels: about −100 to −250°/s, then nothing | **Too strong and wrong-shaped** (it keeps rotating). |
| Air: neutral | −33° in 0.5 s (back-EMF drag stops the wheels) | About zero (nitro clutch free-wheels; ESC drag brake 0–20%) | **Wrong.** |
| Wheel/body inertia ratio | 4 I_w / I_pitch = 6.2e-4 / 0.0445 = 0.014 | ≈ 0.03 (4 × 5.7e-4 / 0.08) | About **2× too light**. |
| Air: steering (gyroscopic) | Full lock at 7–10 m/s: 40–60°/s roll, 14–16° of roll in 0.6 s, −6° yaw reaction | Present; about 2× stronger per m/s (heavier wheels) | Present; about half strength. |
| Landing, 1 m flat drop | Survives; bump stop hit, 52 g spike, 4.6 cm bounce | Lands; bottoms on flat | Realistic-ish. |
| Landing, 1.5 m flat drop | Chassis hits, **30 cm rebound, 0.5 s airborne again**, 1.6 m/s left | Bottoms, small bounce, keeps driving | Too bouncy. |
| Landing, 1 m, 20° nose-up | **Flips onto its roof** | Slaps the rear and comes down | **Too harsh.** |
| Diffs | All three open (equal torque, back-EMF on the mean wheel speed) | 8IGHT-T: three oil diffs (2.0 stock 7k / 10k / 2k cSt); Revo: front and rear diffs, no centre diff (from memory) | **Missing limited slip.** The open centre diff causes the brake spin and the front-wheel spin at launch on low grip. |
| Termination rules | Rolled over: up_z < 0.2, checked mid-air too; stuck: v_fwd < 0.3 m/s for 2 s; off the lane: \|y\| > 1.5 m | — | Mostly sensible. "Rolled over" also fires mid-air at > 78° pitch, which a hard brake now reaches in 0.5 s. |

**Bottom line:** the sim supports in-air pitch control and gyroscopic roll, and catching a slide works. But the "brake hard, rotate, power out" corner needs three fixes before a learner can use it realistically:
1. a real brake that only stops the wheels and doesn't reverse them;
2. limited-slip diffs, at least in the centre;
3. a motor strong enough to spin the wheels on dirt.

With all three prototyped (variant V3b below), the car brakes straight at 0.7 g, does a brake-steer-throttle entry with 24° of slip and drives out at 8.3 m/s (baseline: 4.1 m/s, bogged), and gets +40° in 0.5 s of throttle in the air from 7 m/s (baseline +25°). Full brake in the air stays −45° in 0.5 s (baseline −90°, still turning).

## Real reference numbers

| Quantity | Traxxas Revo 3.3 / E-Revo 2.0 | Losi 8IGHT-T (RTR / 4.0) | Source / note |
|---|---|---|---|
| Mass | 4.45–4.69 kg (Revo 3.3, no fuel); 5.08 kg (E-Revo 2.0) | 4.4 kg (gas RTR); about 3.6 kg (4.0 kit, rolling) | Retailers, Traxxas, CompetitionX |
| Wheelbase | 348 mm (Revo 3.3); 380 mm (E-Revo 2.0) | 379–383 mm | Retailers, LiveRC |
| Width / track | 447 mm wide; track 416–445 mm | About 404 mm wide | Traxxas, Horizon |
| Tires | 6.3 in (160 mm) Maxx tires | 1/8 truggy tires 141–143 mm | AKA, HPI listings |
| Tire + wheel mass | About 180–220 g each (estimate; a 1/8 buggy tire alone is about 120 g) | Same | Listings plus estimate |
| CG height | About 110–130 mm (ground clearance 108 mm) | About 75–90 mm (ride height about 38 mm) | Estimate |
| Wheel travel | About 100+ mm | About 90–110 mm | Estimate |
| Top speed | 45+ mph claimed (Revo 3.3); 50 mph measured on 6S (E-Revo 2.0) | About 35–45 mph | Traxxas, Big Squid RC |
| Acceleration | 0–30 mph in about 2 s (anecdotal); traction-limited | Same | RCU forum |
| Braking | 0.6–0.9 g (traction-limited) | Same | Estimate |
| Lateral grip on clay/dirt | μ ≈ 0.8–1.2 prepped clay, 0.5–0.8 loose (estimate; no measured RC source found) | Same | — |
| Drivetrain | 4WD, front and rear diffs, no centre diff (from memory, not verified) | Three oil-filled gear diffs; 2.0 stock 7k front, 10k centre, 2k rear | Losi setup sheet via search |
| Steering | About 30–35° lock (estimate); servo 0.07–0.12 s/60° | Spektrum S6100 HV metal-gear servo | — |
| Motor | 2200 kV brushless on 6S (≈ 2–3 kW peak); TRX 3.3 nitro ≈ 1.1–1.5 kW (estimate) | .21–.28 nitro / .31 cc gas | Traxxas |

**Scaling.** Froude scaling (same g, length ratio λ = 0.8) maps a 1/8 truggy onto the sim's size. Speeds scale by 0.89, accelerations stay the same, angular rates scale by 1.12, mass by 0.51 and power by λ^3.5 = 0.46. So the sim car "should" weigh about 2.3 kg (it weighs 2.48), launch at about 1 g, top out at 18–20 m/s, and carry about 900 W of motor.

## Details

### a. Acceleration, top speed, braking

- **Launch.** Full throttle gives 5.1 m/s² over the first 0.5 s, and 0–5 / 8 / 10 m/s in 1.2 / 2.4 / 3.9 s.
  - At μ = 1 (and at gate-range 1.25) the wheels never spin. At μ = 0.55 the front wheels spin (7.6 m/s against 1.9 m/s of ground speed) and the car yaws 18° in the first second with the steering straight.
  - Cause: equal torque per wheel, the fronts unloaded by weight transfer, and an open centre diff.
- **Top speed** is 11.7 m/s (14 × 0.95 charge × (1 − 0.07 sag) minus losses).
- **Motor model.** `TAU_MAX = 0.25` N·m per wheel gives F0 = 18 N at the tires. That is 0.74 g at standstill, falling linearly to zero at 14 m/s (back-EMF). Peak power is F0 × v_top / 4 ≈ 63 W. A Froude-scaled truggy has about 900 W and is traction-limited to about 15 m/s.
- **Braking** (`throttle = −1`) applies −1 × the motor torque plus back-EMF drag, equally to all four wheels. Weight transfer (h/L = 0.35, Revo-like) unloads the rear, so the rear wheels stop and are then **driven backwards at full motor torque**. In the 6 m/s trace, the rear wheels reach −12 m/s while the car is still moving at +3 m/s.
  - A rear axle spinning backwards has almost no lateral grip, so the car swaps ends: heading −95° to +150° from 6–10 m/s, on every run.
  - A real ESC brake opposes rotation and fades to zero at standstill. A racing ESC has no reverse at all, and a hobby ESC needs a return to neutral before reversing. A nitro truggy has a disc brake and no reverse.
- **Effective friction** matches the setting: a sideways slide decelerates at 0.97–1.03 g with μ = 1 and 0.55–0.59 g with μ = 0.55. A locked-wheel slide gives 1.06 g. MuJoCo's Coulomb contact has no peak-then-drop slip curve: the force at 40° of slip equals the force at 10°. Slides are therefore easier to hold than on real dirt.

### b. Steady cornering

- Speed held with a P controller, steering 0.25 / 0.5 / 1.0 of lock, 2–8 m/s.
- **Maximum lateral acceleration:** 0.75–0.8 g at μ = 1, 0.40–0.50 g at μ = 0.55 (about 0.8 μ).
- **Below the limit:** yaw rate matches the kinematic rate (L = 0.30 m) within 1–8%, with a small rear slip angle (β 2–9°).
- **At the limit** the car ploughs: the yaw rate falls to 9–40% of the kinematic rate and β goes to 0–3°. Classic 4WD power-on understeer, which is right for a truggy.
- Body roll is 6–7° at the limit (3° at μ 0.55).
- Snapping to full lock at 7 m/s peaks at 1.3 g with 8–9° of roll and no rollover. Real high-grip clay sometimes traction-rolls a Revo, but not an 8IGHT-T, so this is fine.
- Steering reaches its full 22.9° under load (servo `kp = 3` is adequate).
- The teacher (`LoopExpert`, a_lat 3.5 m/s² = 0.36 g, a_brake 3 m/s²) drives at about half the car's grip. Nothing in the current data shows the limit.

### c. Brake, steer, throttle (7 m/s entry, μ 1, 2 s)

| Programme | Max β | Max yaw rate | Heading change | End speed |
|---|---|---|---|---|
| Full steer, throttle 0.6 | 10° | 2.5 rad/s | 132° | 4.9 |
| Full steer, lift | **174° (spin)** | 4.0 | — | 0.2 |
| Brake 0.3 s + steer, then full throttle + steer | 55° | 3.3 | 199° | **4.1 (bogged)** |
| Same, then throttle + counter-steer 0.6 | 47° | 3.6 | 7° (caught) | 6.3 |
| Brake 0.15 s + steer, then throttle | 9° | 2.1 | 116° | 6.8 |
| Brake straight, then steer + throttle | 4° | 1.8 | 106° | 6.4 |

- **Rotation is available**, but only through the reverse-spinning rear (the brake bug) or the 100% drag brake at neutral (lift).
- After rotating, full throttle doesn't drive the car out. Speed falls from 4.6 to 1.6 m/s while the **inside front wheel spins to 14 m/s** (open front and centre diffs; back-EMF on the mean wheel speed cuts torque to all four).
- Counter-steer catches a 45° slide in about 0.5 s.
- At μ 0.7 the same entry gives only 19° of slip (49° with half throttle). The rotation tool is fragile across the gate's friction range.

### d. In-air pitch authority (free flight, level takeoff, wheels at ground speed)

Pitch change after 0.5 s / 1.0 s:

| Command | From 7 m/s | From 10 m/s |
|---|---|---|
| Hold (cruise throttle) | +1° / +2° | 0° / +1° |
| Full throttle | +25° / +60° | +11° / +26° |
| Full brake | −90° (−255°/s) / beyond −90° | −76° / beyond −90° |
| Neutral | −33° / −81° | −47° / −65° |
| 0.1 s brake pulse, then hold | −23° (held) | — |
| 0.1 s throttle pulse, then hold | +9° | — |

- **Angular momentum budget.** Spinning the wheels up or down gives Δω_body = (4 I_w / I_pitch) × Δω_wheel = 0.014 × Δv / r.
  - Throttle can only add v_noload − v_takeoff (12.2 − 10 = 2 m/s at 10 m/s, hence the weak nose-up).
  - Brake currently removes v_takeoff **plus** 12 m/s of reverse spin. Hence a strong, long-lasting nose-down that goes past −90° within a second. The "rolled over" rule (up_z < 0.2) then ends the episode mid-air.
- **Real truggy.** Inertia ratio about 0.03 (2×) and wheel speed headroom of about 1.5–2× jump speed. Stopping the wheels gives roughly 100–250°/s of nose-down; full throttle gives a similar nose-up that builds more slowly. Over a 1 s jump that is enough to correct ±30–90° (the owner's figure).
- **Verdict:** nose-down is too strong in size and wrong in shape (it never runs out). Nose-up is too weak, and gets weaker with speed. Neutral wrongly acts as a brake. With the ESC brake fix (V1): brake −43° in 0.5 s (then it stops, since the wheels have stopped), neutral −7°, throttle unchanged. With more motor (V2/V3b): throttle +40° from 7 m/s and +23° from 10 m/s.

### e. In-air steering (gyroscopic)

- **Full lock held at 7–10 m/s:** roll rate peaks at 41–47°/s at 0.16 s, giving 14–16° of roll and −6° of yaw (reaction to steering the wheels) after 0.6 s. Centring at 0.3 s leaves 5–6° of roll.
- **Steer plus full throttle:** 21–26° of roll and −14° of yaw, since the front wheels' drive torque is now tilted.
- The physics is right (precession of the front wheels' spin). The size is about half a real truggy's per m/s: lighter wheels, though also a lighter roll inertia (0.021 kg·m²).
- Not a priority. Note that the expert holds the steering straight in the air for this reason.

### f. Landings (flat ground, 6 m/s forward, throttle 0.5)

| Drop | Level | 20° nose-down | 20° nose-up |
|---|---|---|---|
| 0.5 m | Clean (15 g) | Clean | Clean, touches bump stop |
| 1.0 m | Bump stop, 52 g, 4.6 cm bounce, OK | OK | **Chassis hit, flips onto its roof** |
| 1.5 m | **Chassis hit, 30 cm rebound, 0.5 s airborne, 1.6 m/s left** | Chassis hit, OK | **Flips** |
| 2.0 m | Chassis hit, bounces | **Flips** | OK |

- **Travel left from sag to the bump stop is 5.6 cm.** The spring (185 N/m per corner) stores about 0.3 J per corner, the damper (18 N·s/m) about 4 J. A 1 m drop brings 24 J. The rest hits the stiff joint limit (`solreflimit 0.005 1`) and then the chassis box.
- Bottoming on flat 1–1.5 m drops is realistic for a truggy. What isn't:
  - the 52 g spikes (the IMU clips at 16 g);
  - the pogo rebound;
  - the 20° nose-up 1 m flip. Real cars slap the rear and come down.
- Course drops land flat from kickers 0.3–0.6 m high (apex 0.6–1.2 m), so this matters.
- **Gate-range extremes** (spring 0.7×, damping 0.6×, mass 1.3×): 94 g peaks; the 1.5 m flat drop rebounds 26 cm.
- **Termination rules:**
  - Stuck (v_fwd < 0.3 m/s for 2 s) and off the lane are sensible. A 90° drift has v_fwd ≈ 0 but passes in well under 2 s.
  - Rolled over (up_z < 0.2) is sensible on the ground. In the air it kills a recoverable rotation past 78°. Suggest applying it only after 0.3 s with wheels or chassis on the ground, or after landing.

### g. Diffs

- The `drive` switch is a fixed tendon with coefficient 0.25 per wheel and the motor on it. That is mathematically three ideal **open** diffs with a massless motor: equal torque to every wheel, motor speed = mean wheel speed, back-EMF on the mean.
- Consequences measured:
  - inside front wheel spin at corner exit (+8 m/s overspeed, exit +2.3 m/s in 1 s against +3.6 m/s straight);
  - front-wheel spin and yaw wander at launch on low grip;
  - rear lock under brake (with the reverse-torque bug, a spin).
- Neither reference car is all-open. The 8IGHT-T runs thick oil diffs, especially in the centre (10k). The Revo has no centre diff at all (front and rear axles coupled), if memory serves.
- **Prototype:** viscous couplings as damped fixed tendons, i.e. torque = c × speed difference.
  - Centre c = 0.03 N·m·s/rad on (front mean − rear mean); front 0.002 on (fl − fr); rear 0.001 on (rl − rr).
  - Centre c ≥ 0.03 makes full-brake stops straight (heading change 1–13° from 8 m/s at μ 0.7–1; c = 0 gives 105–128°, c = 0.01 gives 26–38°).
  - c = 1.0 went numerically unstable (QACC NaN), so stay below about 0.1.

### Earlier reviews' fixes still hold

- The Phase 3 sim review was of the 2D racer (`sdc/racer/dynamics.py`), so its findings don't carry over directly. In MuJoCo a sliding tire's force opposes the contact velocity by construction (its finding 2), and wheelspin is modelled (finding 3), though rarely reached.
- The Phase 4 / 4.5 fixes are intact:
  - tire margin and gap are 0 (tires rest 0.2 mm into the ground, not floating);
  - "on the ground" is measured by distance;
  - chassis contact is checked every 2 ms in the air and the landing window;
  - randomised friction scales the ground too (checked: tire = ground = 1.157 on a gate draw).

## Recommendations, ranked by impact on "can a learner work the car"

Prototype results are on flat ground at μ 1 unless stated (variants: V0 baseline; V1 ESC brake; V2 V1 + motor; V3b V2 + diffs; V5 V3b + wheel inertia).

1. **Make the brake a brake (bug fix).** In `JumpCar._actuate` (env.py), replace `ctrl = u` for u < 0 with a torque that opposes the motor's rotation and fades near standstill, with no reverse:
   - `f = -k_brake*|u|*tanh(ω_motor·r/0.3 m/s)`;
   - neutral (|u| < 0.05) = drag brake 10% of back-EMF instead of 100%;
   - back-EMF moves from the tendon `damping` in `car.py` (set it to 0 under `drive`) into `_actuate`: `f = b·u − v_motor/TOP_SPEED` for u > 0.
   - Optional: reverse only after 0.3 s at neutral while stopped.
   - **Effect (V1):** lift at full lock no longer spins (β 14° against 174°); in-air brake −43° in 0.5 s and then stops, against −90° and still turning; in-air neutral −7° against −33°. Straight braking still spins without item 2.
   - **Risk:** high for existing results. The teacher's in-air PD (`driver.py`, kp 4 / kd 0.5 around the takeoff throttle) and every trained student rely on brake meaning reverse torque and neutral meaning a strong brake. Retune the teacher's air gains (nose-down authority halves) and re-collect.
2. **Limited-slip diffs.** In `car.py` under `drive`, add damped fixed tendons:
   - `dcentre` (0.5·fl + 0.5·fr − 0.5·rl − 0.5·rr), damping about 0.03;
   - `dfront` (fl − fr), about 0.002;
   - `drear` (rl − rr), about 0.001.
   - Randomise the centre value 0.5–2× in `real.RANGES` (it is the setup knob real drivers change).
   - **Effect (V1b = V1 + diffs):** full brake from 8 m/s stops straight (heading 0–1° against a spin), 0.68 g; inside-wheel overspeed at exit drops from 8.2 to 2.2 m/s.
   - **Risk:** medium. Corner exit and in-air behaviour change little. Brake and launch behaviour change.
3. **A motor that can spin the wheels on dirt.** Set `car.TAU_MAX` 0.25 → 0.6 N·m per wheel (F0 ≈ 44 N, 1.8 g) and `TOP_SPEED` 14 → 16 m/s; 0.75 / 18 would be closer to Froude scale.
   - **Effect (V3b = items 1–3):** 0–8 m/s in 1.2 s (against 2.4); max lateral 0.82 g. Brake-steer entry then full throttle gives β 24° and exits at 8.3 m/s (against 4.1, bogged). Counter-steer exit at 9.9 m/s. In-air throttle +40° / +23° in 0.5 s from 7 / 10 m/s (against +25° / +11°).
   - The learner now has to modulate throttle (realistic) and gets power-oversteer as a tool.
   - **Risk:** high. Every throttle-to-speed mapping changes (teacher `throttle = 0.8 e + 0.45` and its jump speeds). The teacher's 0.45 feed-forward would need rescaling by about 1/2.4. Existing students are invalid.
4. **Heavier spinning parts.** Add `armature="1.5e-4"` to each `{w}_spin` joint (wheel plus drivetrain inertia: about 2× spin inertia, ratio to body pitch 0.014 → 0.028, matching a truggy).
   - **Effect (V5 used 3×):** throttle and brake in the air become steadier (less sensitive to timing), gyroscopic roll about doubles, and the inside wheel flares less (overspeed 6.4 → 3.8 m/s). With 3× the in-air nose-up got slightly weaker per 0.5 s (+28° against +40°), because the same torque now spins up more inertia. 2× is the realistic choice.
   - **Risk:** low to medium (slower wheel response; teacher air gains).
5. **Landings: more usable travel and a softer bump stop.**
   - `SOFT_K` / sag: keep about 30% sag, but raise the bump end of the joint `range` from 0.05 to 0.07 (and move `MOUNT_Z` or the chassis box up by 2 cm so the box clears).
   - Make `solreflimit` 0.005 → 0.02 (a progressive bump stop).
   - Raise `SOFT_C` 18 → about 25–30.
   - Target: a 1.5 m flat drop bottoms without a 30 cm rebound, and a 20° nose-up 1 m landing doesn't flip.
   - **Risk:** medium. It was the soft suspension that broke Phase 4.5's learned car (ray tilt), and the "pogo kick" that set `SOFT_C`. Re-run the expert's landing rate before adopting it.
6. **Termination: "rolled over" only on the ground.** Apply `up_z < 0.2` only when the car is not airborne (for example `not self.airborne` and 0.2 s after touchdown), so the learner can explore large in-air rotations and be judged on the landing. Low risk; it slightly changes episode endings.
7. **Tire slip curve (optional, larger work).** MuJoCo's Coulomb contact has no grip drop past the peak. Real dirt tires lose about 20–30% once sliding, which is what makes a drift a balancing act.
   - One way to get it: lower sliding friction once a tire's slip speed exceeds about 1 m/s (per-step `geom_friction` from contact velocity in `_actuate`).
   - Medium risk. Do it only once items 1–3 are in and a learner is trying to drift.
8. **Steering lock (minor).** `STEER_MAX` 0.4 rad (23°) against about 30–35° on a truggy. It limits tight hairpins at low speed, not drifting.

**Suggested order:** do items 1 and 2 together (they fix a bug and are cheap), re-tune the teacher, then item 3, then 4 and 5. Each needs the teacher's landing and lap rate re-baselined on the validation loops before training students. Gate numbers from before the change won't be comparable.

## Sources

- Traxxas Revo 3.3: [AMain listing (TRA53097)](https://www.amainhobbies.com/traxxas-revo-3.3-4wd-rtr-nitro-monster-truck-w-tqi-2.4ghz-traxxas-link-wireles-tra53097/p357466), [Traxxas 53097-3](https://traxxas.com/revo-33-53097-3), [AMain TRA5309](https://www.amainhobbies.com/traxxas-revo-3.3-4wd-rtr-nitro-monster-truck-w-tqi-docking-base-telemetry-tra5309/p148155), [RC Visions](https://rcvisions.com/products/revo-3-3-4wd-1-10-nitro), [Revo manual (rctalk)](https://www.rctalk.com/resources/revo/reveo-manual.pdf)
- Traxxas E-Revo 2.0: [Big Squid RC review (50 mph measured)](https://www.bigsquidrc.com/traxxas-e-revo-2-0-review/), [RC Newb spec list](https://new.rcnewb.com/traxxas-reboots-the-e-revo/), [Traxxas 86086-4](https://beta.traxxas.com/86086-4-e-revo)
- Losi 8IGHT-T: [LiveRC 8IGHT-T 4.0](https://www.liverc.com/news/new-team-losi-racing-8ight-t-40-truggy-kit/), [Horizon 8IGHT-T 4.0 kit](https://horizonhobby.com/product/1-8-8ight-t-4.0-4wd-nitro-truggy-race-kit/TLR04005.html), [CompetitionX 8IGHT-T gas RTR review](https://www.competitionx.com/?p=2618), [Big Squid RC gas 8IGHT-T review](https://www.bigsquidrc.com/losi-gasoline-8ight-t-rtr-truggy-review/), [8IGHT-T 4.0 manual setup page](https://manualslib.mx/manual/262902/Team-Losi-Racing-8Ight-T-4-0.html?page=65), [RCTech 8T 2.0 setup threads](https://www.rctech.net/forum/nitro-off-road/311723-8t-2-0-setup-sheets-post-here-3-printthread.html)
- Tires: [AKA EVO Gridiron truggy tire](https://store.hobbyetc.com/parts/view/104996), [HPI Voodoo truggy tire 143 mm](https://www.hpiracing.com/ko/part/160292)
- Acceleration (anecdotal): [RCUniverse Hyper 7 thread](https://www.rcuniverse.com/forum/rc-1-8th-scale-buggies-244/4728242-hyper-7-a.html)
- Not found: measured μ or lateral g for RC tires on clay, or measured in-air rotation rates. Those figures here are estimates from the inertia ratio and are marked as such.
