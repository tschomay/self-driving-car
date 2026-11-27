# In-air pitch control on the truggy physics (side study, Phase 4.7)

Question (owner): throttle and brake barely change the car's pitch in the air. Is the sim's air control too weak?

Short answer: **yes, by about 2× (likely 1.5 to 3×).** The mechanism is real physics. The size is set by how much spin inertia the chassis can push against, and that is too small. The Phase 4.6 "heavy" switch was meant to double it, but it uses MuJoCo *armature*, which the chassis never feels.

Script: `sdc/phase47/air_control.py` (single process, about 3 s). Nothing in the sim or its defaults was changed.

## 1. How the sim does it

- **Physics only.** Nothing is scripted for the air: no airborne branch in the actuators, no pitch-rate clamp, no extra angular damping on the chassis. The only airborne-specific code is the per-2 ms chassis-contact check.
- **Path from throttle to pitch (truggy preset):** `env.JumpCar._actuate` → one motor on a fixed tendon to the four `*_spin` hinge joints (`drive`, gear 4 × 0.6 N·m) → the wheels' spin changes → the chassis turns the other way, because angular momentum is conserved. Spinning the wheels up tips the nose up, and braking them tips it down.
- **Motor (`power` + `esc` + `battery`):** u = 0.95 × (1 − 0.07) × throttle − w, where w = wheel speed / 16 m/s. That is 2.1 N·m at the wheels when stalled, and the wheels stop gaining speed at **14.1 m/s** (no-load).
- **Brake (`esc`):** −0.65 × 0.88 × tanh(ω/0.3) of the motor torque, about 1.4 N·m in total. It stops the wheels and does not drive them backwards. Neutral is a light drag brake (−0.1 w).
- **Other drags:** the limited-slip diff tendons (`lsd`) damp only *differences* between wheel speeds, so they do nothing when all four wheels spin together. The spin joints have no damping (1e-5).
- **Wheel inertia:** each wheel is a 0.12 kg capsule of radius 5.5 cm, so its spin inertia is **1.56e-4 kg·m²**, like a solid ball (I/mr² = 0.43). `heavy` adds `armature = 1.5e-4` to each spin joint.
- **Armature does not turn the chassis.** MuJoCo adds armature only to the joint's own diagonal entry of the mass matrix. It is not coupled to the free joint, so it absorbs motor torque without any reaction on the chassis. Measured: chassis rate change / wheel spin change = **0.0133**. That matches the wheel bodies alone (0.0133), not bodies plus armature (0.0261). So `heavy` makes the wheels slower to spin up, and adds **no** air authority. The Phase 4.6 review assumed it took the ratio from 0.014 to 0.028.

## 2. Measured (truggy, nominal physics, level launch, command held)

Free flight with no ground in reach. The wheels start rolling at the takeoff speed, and the command acts with no latency. Pitch is the integral of the pitch rate (nose up > 0). In the existing logs, air time is 0.44 to 0.70 s (10th to 90th percentile, median 0.58 s; `runs/p46-e10-b-avg.json`).

| Takeoff | Command | Pitch at 0.25 / 0.6 / 1.0 s | Rate at 0.6 s | Wheels: start → end (time to 90%) |
|---|---|---|---|---|
| 6 m/s | full brake | −17° / −46° / −79° | −83°/s | 6 → 0 m/s (0.10 s) |
| 6 m/s | neutral | −2° / −9° / −23° | −28°/s | 6 → 3.1 m/s (slow drag) |
| 6 m/s | full throttle | +14° / +50° / +95° | +110°/s | 6 → 14.1 m/s (0.36 s) |
| 7 m/s | full brake | −19° / **−53°** / −92° | **−97°/s** | 7 → 0 m/s (0.12 s) |
| 7 m/s | neutral | −2° / −10° / −27° | −32°/s | 7 → 3.6 m/s |
| 7 m/s | full throttle | +13° / **+44°** / +83° | **+96°/s** | 7 → 14.1 m/s (0.36 s) |
| 8 m/s | full brake | −21° / −59° / −104° | −111°/s | 8 → 0 m/s (0.14 s) |
| 8 m/s | neutral | −2° / −12° / −30° | −37°/s | 8 → 4.1 m/s |
| 8 m/s | full throttle | +11° / +38° / +71° | +82°/s | 8 → 14.1 m/s (0.36 s) |

- **Pulses at 7 m/s** (then neutral), pitch at 0.6 s: throttle for 0.1 s gives +15°, for 0.2 s +29°; brake for 0.1 s gives −46°, for 0.2 s −53°.
- **The wheels saturate quickly.** Brake stops them in 0.10 to 0.14 s, and after that it can do nothing more. Throttle reaches the 14.1 m/s no-load limit in about 0.36 s. The chassis rate then stays constant (at 13.9°/s per m/s of wheel speed change) until landing.
- **The rate fades with speed:** throttle headroom is 14.1 − v_takeoff (8 m/s at 6 m/s, 6 m/s at 8 m/s). Brake gains with speed.
- **"Barely" is partly how the drivers use it.** Held full commands give ±45 to 53° over a typical flight, which is visible. Small throttle corrections around the takeoff throttle (the teacher's PD) give much less. It's worth checking what the trained cars actually command in the air before blaming only the physics.

## 3. Real 1/8 truggy (estimates)

| Quantity | Real truggy (estimate) | Sim truggy |
|---|---|---|
| Mass | about 4 kg [3][4] | 2.48 kg |
| Tyre outer radius | about 72 mm (143 mm OD [1]) | 55 mm |
| Wheel + tyre + insert mass | about 0.2 kg (wheel 65 g [2], tyre + foam about 0.14 kg, estimate) | 0.12 kg |
| Spin inertia per wheel (coupled) | **5 to 8e-4 kg·m²** (tread far out: I/mr² ≈ 0.65) | 1.56e-4 |
| Chassis pitch inertia (whole car, about its COM) | **0.06 to 0.09 kg·m²** (0.595 m long, 0.38 m wheelbase [3]) | 0.047 |
| 4 I_w / I_pitch | **≈ 0.037 (0.025 to 0.05)** | **0.0133** |
| Pitch rate per m/s of wheel speed change (ratio / r) | **≈ 30°/s (20 to 40)** | 13.9°/s |
| No-load wheel speed in the air | about 20 m/s (4S, 2000 kV, about 11:1 overall, estimate) | 14.1 m/s |
| Full brake from 7 m/s | wheels stop in under 0.1 s; **≈ −210°/s (−140 to −280)**; about −110° (−80 to −160) by 0.6 s | −97°/s; −53° |
| Full throttle from 7 m/s | 13 m/s of headroom; **≈ +200 to +400°/s** once spun up | +96°/s; +44° |

How the estimates are worked out: angular momentum is conserved, so Δω_chassis = (Σ I_w / I_pitch) × Δv_wheel / r. The motor rotor and centre diff in a 1/8 truggy spin about the car's long axis, so they twist roll, not pitch. Only the wheels, axles and diff ring gears count for pitch. Every real figure here is an estimate from rough part numbers. I found no measured in-air rotation rates.

## 4. Verdict and proposal

**Too weak by about 2.1× per m/s of wheel speed change** (ratio about 2.8× too small, partly offset by the sim's smaller wheels). Throttle is also short of headroom, about 1.9× less than a real car. Causes, most important first:

1. **The wheels' coupled spin inertia is too small** (1.56e-4, a solid ball). A real tyre carries its mass at the tread.
2. **`heavy` uses armature, which has no reaction on the chassis.** It lengthens spin-up by 2× and adds nothing in the air.
3. **Throttle runs out of headroom at the 14.1 m/s no-load speed** (the effect fades for fast takeoffs).
4. Not a cause: there is no air-specific clamp or damping, and the brake does act on free wheels.

**Proposal: a new opt-in switch (for example `wheels`), with defaults unchanged.** Give each wheel body an explicit inertial with a coupled axle inertia of **3.4e-4 kg·m²** (range 2.5 to 4.5e-4): `<inertial pos="0 0 0" mass="0.12" diaginertia="1.7e-4 3.4e-4 1.7e-4"/>` (axle = body y). Keep the armature as the motor/drivetrain's reflected inertia. Tried on a private copy of the model at 7 m/s:

| Variant (7 m/s) | Brake: at 0.6 s / rate | Throttle: at 0.6 s / rate | Wheel spin-up (time to 90%) |
|---|---|---|---|
| As is | −53° / −97°/s | +44° / +96°/s | 0.36 s |
| **Axle inertia 3.4e-4, armature kept** | **−106° / −209°/s** | **+80° / +193°/s** | 0.54 s |
| The same plus brake cap 1.0 | −113° / −209°/s | +80° / +193°/s | 0.54 s (brake 0.12 s) |
| Armature moved into the wheel bodies | −102° / −188°/s | +86° / +186°/s | 0.36 s |

The proposal lands in the middle of the real brake estimate and at the low end of the real throttle estimate. Raising the no-load speed to about 20 m/s would close the throttle gap, but it also raises top speed on the ground. Leave that for a separate decision.

**What could break (every trained car learned the current physics):**
- **Every trained policy and the teacher will over-rotate.** Air authority doubles, so the teacher's air gains (kp 6, kd) need about halving, and students need retraining or fine-tuning. Expect landing rates to drop until then.
- **"Rolled over" mid-air:** the rule (up_z < 0.2, more than about 78°) fires mid-air. Full brake now reaches it in about 0.45 s. Apply it only on the ground (the Phase 4.6 review's item 6) before training with the new switch.
- **On the ground:** the effective rolling mass rises from about 0.40 to 0.65 kg (car 2.48 kg). Acceleration and braking drop about 8%. Gyroscopic roll from steering in the air about doubles. Landing with a wheel-speed mismatch lurches harder.
- **Gate comparability:** results with the new switch aren't comparable to earlier gates. Keep it opt-in, and re-baseline the teacher on it.

## Sources

1. HPI Voodoo 1:8 truggy tyre: 143 mm overall diameter, for 4.0" wheels. https://www.hpiracing.com/ko/part/160292
2. Avid RC Truss 4.0" 1/8 truggy wheel: 65.4 g each. https://www.amainhobbies.com/avid-rc-truss-4.0-1-8-truggy-wheels-4-black-avd1109-b/p1439582
3. RC Car Action truggy spec block: 4000 g, wheelbase 375 to 385 mm, length 595 mm. Search-result snippet only, model not confirmed (the page was blocked in this session). https://www.rccaraction.com/?p=269264
4. Tekno ET48 2.0 kit: 9 lb shipping weight. https://beachrc.com/tekno-rc-et48-2-0-1-8th-4wd-competition-electric-truggy-kit

The tyre + foam mass, the gearing, the motor figures and the pitch inertia are my estimates, not sourced.
