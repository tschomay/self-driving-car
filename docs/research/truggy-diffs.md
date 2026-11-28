# Truggy differentials: how limited are 1/8 viscous diffs, and how to model them in MuJoCo

2026-11-28. Background research for the `size8` truggy (`sdc/jumps/car.py`, `drive` + `lsd` switches). Three questions:
1. What diff oils do 1/8 electric truggies run?
2. How much locking torque does that oil make?
3. How should the diffs be built in MuJoCo?

No file under `sdc/` or `tests/` was edited. The estimate script (`diffest.py`) and a MuJoCo prototype of the rotor joint (`rotor.py`) are in the session scratchpad.

## Summary

**Stock oils (front / centre / rear, cSt).** From kit manuals:

| Kit | Oils | Source |
|---|---|---|
| Tekno ET48 2.0 | 20k / 20k / 10k | manual |
| Team Associated RC8T3.1e and RC8T3.2e | 10k / 10k / 5k | manual |
| Losi 8IGHT-T 2.0 | 7k / 10k / 3k | manual, seen only as a search snippet |

The typical 1/8 truggy runs **about 10k front, 10k centre, 5k rear**. Thicker oil in front than rear, and the centre about as thick as the front.

**Physics.** Each diff is a bevel-gear diff filled with silicone oil. The locking torque is roughly linear in the speed difference between its outputs (a viscous coupling). It is weak:

- A 10k axle diff gives about **0.002 N·m·s/rad** of wheel-referred damping. The range is 0.0008–0.006, and it is an estimate.
- So a 50 rad/s (3.6 m/s) wheel speed difference across an axle is resisted by about 0.1 N·m.
- The motor gives each wheel 1.25 N·m.

The centre diff sits behind the axle-diff ring and pinion (40/9 to 46/10, so a ratio of 4.4–4.6). Its wheel-referred damping is therefore about 20× larger for the same oil.

**The repo's current numbers (`LSD = centre 0.03, front 0.002, rear 0.001`) already fall inside the estimated range for 10k / 10k / 5k.** The centre value sits a little low against the midpoint.

**Recommended coefficients.** Wheel-referred, N·m·s/rad, for the tendons exactly as `car.py` defines them: `dfront` = fl − fr, `drear` = rl − rr, `dcentre` = front mean − rear mean.

| Diff | Reference oil | Recommended `damping` | Plausible range (estimate) | Tekno 20k/20k/10k variant |
|---|---|---|---|---|
| front | 10k | **0.002** (keep) | 0.0008 – 0.006 | 0.004 |
| centre | 10k | **0.04** (was 0.03) | 0.015 – 0.12 | 0.09 |
| rear | 5k | **0.001** (keep) | 0.0004 – 0.003 | 0.002 |

- Randomise each diff by about ×0.5–2 (log-uniform). The range covers:
  - oil choice (3k–20k);
  - shear thinning (−20 to −50% at high slip);
  - warm-up (viscosity falls about 1.5%/°C);
  - the geometry uncertainty.
- Keep the damping linear. MuJoCo's polynomial tendon damping could add shear thinning with a negative `b`, but that force peaks and then reverses. Don't.

**Model structure.**

1. **The present drive tendon is already an exact open-diff tree** (three ideal open diffs with massless gears and rotor). The brief described it as "not a kinematic diff", but it is one:
   - The drive tendon's length is 0.25·Σq, so its velocity, the motor speed, is the mean wheel speed.
   - The tendon force maps to 0.25·f on every wheel.
   - Ideal open bevel diffs with massless internals do exactly this. Torque splits equally whatever the wheel loads, and a lifted wheel keeps receiving its quarter while it spins up.
   - "Limited by the lowest-reaction wheel" emerges only through motor speed: back-EMF on the mean speed (already in `env._actuate`, `esc`) and rotor inertia (missing).
   - A carrier body with equality constraints would reproduce the same equations, only softer and with extra DOFs. It isn't needed for the diff itself.
2. **Missing: the motor's reflected inertia.** The rotor turns about 14× faster than the wheels. Its inertia referred to the mean wheel speed is about **1.5e-3 kg·m²** (estimate, range 0.7–2.5e-3). The four wheels' spin inertia totals 2.6e-3. Two ways to add it:
   - (a) Cheap: `armature="1.5e-3"` on the `drive` tendon. It adds inertia to the common mode only, not to the wheel-difference modes. But its reaction torque lands on the wheel axles (pitch), while the real rotor spins about the car's long axis (roll).
   - (b) Physically right: an explicit **rotor body**, a hinge about chassis x, coupled to the drive tendon by an `<equality><tendon tendon1="rotor" tendon2="drive" polycoef="0 N 0 0 0"/>`, with the motor actuator on the rotor joint. This gives torque roll on throttle and rotor gyroscopics. Prototyped in MuJoCo 4.0.1: the ratio holds to 1e-9 with `solref="0.004 1"`.
   - A **joint** equality can't do this, because it couples only two joints. A **tendon** equality can, because the drive tendon already averages the four wheels.
   - Either way, move the drivetrain part of the `heavy` switch's per-wheel armature into this term.
3. **Diff damping stays on fixed tendons,** with the coefficients above.
   - For the centre diff, the spider-gear speed is ratio × (front mean − rear mean). So wheel-referred c = internal c × 4.44² ≈ 20 × internal c.
   - For the axle diffs, the outdrives turn with the wheels (1:1), so no ratio applies. The ½ from side-gear-relative speed is folded into the estimate.
   - Optional: a small `frictionloss` (about 0.005–0.02 N·m, estimate) on each diff tendon for gear and seal friction. It gives a little lock at low slip, where the viscous term is near zero.

## Evidence

### 1. Stock diff oils (primary: kit manuals)

| Car | Front | Centre | Rear | Source |
|---|---|---|---|---|
| Tekno ET48 2.0 (1/8 e-truggy) | 20,000 cSt | 20,000 | 10,000 | Build step: "Fill with #20,000 CST oil FRONT / #10,000 CST oil REAR"; setup page: "We recommend 20k in the front diff … 20k in the center diff … 10k in the rear diff"; kit setup sheet 20k / 20k / 10k. [ET48 2.0 manual REV4](https://www.teknorc.com/wp-content/uploads/2021/04/ET482.0_instructions-REV4.pdf) |
| Team Associated RC8T3.1e | 10,000 (#5455) | 10,000 | 5,000 (#5453) | Build steps and kit setup sheet ("Front 10k / Center 10k / Rear 5k"). [RC8T3.1/T3.1e manual](https://img2.associatedelectrics.com/pdf/cars_and_trucks/RC8T3.1e/Team/manual-RC8T3.1-T3.1e.pdf) |
| Team Associated RC8T3.2(e) | 10,000 | 10,000 | 5,000 | Build steps. [RC8T3.2e Nitro Combo manual](https://img2.associatedelectrics.com/pdf/cars_and_trucks/RC8T3.2/Team/RC8T3.2e-Nitro-Combo-Manual-11-15-2021.pdf) |
| Team Associated RC8T3 / T3e, V2 diffs | 10,000 | 10,000 | 7,000 | [RC8T3/T3e V2 diff supplement](https://img2.associatedelectrics.com/pdf/cars_and_trucks/RC8T3/RC8T3-T3e-V2-Diff-Supplement.pdf) |
| Losi 8IGHT-T 2.0 RTR | 7,000 | 10,000 | 3,000 | Operation manual p. 8, read only as a search snippet (manualslib is not reachable from this session). [manualslib](https://manualslib.com/manual/739267/Team-Losi-8ight-T-2-0-Rtr-Truggy.html?page=8). An earlier repo review quoted 7k / 10k / 2k for the same car ([physics review](physics-review-phase46.md)). Unverified. |
| Losi / TLR 8IGHT-XTE | — | — | — | No manual found. TLR diff fluids listed as fitting the 8IGHT-X line ([HobbyEtc 7k](https://store.hobbyetc.com/parts/view/87157)). |
| Kyosho MP10Te | 5k? | 5k? | 3k? | No manual found. Forum posts say the MP10 / MP9e manuals suggest 5k / 5k / 3k ([RCTech MP10 thread](https://www.rctech.net/forum/nitro-off-road/1032506-official-kyosho-mp10-thread-post15403995.html), [MP9e thread](https://www.rctech.net/forum/electric-off-road/1055463-kyosho-mp9e-evo.html)). Unverified. |
| Mugen MBX8T / MBX8T Eco, HB E8T | — | — | — | No manual or setup sheet found from this session. |

- Gear ratios from the same manuals:
  - Tekno ET48 2.0: ring 40T and diff pinion 9T (4.44), spur 44T. The Tekno gearing guide recommends 12–16T motor pinions for the ET48. Overall reduction: 44/14 × 4.44 ≈ **14** (12.2–16.3).
  - Associated: ring 46T and diff pinion 10T (4.6), spur 46T.
- Tekno's tuning text: "Changing center diff oil affects the front-to-rear drive balance". Thicker centre oil gives more on-power steering and acceleration, and less off-power steering. ([ET48 2.0 manual](https://www.teknorc.com/wp-content/uploads/2021/04/ET482.0_instructions-REV4.pdf); [Tekno diffs and gearing sheet](https://www.teknorc.com/wp-content/uploads/2014/03/Differentials-and-Gearing1.pdf)).
- The diff fluid line goes from 1,000 to 1,000,000 cSt ([AE parts list](https://img2.associatedelectrics.com/pdf/cars_and_trucks/RC8T3.1e/Team/manual-RC8T3.1-T3.1e.pdf)). So 5k–20k sits near the open end: a "locked" diff runs about 100× thicker oil.

### 2. Torque against speed difference (estimate)

No bench measurement of RC diff torque against slip speed was found. Searches turned up only a commercial "diff analyzer" with no published units. The numbers below are derived from viscosity and estimated geometry.

**Oil.**
- PDMS (polydimethylsiloxane) silicone, density 0.975 at 10k cSt. So 10,000 cSt ≈ 9.75 Pa·s.
- Viscosity-temperature coefficient 0.61, meaning viscosity falls 61% from 38 to 99 °C, about 1.5%/°C ([Dow XIAMETER PMX-200 data sheet, Table 1](https://www.galindberg.se/storage/AC2B4499EE08763D9A209A5DE85530A61D7ECB8C1A3CFACD20215DDEB3938B48/d05ea6456215409586fe061ab5ae3127/pdf/media/ab7fe11104db4410b829991a8eff5db3/DOW_XIAMETER_PMX_200_Silicone_Fluids.pdf), distributor mirror). RC diff fluids are assumed to be PDMS of this kind; brands don't publish their base fluid.

**Mechanism.** Take an axle diff with outputs ω_L and ω_R:
- The carrier turns at their mean.
- Each side gear turns at ±Δω/2 relative to the case.
- Each spider gear spins on its pin at (z_side/z_spider)·Δω/2, with z_side/z_spider ≈ 1.6.

The oil shears in the thin gaps between moving parts and the case, and it is pumped through the gear meshes. Power matching gives the damping on the tendon Δω = ω_L − ω_R:

c_Δ = Σ c_i·(ω_i/Δω)²

Here c_i is each surface's viscous coefficient, with T = c_i·ω_i:
- disc face: c = πμ(r_o⁴ − r_i⁴)/(2h);
- journal: c = 2πμr³L/h.

**Geometry (estimate, 1/8 diff with a case of about 30 mm inside diameter).**
- Side-gear back face: r 4–9 mm.
- Side-gear hub in the case bore: r 4 mm, 6 mm long.
- 4 spiders on cross pins: r 2.5 mm, 5 mm long.
- Spider back faces: r 2.5–5 mm.
- Face gaps h = 0.2–0.5 mm, journal gaps 0.03–0.1 mm.
- Gear-mesh pumping: an unknown multiplier of 1–5× the shear terms. This is the largest uncertainty.

**Result (`diffest.py`, wheel-referred, N·m·s/rad, low / mid / high).**

| Oil | Axle diff `dfront` / `drear` | Centre diff (× 4.5² ≈ 20) |
|---|---|---|
| 3k | 0.0002 / 0.0006 / 0.0017 | 0.005 / 0.013 / 0.035 |
| 5k | 0.0004 / 0.0011 / 0.0029 | 0.008 / 0.022 / 0.058 |
| 7k | 0.0005 / 0.0015 / 0.0040 | 0.011 / 0.030 / 0.082 |
| 10k | 0.0008 / 0.0021 / 0.0058 | 0.015 / 0.043 / 0.117 |
| 20k | 0.0015 / 0.0043 / 0.0115 | 0.031 / 0.087 / 0.233 |

Torque is linear in viscosity, so scale by cSt for other oils.

**Centre ratio.**
- The centre diff's outputs drive the front and rear diff pinions.
- Each pinion turns at the ring ratio (4.44 Tekno, 4.6 AE) × that axle's carrier speed, which is the axle's mean wheel speed.
- So the centre diff's internal Δω = ratio × (front mean − rear mean).
- Wheel-referred damping = internal damping × ratio² ≈ 20× for the same diff hardware.
- Associated and Tekno use the same diff internals front, centre and rear (same diff gear set part numbers in the manuals above).

**What it means on the car** (size8: wheel spin inertia 6.5e-4 kg·m², drive 1.25 N·m per wheel, r = 71.5 mm):
- Front 10k (c = 0.002), inside front wheel 8 m/s over its partner (Δω = 112 rad/s): 0.22 N·m transferred. That is 18% of one wheel's drive torque, so the axle diffs stay close to open.
- Time constant for an axle's two wheels to re-equalise with no ground: τ = I/(2c) = 0.16 s at 0.002, 0.07 s at 0.005.
- Centre at 0.04: a 2 m/s front-to-rear mean speed difference (28 rad/s) gives 1.1 N·m, about one wheel's full drive torque. The centre is strongly limited: front wheelspin is quickly bled to the rear, and a full brake stays straight. This matches the earlier prototype, where c ≥ 0.03 made braking straight ([physics review](physics-review-phase46.md)).
- Sanity check: by hand, a 10k diff turns freely but with noticeable drag. Turning one outdrive at 2 rev/s against the other held gives Δω ≈ 12.6 rad/s, so 0.025 N·m at the midpoint. That is a light twist. Plausible, not measured.

**Shear thinning.**
- In the estimated gaps, shear rates are 150–1,700 s⁻¹ (faces) and 400–4,400 s⁻¹ (journals) for Δω = 10–110 rad/s.
- PDMS oils are Newtonian at low rates and thin above a critical rate that "increases with decreasing zero-shear-rate viscosity". A 1,000 cSt oil (1.05 Pa·s) has γ̇c ≈ 2×10³ s⁻¹ ([Huneault & Higgins, Shock-wave-induced cavitation of silicone oils, Table II](https://arxiv.org/pdf/1906.00310)).
- The same paper cites capillary rheometry on high-viscosity PDMS: 20 → 12 Pa·s (−40%) and 125 → 25 Pa·s (−80%) between 10² and 10⁴ s⁻¹.
- So 10k–20k oil is likely already thinning at large slip. Expect the effective c to fall by about 20–50% between small and large Δω (estimate). Locking torque then grows sub-linearly. The ×0.5–2 randomisation covers this.

### 3. MuJoCo implementation

**What MuJoCo offers** (all from the [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)):
- `equality/joint`: "constrains the position or angle of one joint to be a quartic polynomial of another joint". Two joints only, so it can't express a 3-member differential (carrier = (L + R)/2).
- `equality/tendon`: "constrains the length of one tendon to be a quartic polynomial of another tendon". Fixed tendons are linear combinations of joints, so the tendon equality can tie a rotor joint to the 4-wheel average: `tendon1="rotor"`, `tendon2="drive"`, `polycoef="0 N 0 0 0"`. With `tendon2` omitted, it pins one tendon to a constant. That would let an explicit carrier joint satisfy carrier − ½L − ½R = 0.
- Equality constraints are soft (`solref` / `solimp`; [computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html)). Use `solref` of about 2× the timestep (0.004 1) for a gear.
- Tendon `damping`: linear, or polynomial f(v) = −(a·v + b·v|v| + c·v³). Tendon `armature`: "adds a kinetic energy term ½mv², where v is the tendon velocity". Tendon `frictionloss`: dry friction. All three are present in the installed MuJoCo 4.0.1 (`tendon_dampingpoly`, `tendon_armature`).
- Joint `armature` is the "reflected inertia … the inertia of the spinning body multiplied by the square of the gear ratio".

**Why the current tendon is already a correct open diff.**
- Joint torques are Jᵀf = 0.25·f per wheel, and the tendon velocity is the mean of the wheel speeds.
- Explicit carriers with ideal (rigid, massless) bevel gears give exactly the same equations: equal torque to all four wheels, and centre-carrier speed = mean of the axle-carrier speeds = mean wheel speed.
- Building carrier bodies with equality constraints only adds DOFs and constraint compliance. It is worth it only if the carriers or the rotor should carry inertia of their own.

**Recommended structure** (inside `if "drive" in real:`):

```xml
<!-- rotor about the chassis' long axis, inside the chassis body: torque roll + gyroscopics -->
<body name="rotor" pos="0 0 0.01"><joint name="rotor" type="hinge" axis="1 0 0"/>
  <inertial pos="0 0 0" mass="0.14" diaginertia="8e-6 4e-5 4e-5"/></body>
...
<tendon>
  <fixed name="drive"><joint joint="fl_spin" coef="0.25"/> ... (rl, rr, fr) </fixed>
  <fixed name="rotor_t"><joint joint="rotor" coef="1"/></fixed>
  <fixed name="dfront"  damping="0.002"> fl +1, fr -1 </fixed>
  <fixed name="drear"   damping="0.001"> rl +1, rr -1 </fixed>
  <fixed name="dcentre" damping="0.04">  fl, fr +0.5; rl, rr -0.5 </fixed>
</tendon>
<equality><tendon tendon1="rotor_t" tendon2="drive" polycoef="0 14 0 0 0" solref="0.004 1"/></equality>
<actuator><motor joint="rotor" gear="{4*tau_max/14}" ctrlrange="-1 1"/></actuator>
```

- The scratchpad prototype (`rotor.py`: free chassis, rotor, four wheels) holds rotor/mean-wheel = 14.000000 under full torque. The chassis gets both a roll reaction (rotor) and a pitch reaction (wheels).
- `env._actuate` can then read the motor speed as `qvel[rotor]/14` instead of the wheel mean (identical while the constraint holds).
- **Cheaper alternative:** skip the rotor body and put `armature="1.5e-3"` on `drive`. It gives the same common-mode inertia and none in the wheel-difference modes, but the rotor reaction lands on the pitch axis. That matters for air control ([air-control notes](../phases/air-control.md)): real in-air throttle pitch comes only from the wheels, while the rotor twists roll.

**Rotor inertia (estimate).**
- A 1/8 4-pole brushless rotor of about 21 mm diameter × 55 mm, steel and magnet, weighs about 0.14 kg. I ≈ ½·m·r² ≈ 8e-6 kg·m² (range 5e-6 – 1.2e-5).
- Referred to the mean wheel speed at N ≈ 14: 8e-6 × 196 ≈ 1.6e-3 kg·m².
- The spur and centre diff add about 2e-4 (estimate).
- No manufacturer publishes rotor inertia that I found.

**Numerics.**
- `implicitfast` integrates joint and tendon damping implicitly ([computation docs, Integrators](https://mujoco.readthedocs.io/en/stable/computation/index.html)), so the recommended coefficients (≤ 0.12) are not a stability risk on their own.
- The earlier review saw NaNs only at c = 1.0 ([physics review](physics-review-phase46.md)).
- Retest any change at the top of the randomisation range together with the "heavy" and "wheels" switches.

## Open items

- Losi 8IGHT-XTE, Kyosho MP10Te, Mugen MBX8T Eco and HB manuals weren't reachable or weren't found. The Losi and Kyosho numbers above are search snippets and forum posts. These hosts would have to be allowed in the session's network settings to check them: manualslib.com, manuals.online, researchgate.net.
- No measured RC diff torque curve exists in what I found. A bench test (hold one outdrive, drive the other at a known speed, read torque) would replace the 1–5× gear-mesh guess.
