# Plan: a worn track surface (groove, loose dirt, braking bumps) and the `truggy3` preset

The owner asked for this on 2026-11-28 (phase-4.7.md, "Owner on surfaces"). It needs an independent realism review before any code is written.

## What a real dirt track looks like (owner)

"On a real track, grip is mostly uniform apart from two things - areas that are bumpy and areas where cars have either made the dirt smooth or loose." The groove follows "the RL racer's line, since it's actually faster and where cars will race. But the ideal racing line plus some noise would work, too."

## Today's surface (`sdc/jumps/loop.py`, `truggy2`)

- Grip is one number per run: friction ×0.55–1.25, applied to the tyres, the ground and the skids.
- Roughness is a few cm of smooth bumps along s, the same across the lane's width. There is one stretch of whoops, and jumps, and some banked turns.
- The heightfield grid is 0.05 m. The lane is 3 m wide with banks.

## Proposal

### 1. Race line per loop

- **Problem:** the RL racer's own line doesn't behave like a racing line. On the old chassis it sat 0.39 m from centre on average with no inside bias in corners (phase-4.7.md, racing-line table). Running the racer on every practice loop (an unbounded set) would also be expensive.
- **Plan:** compute a **minimum-curvature line** per loop from the centreline. This is the standard racing-line approximation.
  - Lateral offset d*(s) inside ±1.0 m, leaving 0.5 m margin to the banks.
  - Solved by iterative smoothing or a small QP on the stored centreline.
  - Each loop gets a slow random wander added: about 0.15 m RMS over a wavelength of about 10 m. That is the owner's "plus some noise", and it stands for drivers not all taking the same line.
- **Check against the racer:** retrain the racer on the new surface, then measure its line on 20 validation loops. If its lap is faster near d*, the groove is self-consistent.

### 2. Grip field, per tyre per step

- The local grip multiplier g(s, d) depends on the distance from the line, e = |d − d*(s)|:
  - **Groove, e < 0.4 m:** ×1.08, packed and smooth.
  - **Transition, 0.4–0.8 m:** blends linearly to ×1.0.
  - **Loose (marbles), e > 1.0 m and on the outside of a corner:** ×0.88. "Outside" means the side of the line away from the corner's centre, where |κ| > 0.1/m. The band extends 3 m before and after the corner, which is where cars push loose dirt.
- **Tyre grip:** each step, a tyre's friction = the base friction × the run's grip × g at the tyre's position × the existing `slide` drop.
- **Tyre position:** the car's (s, d) plus the tyre's offset rotated by the car's yaw relative to the track. This is an approximation, but only s and d are needed.
- **Ground geom:** its friction stays at the run's grip, so the tyre decides. With `slide` that already holds (`geom_priority`).

### 3. Braking and acceleration bumps (heightfield)

- **Where:**
  - The 2–6 m before each corner's entry, where the race-line speed drops, centred on d* and about 1.0 m wide (tapered at the edges).
  - The 1–4 m after each jump's landing.
- **Shape:** ripples across the line of travel.
  - Wavelength 0.35–0.6 m.
  - Amplitude 0.8–2.0 cm, deepest where braking is hardest and fading at the ends.
  - Random phase and amplitude per loop.
- **Grid:** 0.35 m is 7 grid cells, so the hfield can represent it. The tyre radius is 7.2 cm on `size8`.
- **Implementation:** `LoopCourse.surface(i, d)` gains a d-dependent term.

### 4. `rotor`

Already built (opt-in): the motor's rotor inertia plus 10k centre-diff damping (docs/research/truggy-diffs.md).

### 5. Preset

- `truggy3` = `truggy2` + `rotor` + `groove` (sections 1–2) + `bbumps` (section 3).
- New loops keep the same seeds, so validation and gate loops gain grooves and bumps deterministically.

## Re-baselining (all scores change)

1. Teacher on `truggy3`, 5 passes: the new bar.
2. RL racer: retrained on `truggy3`. Its privileged obs gain g(s, d) at the four tyres and d − d* ahead, so it can find the groove.
3. Sensor students (E5 recipe), then E7 on `truggy3`. The sensors can't see the groove, but they can feel it (grip and bumps) and remember it lap to lap.
4. Race score as now: 20 × median lap + 20 × crash rate × 7 s.

## Owner's prediction (2026-11-28, verbatim)

"Grooves will add bumps for both cars, but they should interfere more with the RL car. However, traction should help the RL car more. I think the net effect will benefit the RL car. And I think that's true to reality - the track can work against you if you get off line."

## Open questions for the review

- Are the groove and loose-dirt magnitudes (+8%, −12%) realistic for a 1/8 off-road track?
- Are the braking-bump wavelength and depth right for 1/8 scale cars at these speeds (5–12 m/s)?
- Is a minimum-curvature line with wander a fair stand-in for where a groove forms?
- Is there anything else that defines a real worn track and is missing here?

## Revisions after the realism review ([review](../research/surface-plan-review.md))

Adopted:
- **Track state per loop**, drawn from loam to blue groove. It sets the groove gain (+3 to +15%), the off-line loss everywhere (−3 to −15%), marbles on corner exits (−10 to −30%) and bump size, which is larger on loam.
- **Smooth grip everywhere.** Blends replace both the step at 1.0 m from the line and the corner-threshold switch. Folded into the existing `slide` friction write. Grip is rasterised on the hfield grid once per loop and sampled at the four tyres. The total range is clipped, and the low end is checked to be drivable.
- **Groove shape:** apex later than the minimum-curvature line, and wider in corners.
- **Bumps:**
  - Amplitude is peak-to-peak, 0.8–2 cm.
  - Wavelength scales with the race-line speed, 0.25–0.55 m.
  - Placement follows the race-line speed profile: where it brakes hard, plus acceleration bumps on corner exits.
  - Ridges are skewed up to ±15°.
  - A touchdown dish replaces the ripples after landings.
- **Implementation:** heights computed at the grid nodes, not splatted (splatting errs by up to 9 mm), and the landing-snag check rerun on the new surface.

Deferred (scope): ruts and lips on the kicker face, a weak inside line, and the groove coming in or the track drying over a race.

## Built (2026-11-28)

`sdc/jumps/surface.py`, switches `groove` and `bbumps`, preset `truggy3` = `truggy2` + `rotor` + `groove` + `bbumps`. `truggy2` is unchanged.

- **Race line:** minimises bending energy (Σκ²ds, L-BFGS on the true curvature) within ±1.0 m, apex 0.75 m later, plus a wander of 0.15 m RMS on a 25 m wavelength. A 10 m wavelength made the straights curvy.
  - On three loops the line uses the full width (±1.1 m with the wander) and sits 0.7 m to the inside in corners.
  - Built in 2–5 s per loop and cached per loop.
- **Speed profile** for placing the bumps: 7 m/s² cornering, 6 m/s² braking, 4 m/s² drive, 12 m/s top speed. It brakes over about 20–25% of a lap.
- **Grip map:** rasterised on the hfield grid and looked up under each wheel every physics step, multiplied into the `slide` friction write. Marbles follow the centreline's curvature, not the line's, which has small kinks.
  - On loop 100187 (track state 0.45), the teacher's tyres saw 0.55–1.08 × the run's grip (the low end includes `slide`), with a median of 0.92: the centre line is mostly off the groove.
- **Bumps** are computed at the grid nodes from an exact local projection. On loop 10001 (track state 0.57) they are about 1.3 cm peak-to-peak, with touchdown dishes after landings.
- **Picture:** loop 10001's grip map and bumps, in [surface-10001.png](surface-10001.png).
- **Smoke test:** the teacher finishes loop 100187 from all 3 starts on `truggy3` (23.3–23.7 s, against 23.3–23.6 s on `truggy2`).
