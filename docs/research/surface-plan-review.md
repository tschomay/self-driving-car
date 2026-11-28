# Review: worn track surface plan (docs/phases/surface-plan.md)

This is an independent realism review, written 2026-11-28. It sets the plan against evidence on real 1/8 off-road dirt tracks and checks it for MuJoCo pitfalls. Estimates of my own are marked **[est]**.

Sources are thin. There are no published grip measurements for RC tyres on a groove versus off it. The evidence is track-operator and racer reports, tyre-maker guides, full-size friction tables and washboard-road physics. Any number below that is not quoted from a source is an estimate.

Fetching was blocked in this session (arxiv.org and ar5iv.arxiv.org would not resolve). The paper figures below come from search-result abstracts and snippets, not full texts.

## Verdict table

| Plan item | Verdict | Short reason |
|---|---|---|
| Groove grip ×1.08 | **Realistic for a loamy track. Too small for a hard-pack or blue-groove track** | Racers describe off-line grip on blue-groove clay as "dust on a gym floor" against "pavement-like" grip on the line. On loose loam the line gains less grip. |
| Loose grip ×0.88 | **Too small, and too narrow in where it applies** | Full-size tables give dry earth 0.68 against gravel 0.60 (−12%), but loose sand 0.3–0.4 against rolled gravel 0.6–0.7 (−45%). Marbles on a groove track sit closer to −20…−35% **[est]**. Real tracks are dusty everywhere off-line, not only outside corners. |
| Groove core e<0.4 m, transition to 0.8 m | **Realistic** | The car is about 0.42 m wide over the tyres (corner_y 0.178 plus the tread). Car-to-car spread adds roughly ±0.2 m, so a groove about 0.8–1.0 m wide fits **[est]**. |
| Gap between 0.8 and 1.0 m, then a step to 0.88 | **Wrong shape** | There is a hard step at e = 1.0 m and at the |κ| = 0.1 corner threshold. Blend it, as the groove side already does. |
| Loose only outside corners, ±3 m | **Too narrow** | Keep the outside of corner exits as the worst spot (marbles), but give all untracked lane a milder loose level, and the inside apex area too. |
| Min-curvature line + 0.15 m RMS wander over about 10 m | **Mostly realistic. Shape slightly wrong in corners** | Min-curvature is the standard approximation. Real 1/8 lines brake late and apex late, so the groove sits later than min-curvature on entry. Some corners also carry two lines (inside "protect" and outside momentum). The wander amount is fine **[est]**. |
| Braking-bump wavelength 0.35–0.6 m | **Slightly long, but plausible. It should vary with speed** | Washboard wavelength grows with speed (speed-wavelength law). Full-size cars make 0.5–1.0 m at 10–15 m/s. Scaling to the 1/8 truggy's lighter mass and stiffer, higher-frequency suspension gives about 0.2–0.5 m at 5–10 m/s **[est]**. Use λ ∝ local speed, inside about 0.25–0.5 m. |
| Amplitude 0.8–2.0 cm | **Realistic only if it means peak-to-peak. Too big if it means half-amplitude** | Half-amplitude 2 cm is 4 cm peak-to-peak, 56% of the 7.15 cm tyre radius, at a steepness of 0.11. That suits a bad loam track late in the day. A typical well-kept track has about 0.5–2 cm peak-to-peak **[est]**. The plan should say which it means. |
| Bumps 2–6 m before corners | **Right idea, wrong rule** | Braking from 12 to 5 m/s at 0.8–1 g takes about 6–9 m **[est]**. Place the bumps where the race-line speed profile decelerates hard, not in a fixed window. |
| Bumps 1–4 m after landings | **Partly right** | Landings mainly make a **dish or hole at touchdown**, which moves with jump length, plus ruts. Ripples follow only if a corner or braking comes next. |
| Bumps on the line only, about 1.0 m wide, tapered | **Realistic** | Bumps form where cars brake, which is the line. A 1.0 m width with tapers matches car width plus spread. Add a ±15° random skew to the ridges. |
| Acceleration bumps on corner exits | **Missing** | Corner exits under 4WD power are the classic place for acceleration bumps and ruts, about 0–3 m after the apex **[est]**. The plan's title names them but section 3 never places them. |
| Groove regime (rutted loam vs slick blue groove) | **Missing** | These are two different track states with opposite signatures. See below. |
| Kicker lip and face wear | **Missing** | Ruts on the take-off face and a built-up lip change the launch angle. This matters for a jump-heavy sim. |
| Dust, drying, wet patches, berms | **Missing (lower priority)** | Mainly a variation over time and an outside cushion. See below. |

## Evidence and reasoning

### Groove vs loose: two regimes, not one

- **Blue-groove or hard-pack clay.** Tyre rubber polishes a dark line into hard dirt, "that's where the best grip is". Off it, "you will spin, slide, or crash". One racer puts sealed-clay grip between "dust on a gym floor" and "pavement-like" ([RCTalk: blue groove](https://www.rctalk.com/forum/threads/blue-groove.48569/), [RC Tech: what is blue groove](https://www.rctech.net/forum/electric-off-road/53880-what-hell-blue-groove-print.html), [RC Tech: clay track building](https://www.rctech.net/forum/electric-off-road/434051-clay-track-building-maintaining.html)).
  - Tracks blow dust off and water "several times a day" because dust returns to the lanes after a race or two. They sweep loose dirt off the race line daily ([RC Tech: traction for race surface](https://www.rctech.net/forum/electric-off-road/162932-looking-help-traction-need-race-surface.html)).
  - Here the groove-to-off-line contrast is large: off-line about −30…−50% relative to the groove **[est]**.
  - Bumps are small, because hard surfaces resist cutting. Tyre makers sell a separate "groove" compound for this regime (JConcepts lists white as "Medium – groove conditions"; [JConcepts tyre guide PDF](https://horizonhobby.com/on/demandware.static/-/Sites-horizon-master/default/dwea7dd922/Manuals/JConcepts_Tire_guide.pdf)).
- **Loam or loose dirt.** The line gets rutted and bumpy rather than slick. Grip on and off the line is closer, because both are loose. One test report notes "loam on the surface caused too much wheel spin". Reviews mention "ruts and bumps put there by big 1/8 vehicles" ([RC Driver](https://www.rcdriver.com/?p=18249), [RC Driver](https://www.rcdriver.com/?p=10794)). Horizon recommends medium compounds "if there's loose dirt on the surface" and soft compounds for "well-packed tracks that have little loose material" ([Horizon tyre article](https://www.horizonhobby.com/blog-article-rc-tires)).
- **Over a meeting:** "As the day progressed and the track dried out, traction fell off". Worlds tracks "slowly deteriorate throughout the week" ([RC Car Action, 2016 1/8 Worlds](https://www.rccaraction.com/?p=288217)).

**Friction tables (full-size, for scale only).** Dry earth road peaks at 0.68 and gravel at 0.60, about −12%. The hpwizard page also notes that sliding is close to peak on loose surfaces ([hpwizard](https://hpwizard.com/tire-friction-coefficient.html)). Loose sand is 0.3–0.4 against dry rolled gravel 0.6–0.7 ([Engineering Toolbox](https://www.engineeringtoolbox.com/amp/tractive-effort-d_1783.html)). So −12% is the *mild* end, about dirt against gravel. Real marbles on a hard base act like ball bearings and sit lower.

**Recommendation.** Add one per-loop "track state" scalar w ∈ [0, 1], drawn from the seed, running from loamy (0) to blue groove (1). It sets:

| w | Groove | Off-line base | Outside-exit marbles | Bump amplitude |
|---|---|---|---|---|
| 0 (loam) | ×1.03 | ×0.97 | ×0.90 | full |
| 1 (blue groove) | ×1.15 | ×0.85 | ×0.70 | about ×0.3 |

All values in this table are **[est]**.

- The plan's (+8%, −12%) then sits mid-range.
- Do not let the product of run grip × g fall outside what the run range already covers by too much. The 0.55–1.25 run range is itself ±40%. Clip the product, to about [0.45, 1.35] **[est]**.
- Optional: loose areas have a flatter μ-slip curve (sliding ≈ peak, per hpwizard), so scale `SLIDE_DROP` down in loose zones, for example 0.25 → 0.1 **[est]**.

### Groove location

- Min-curvature is a fair first stand-in. Real drivers trade some curvature for a straighter exit, which means a later apex (closer to minimum time). A cheap fix is to shift the apex of d* forward by about 0.5–1 m in s on corners, or to blend min-curvature with a shortest-path line **[est]**.
- Wander of 0.15 m RMS over about 10 m is fine. Separately, spread the groove's *width* more in corners than on straights (about 0.5 m core on straights, about 0.9 m in corners **[est]**). In about 25% of corners, add a weaker second line on the inside (the "protect" line) at half strength.
- The 0.5 m bank margin is right. A track walk at the 2016 Worlds noted "a variety of lines" ([RC Car Action](https://www.rccaraction.com/?p=288217)).

### Braking and acceleration bumps

**Physics.** Washboard ripples form above a critical speed and do *not* need the suspension to be forced into oscillation ([Bitbol et al. 2009, abstract](https://www.physics.utoronto.ca/nonlinear/abstracts/BTMM09abstract.html); [Taberlet, Morris & McElwaine 2007](https://arxiv.org/pdf/0711.3872), [Taberlet page](https://perso.ens-lyon.fr/nicolas.taberlet/washboard)).

- In the lab a 100 mm wheel made ripples 50–500 mm long, a few mm up to 50 mm high.
- Full-size dirt roads show λ = 0.5–1.0 m at 10–15 m/s, and amplitude up to 50 mm ([arXiv 2007.05109](https://arxiv.org/pdf/2007.05109)).
- A "speed-wavelength law" makes λ equal to the mean speed × half the tyre's tangential-oscillation period, so λ ∝ v ([arXiv cond-mat/0011510](https://arxiv.org/abs/cond-mat/0011510v2)).
- Grain size does not set the wavelength (Taberlet et al.).

**Scaled to a 4 kg truggy at 5–10 m/s under braking.** Its suspension and wheel-hop frequencies are well above a car's, roughly 10–20 Hz against about 10 Hz **[est]**. That gives λ ≈ 0.2–0.5 m. So 0.35–0.6 m is on the long side but tolerable. Better: λ(s) = clip(v_line(s) × 0.04 s, 0.25, 0.55) m **[est]**, so the ripples tighten toward the apex as speed falls. That matches what riders see.

**Amplitude.** Ripple height in the washboard literature scales with wavelength (Taberlet: a few mm at short pitch, up to 50 mm at 500 mm).

- Steepness A_pp/λ ≈ 0.03–0.08 **[est]** gives 1–3 cm peak-to-peak at 0.35 m.
- The plan's "0.8–2.0 cm" works as **peak-to-peak**.
- As half-amplitude its upper end, 4 cm peak-to-peak, is a late-day loam worst case.

**Placement.**

- Use the race-line speed profile (from the min-curvature line and a friction-circle speed profile): put bumps where deceleration exceeds about 0.4 g.
- Add exit acceleration bumps where acceleration under power is high, from the apex to about 3 m after.
- Keep bumps off the take-off ramp itself.
- Corner marshals and crews hoe these out between rounds: "scraping acceleration and braking bumps near corners with a hula hoe" ([LiveRC, Cactus Classic track crew](https://www.liverc.com/news/the-cactus-classic-track-crew-talks-about-trying-to-maintain-traction-for-nearly-500-entries)). That supports randomising severity per loop, including near zero.

**Landings.** Replace "ripples 1–4 m after landing" with:

- a dish at the typical touchdown point, about 1.5–3 cm deep and 0.6–1.0 m long, centred on d* **[est]**;
- ripples after it only if braking follows.

### Missing features, in priority order

1. **Track-state regime** (above). It is cheap and it drives both grip contrast and bump size.
2. **Exit acceleration bumps and ruts** (above).
3. **Kicker wear.** Two shallow ruts on the take-off face along the wheel tracks (about 1 cm **[est]**) and a small lip at the top (about 1–2 cm **[est]**). These change launch pitch, which this sim cares about.
4. **Landing dish** (above).
5. **Outside cushion or berm.** On loose tracks pushed dirt piles up at the outside edge of corners: a low ridge about 1–3 cm high, 0.3 m wide, just inside the bank **[est]**. It gives loose grip but some lateral support. The inside apex often gets a pocket or rut that cars lean on. Both are optional.
6. **Variation over time within an episode.** The groove "comes in" and the track dries. Put a slow per-lap drift on w, for example +0.02 per lap **[est]**. This rewards the students' lap-to-lap memory honestly.
7. **Wet patches.** These are low-value for this project. Watering is uniform after prep, and the shade and puddle cases are rare. Skip them.

### Owner's prediction

Per the regimes above:

- On loam tracks (low w) the net effect favours bump-tolerant driving over line-holding.
- On blue groove (high w) it strongly favours the car that holds the line.

So the prediction is a function of w. Score both ends separately, or the result will hinge on one unrealistic midpoint.

## Simulation pitfalls

1. **Nearest-cell splatting aliases the bumps.** `loop.py:hfield()` writes each fine (s, d) sample to its nearest 0.05 m cell. A 2 cm half-amplitude ripple at λ = 0.35 m has a maximum slope of 2π·0.02/0.35 ≈ 0.36. A position error of up to 0.025 m then gives about 9 mm of height error, which is as large as the bump itself.
   - Ripples run oblique to the world grid in corners, so the error varies spatially and shows up as moiré.
   - Fix: build the bump term by evaluating the surface *at the grid nodes*, mapping each node's (x, y) to (s, d), or by area-weighted averaging. Do not use nearest-sample.
2. **Minimum wavelength vs grid and tyre.** At 0.05 m a 0.25 m ripple is 5 cells along an axis and only about 3.5 cells across a diagonal (the diagonal cell pitch is about 0.07 m). Keep λ ≥ 0.25 m.
   - The tyre capsule (r 7.15 cm) follows a sinusoid while A·(2π/λ)² < 1/r. At λ = 0.25 m that means half-amplitude < 2.3 cm. Fine.
3. **Contacts on a bumpy hfield.** MuJoCo collides the capsule against the hfield's triangular prisms and can produce many contacts per pair. More relief means more contacts and more cost, and can wake up the landing-snag behaviour noted in `car.py:146`.
   - Re-run the landing-loss check on `truggy3`, especially with the landing dish and kicker ruts.
   - Contact softness (solref) penetration of a few mm is small next to 1 cm bumps. OK.
4. **When friction is updated.** `env.py:496` already rewrites `geom_friction[tires, 0]` per control step for `slide`. The groove factor must go into that **same** write (μ0 × g × slide drop), or one overwrites the other.
   - MuJoCo reads geom friction in `mj_collision` each step, so the change takes effect on the next step.
   - Per control step the tyre moves up to about 0.24 m at 12 m/s, if one mj_step is 10 substeps of 2 ms (env.py:538). The 0.4 m transition is wider than that, which is acceptable, but every boundary in g must be smooth. That is another reason to remove the e = 1.0 m and |κ| = 0.1 steps.
   - The `priority` setup that makes the tyre's friction win is already there for `slide`. Confirm the skids keep their own friction (`SKID_MU`).
5. **Cost of per-tyre lookups.** It is negligible either way, but the simplest robust choice is to **rasterise g onto the hfield's (x, y) grid once per loop** and sample it bilinearly at the 4 tyre contact points. That is 4 lookups per control step.
   - It avoids closest-point s search errors at the s wrap and in hairpins.
   - It also gives the same field to the privileged observations and to any render texture.
   - The plan's yaw-offset (s, d) approximation is itself accurate: the error is about offset²·κ/2 ≈ 0.19²/(2·3) ≈ 6 mm at the 3 m minimum radius.
6. **Grip product range.** Run grip 0.55–1.25 × g 0.70–1.15 gives a range of 0.39–1.44. Check that the low end does not make loose exits undriveable for the teacher, and clip if needed.
7. **Bump frequency.** v/λ ≈ 12/0.35 ≈ 34 Hz, which sits near a 1/8 truggy's wheel-hop band **[est]**. The 500 Hz physics step resolves it. The 50 Hz control loop cannot react to single bumps, which is realistic.

## Recommended changes (summary)

1. Add a per-loop track-state w (loam ↔ blue groove) that sets groove gain (+3…+15%), off-line loss (−3…−15% everywhere off-line), marbles (−10…−30% on corner exits) and bump amplitude (inversely).
2. Make g smooth everywhere: no step at e = 1.0 m or at the κ threshold.
3. Groove: shift the corner apex later than min-curvature, widen the groove in corners, add an optional weak inside line.
4. Bumps: say that the amplitude is peak-to-peak; λ ∝ local race-line speed (about 0.25–0.55 m); place them by the speed profile's deceleration and acceleration, adding exit acceleration bumps; replace post-landing ripples with a touchdown dish; add a small ±15° skew.
5. Add kicker-face ruts and lip, and optionally an outside cushion and a lap-to-lap drift in w.
6. Implementation: build relief at the hfield nodes (no nearest-cell splatting); rasterise g to the grid; fold g into the existing `slide` friction write; re-test landing snags; clip the grip product.
