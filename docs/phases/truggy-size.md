# The full-size 1/8 truggy (`size8`, part of `real="truggy2"`)

The owner noticed the collision box rode about 4 cm higher than a real chassis would, and asked for a check of the car's size and suspension against one real car. The answer: none of it was chosen on purpose. Phase 4 built a generic 1/10 buggy, and the later "truggy" switches changed the motor, brake, diffs and grip, but not the body. The owner picked the full-size 1/8 truggy (2026-11-26: "A, go with the full-size truggy").

`real="truggy2"` = `"truggy"` plus `wheels` (air-control.md), `groundroll` and `size8`. The old presets build exactly the same model as before (checked: identical MuJoCo XML).

| | Old sim car | `size8` | Real 1/8 truggy (ET48 / 8IGHT-T) | Source |
|---|---|---|---|---|
| Wheelbase | 300 mm | 380 mm | 375–385 mm | [1] |
| Track, wheel centres (over the tyres) | 280 mm | 356 mm (about 420 mm) | 416–420 mm over the tyres | [1] |
| Tyre diameter | 110 mm | 143 mm | 141–143 mm | [2][3] |
| Mass | 2.48 kg | 4.02 kg | about 4.0 kg | [1] |
| Collision box (chassis and body) | 0.46 × 0.18 × 0.07 m | 0.58 × 0.20 × 0.07 m | 595 mm long | [1] |
| Chassis bottom above ground, at rest | 76 mm | 34 mm | about 30–38 mm (setup blocks and sheets use about 30 mm) | [4][3] |
| Centre of mass height | 103 mm | 69 mm | about 75–90 mm (estimate) | [3] |
| Wheel + tyre mass | 0.12 kg | 0.20 kg | about 0.2 kg (estimate) | air-control.md |
| Wheel spin inertia | 1.56e-4 kg·m² | 6.5e-4 kg·m² | 5–8e-4 kg·m² (estimate) | air-control.md |
| Wheel travel, sag | 8 cm, 30% | 8 cm, 30% | not found; long-travel class | — |
| Spring per corner, bounce, damping | 185 N/m, 3.2 Hz, about critical | 296 N/m, 3.2 Hz, about 0.9 of critical | not found | estimate |
| Motor (per wheel, no-load speed) | 0.6 N·m, 16 m/s | 1.25 N·m, 18 m/s | 4S, about 20 m/s no-load (estimate) | air-control.md |

Notes:
- **Ride height.** The chassis bottom is now 34 mm off the ground, so on a big landing the box bottoms out before the suspension's 5.6 cm of bump travel is used up, as a real truggy's chassis does.
- **Springs and damping** keep the old car's 30% sag and bounce rate, scaled to the new corner load. I found no published spring rates or wheel travel for a 1/8 truggy. Shock specs (spring rate, stroke, motion ratio) would pin these down.
- **Motor** keeps the old thrust per kg, so the car accelerates as before (0 to 7.3 m/s in 1 s). The no-load speed rises from 16 to 18 m/s, a little under a 4S truggy.
- **Air control** at 7 m/s: full brake −172°/s (real estimate about −210, range −140 to −280), full throttle +125°/s (real estimate +200 to +400; short of it because the wheels reach the 18 m/s no-load speed quickly).
- **Not changed:** the tracks (lane width, jumps), steering lock and servo, sensors (moved to the new bumper and sides), tyre grip.

## Sources

1. RC Car Action spec block, Tekno ET48 (length 595 mm, wheelbase 375–385 mm, width 416–420 mm, 4000 g). https://www.rccaraction.com/?p=269264
2. HPI 1:8 truggy tyre, 143 mm overall. https://www.hpiracing.com/ko/part/160292
3. Physics review, Phase 4.6: `docs/research/physics-review-phase46.md` (8IGHT-T ride height about 38 mm, CG estimates).
4. Hudy off-road set-up book (30 mm blocks for 1/8 off-road and truggies). https://baronerosso.it/forum/attachments/automodellismo-mot-elettrico-road/291989d1389744677-le-3-c-camber-caster-convergenza-hudy_off-road_set-up_book.pdf
5. Traxxas Revo 3.3, for comparison (108 mm ground clearance, 348 mm wheelbase, 4.69 kg). https://traxxas.com/revo-33-53097-3
