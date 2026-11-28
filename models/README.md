# Champion cars

The leading trained cars, kept in git so a fresh cloud session can drive, score or replay them without retraining
(each is about 350 KB). Full training runs and checkpoints stay out of git (`runs/` is ignored; history is on the
`car-weights` branch).

Each folder is a run directory as `sdc.phase46.common.load_student` reads it: `model.pt` plus a slim `metrics.json`
(the training summary without its per-episode lists). `metrics.json["env46"]` is the sim the car was trained and
scored in; tools that drive the car use it rather than `SDC_REAL`. `metrics.json["champion"]` says where it came from.

| Car | What | Sim | Race (20 laps, 7 s per crash) |
|---|---|---|---|
| `e7` | Phase 4.7 E7: sensors-only student of the privileged RL racer, then RL on its own | full-size truggy (`truggy2`) | 563.6 s estimated (teacher 582.8 s) |

Use: `Ensemble(("models/e7",))` (`sdc.phase46.ensemble`), or `python -m sdc.race.record --car models/e7`.
