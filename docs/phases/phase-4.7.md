# Phase 4.7 — Beyond the teacher: drills and residual reinforcement learning

- **Status:** in progress
- **Goal:** the car learns to handle itself from outcomes, not only by copying, and ends up better than the teacher it was copied from. Phase 4.6's copied car is kept frozen; reinforcement learning (the owner's PPO) learns a small correction on top of it, first in drills (short episodes started in hard states), then on full laps.
- **Gate (set at entry, 2026-11-19):** on the test loops (46000–46019 × 3 starts, gate-range physics, truggy physics, timed lap only), the car beats the re-tuned teacher on the same 60 runs (teacher: 52 / 60 at the 4.6 gate retry) in **either** way (owner: "either option counts as success"):
  - **(a) more reliable:** more laps finished than the teacher, with a median lap no slower than the teacher's; or
  - **(b) faster:** at least as many laps finished as the teacher, with a faster median lap.
- **Code:** `sdc/phase47/` (to come); base car `sdc/phase46/bc_loop.py` + `ensemble.py`; rebuild in a fresh container with `bash sdc/phase46/rebuild.sh`.
- **How to reproduce the gate:** to come.

## Entry decisions (2026-11-19)

1. **Phase:** a new phase 4.7, not a 4.6 stretch (4.6's gate passed; this is a new method). Owner agreed.
2. **Success:** either more finishes than the teacher, or the same finishes with faster laps (gate above). Owner's choice: either counts.
3. **First drills:** too-fast corner entries and bad takeoffs, the base car's main ways of failing (off the lane, rolled over). Mid-slide drills later. Owner agreed.
4. **Compute:** the container (4 CPUs). Modal can't be reached from cloud sessions: its client uses gRPC, which the sandbox proxy doesn't carry (tested 2026-11-19 with the token and an unrestricted network). A 1M-step run is about 2 h on one core; 4 at once.

## Why reinforcement learning failed before, and what changes

- Phases 4 and 4.5 fine-tuned the **whole copied network** with PPO. It drifted: it traded landings for speed (Phase 4), or lost 10–15 points within 50k steps (Phase 4.5); a finish bonus slowed the slide but didn't stop it. Checkpoints were picked on noisy 20-course scores.
- Changes: (1) the copy is **frozen**; PPO learns only an additive **residual**, which starts at zero and is capped, so the car can't forget how to drive; (2) **drills** start the car where mistakes happen, so a run sees hundreds of near-crashes instead of about one crash per 7 laps; (3) a crash penalty and finish bonus on full laps; (4) checkpoints picked on 60 validation runs.

## Plan

0. **Build (agent):** rebuild the base car (E10 arm B, the averaged expo pair: 52 / 60 on validation, teacher 55). Drill generator: the base car drives practice loops (training-range physics) while snapshots are taken (physics, sensors, filter, and both students' memories; `sdc/phase46/hard.py`). Drill starts: 1 s before a corner with speed raised 10–40%; 1 s before a kicker with speed changed ±25% and the nose pitched up or down; 1–3 s before the base car's own crashes. Residual wrapper around the env, so the owner's PPO core (`sdc/phase1/my_ppo.py`) is used unchanged.
1. **E1 — Drills alone.** Residual trained only on drills (3 s each: metres along the lane, −50 for a crash), scored on full validation laps.
2. **E2 — Drills, then full laps.** E1's residual continued on full practice laps mixed with drills: progress, finish bonus, crash penalty.
3. **E3 — Speed** (if E1–E2 make it more reliable but not faster): a lap-time reward, or drills at higher target speeds.
4. **E4 — Mid-slide drills** (if useful): starts in a held slide.

Every experiment gets the owner's prediction before it runs.

## Experiments

### E1 — Drills alone

- **Prediction (owner, before the run):** "54 and a little slower" (validation laps finished of 60; base car 52, teacher 55).
- **Setup:** frozen base car (E10 arm B pair, averaged); residual = the owner's Phase 1 network (two 64-unit layers) on sensors + the base car's action, output capped ±0.3, starting at zero; owner's PPO core. Drills only (3 s; +1 per metre along the lane, −50 for a crash, small cost on the residual): too-fast corner entries, bad takeoffs, and starts 1–3 s before the base car's own crashes, on practice loops at training-range physics. 1M steps, seeds 0–1; checkpoint kept by 60 validation runs (gate range) every 100k steps.
