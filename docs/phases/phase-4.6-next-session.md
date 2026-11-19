# Phase 4.6: notes for the next session (2026-11-19)

Pick-up notes for a new Claude session, written at the end of a long session. Read `AGENTS.md`, `README.md`,
`docs/roadmap.md` and `docs/phases/phase-4.6.md` first; this file only adds what a fresh session can't infer.

## Where things stand

- **Gate passed** (2026-11-19): 50 / 60 timed laps on the truggy physics. The owner changed the rule after the run:
  a car that doesn't use the out-lap's map skips the out-lap (with it counted: 47 / 60).
- Best car: E10 arm B, `runs/p46-e10-b-s0` + `runs/p46-e10-b-s1` averaged (`sdc/phase46/ensemble.py`). `runs/` is
  gitignored, so a new container won't have it: retrain with `bash sdc/phase46/e10_runs.sh` (arm B only:
  `python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed S --out runs/p46-e10-b-sS`, about 2.5 h
  for two seeds on 4 CPUs) or on Modal.
- Physics: everything from here uses `real="truggy"` (owner's decision). Teacher: `teacher.TRUGGY`; out-lap stand-in:
  `teacher.OUTLAP_TRUGGY`; steering expo 0.4 stays on. Time reference laps: `runs/p46-teacher-val-gate-truggy.pkl`
  (regenerate: run `LoopExpert(**TRUGGY)` on `cm.VAL_SEEDS`, 3 starts, `dict(cm.CAR, real="truggy", dr="gate")`,
  pickle the episodes).
- Out-lap maps for practice/validation (`runs/p46-maps-*.json`) were made on the old physics; `bc_loop.py` filters
  practice loops by them (`maps[f"{c}|0.00"]["kappa"]`). Regenerate with `python -m sdc.phase46.maps` if missing.

## Plan agreed with the owner (2026-11-18/19)

1. Done: truggy physics, teacher re-tuned, expo (E10).
2. Done/dropped: jump notes. Perfect notes exist (`jump_notes="true"`) and the out-lap can read jump kinds (86%,
   `sdc/phase46/notes_probe.py`), but on the truggy physics a teacher that varies jump speed by kind or by each
   jump's design speed gains nothing, so notes have little to teach (log: "Plan step 2 prep").
3. **Next: drills with residual reinforcement learning, on Modal.** Freeze the copied driver; learn a small additive
   correction with the owner's PPO (`sdc/phase4/train*.py`; learner-owned: don't rewrite it, wrap the env instead).
   Drills = episodes started in hard states (too-fast corner entry, mid-slide, bad takeoff pitch), scored over the next
   2–3 s; then full laps with a progress reward and a crash penalty; keep a copying (anchor) loss; pick checkpoints on
   many runs (60+), not 20. Snapshot/restore for drill starts exists in `sdc/phase46/hard.py` (`snapshot`,
   `restore`: keep MuJoCo's state vector, not MjData copies, which leaked memory).
4. The gate is passed; RL and drills are the owner's follow-up interest (Phase 4.6 stretch or the next phase:
   ask). Every experiment still needs the owner's prediction before it runs.

## Modal

- The owner added a Modal token in a **different cloud environment** from the one this session ran in ("Open").
  Start the new session in that environment. The token should be readable as the environment variables
  `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (setup checklist: https://claude.ai/artifact/H2dCAtoE5HEzeC4uJ2hDhh).
  If they're missing, call `get_environment` and ask the owner which environment holds them; never ask for the
  values in chat.
- Check: `pip install modal && modal app list` (should list nothing, without an auth error).
- Budget: free $30/month credit, no card on file (runs stop when it's used up). Ask before anything that would
  exceed it. A Modal function needs the repo code: mount it with `modal.Image.debian_slim().pip_install(...)
  .add_local_dir(".", "/root/sdc-repo")` or similar, and `apt_install("libosmesa6")` only for GIF rendering.
  CPU containers are fine (MuJoCo is CPU-bound); several in parallel is the point.

## Gotchas learned this session

- Background tool commands stop at 2 h: launch long runs with `setsid nohup ... & disown` and poll for files.
- At most 4 training runs at once on the 4-CPU container; the truggy physics is about 1.6× slower to simulate.
- Containers restart without warning: copied data isn't saved between rounds, so a killed run restarts from zero.
- The session's artifact watch limit (10) was reached; new artifacts publish fine but aren't watched.
