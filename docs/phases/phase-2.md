# Phase 2 — Pixels → CNN

- **Status:** in progress; started 2026-11-08
- **Goal / gate:** replace rays with a low-res camera image (top-down crop first, then ego/chase view) through a small CNN encoder. **Gate:** the CNN policy reaches ≥ 90% of the Phase 1 policy's held-out lap completion (Phase 1 gate: 20/20, so ≥ 18/20 on tracks 10000–10019). A saliency/attention visualization shows the network keying on track edges, not artifacts.
- **Learn:** convolutions, frame stacking, why pixels are sample-hungry, data augmentation (DrQ-style random shifts), imitation learning (behavior cloning from the Phase 1 expert) vs. RL from scratch, saliency maps.
- **Code:** `sdc/racer/camera.py` (camera, frame stack), `sdc/racer/camera_env.py` (SB3 adapter, camera policy for lap evaluation), `sdc/phase2/train.py` (SB3 PPO on pixels, checkpoint/resume), `tests/test_camera.py`
- **How to reproduce the gate:** _to come_
- **Previous gate:** Phase 1 passed (2026-11-08).

## Phase-entry decisions (owner, 2026-11-08)

1. **Long runs: CPU with checkpoint-and-push; Modal only if a run needs more than about 2 h.** Basis (agent's benchmark on this container, 1 thread): a 64×64 camera view rendered from a pre-drawn track bitmap keeps the sim at about 6.7k steps/s; a small CNN (3 conv layers) trains at about 6.8k samples/s and infers at about 22k steps/s. Pixel PPO should run at about 600 steps/s per thread, so 1–3M steps is about 30–90 min per run, 4 in parallel. Runs save a checkpoint and push every ~10 min and resume after a container loss. No Modal account or token for now.
2. **The gate counts only for the owner's CNN encoder.** SB3's built-in CNN (NatureCNN) is the reference, as SB3 PPO was in Phase 1.

## Experiment plan

Each experiment is predict → run → explain. The camera is ego-centred and heading-up (the car sits near the bottom, looking up the image), grayscale, 64×64. The rest of the setup is unchanged from Phase 1: same tracks, physics, progress reward, γ 0.99 and a fresh track every episode.

| # | Question | Setup | Concept |
|---|---|---|---|
| E1 | How much harder is driving from pixels than from rays? | SB3 PPO with NatureCNN on the camera image plus speed, against the Phase 1 ray policy at equal steps, then longer runs; 3 seeds | Convolutions, why pixels are sample-hungry |
| E2 | Can the car tell its speed from pictures alone? | Remove the speed number: 1 frame vs. a stack of 4 frames | Frame stacking, partial observability |
| E3 | Copy the expert or learn from scratch? | Behavior cloning from the Phase 1 ray policy (supervised learning on its camera images and actions), then PPO fine-tuning, against E1's from-scratch run | Imitation learning, distribution shift (the copy never sees its own mistakes) |
| E4 | Does shaking the picture help? | DrQ-style random shifts (pad the image, then crop at a random offset) on vs. off | Data augmentation, sample efficiency, generalization |
| E5 | Your own CNN encoder | Owner-written encoder (learner-owned, scaffolded piece by piece) in the owner's PPO from Phase 1, using the best recipe from E1–E4; saliency maps of what it looks at | Convolution layers, strides, receptive fields, saliency |

Optional, if time and interest allow: a chase/ego perspective camera (a view from behind and above the car, not straight down), as the roadmap's second step toward a real camera.

Then the gate check: the owner's CNN on the 20 held-out tracks, plus a saliency report.

## The camera (what the car sees)

- **Image:** 64×64 grayscale, straight down, centred on the car and turned with it ("up" = the car's heading). 0.625 m per pixel, so 40 m × 40 m: 35 m ahead, 5 m behind, 20 m to each side. Rays reached 40 m. Grass 40, road 110, white edge lines 230 (0.75 m wide). The road is about 10 pixels wide. Other parts of the track can show up in the image.
- **Sample:** [held-out track 10000 vs. the camera view](phase2-camera-sample.png). The 41% panel is the 6.9 m corner that caused most Phase 1 failures.
- **Speed (agent's benchmark, 1 thread):** SB3 PPO with NatureCNN trains at about 320 steps/s including evals, so 2M steps is about 1.7 h per run.

## Experiments

### E1 — Pixels vs. rays

- **Prediction (owner, before the run):** (1) At 300k steps: none of the 20. It'll just start learning how to control the car. (2) Yes, a longer run closes the gap, maybe around 1M steps. (Added after seeing the 64×64 sample, before any results:) Corners are less resolved, which looks like more noise in training, so it might learn more caution; if it completes the course it'll be slower.
- **Setup:** `python -m sdc.phase2.train --seed S --steps 2000000 --out runs/p2-e1-sS --label "PPO pixels (NatureCNN)"` for S = 0, 1, 2, 1 torch thread each, in parallel. 1 frame plus speed. Phase 1 base PPO settings (8 cars, 256 steps per update, else SB3 defaults). Held-out eval every 50k steps. Compared with Phase 1 E1 (rays, same settings).

## Gate check

## Handoff to next phase
