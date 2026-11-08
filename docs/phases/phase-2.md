# Phase 2 — Pixels → CNN

- **Status:** in progress; started 2026-11-08
- **Goal / gate:** replace rays with a low-res camera image (top-down crop first, then ego/chase view) through a small CNN encoder. **Gate:** the CNN policy reaches ≥ 90% of the Phase 1 policy's held-out lap completion (Phase 1 gate: 20/20, so ≥ 18/20 on tracks 10000–10019). A saliency/attention visualization shows the network keying on track edges, not artifacts.
- **Learn:** convolutions, frame stacking, why pixels are sample-hungry, data augmentation (DrQ-style random shifts), imitation learning (behavior cloning from the Phase 1 expert) vs. RL from scratch, saliency maps.
- **Code:** _to come_ (`sdc/racer/camera.py`, `sdc/phase2/`)
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

## Experiments

## Gate check

## Handoff to next phase
