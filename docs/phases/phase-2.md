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
- **Setup:** `python -m sdc.phase2.train --seed S --steps 2000000 --out runs/p2-e1-sS --label "PPO pixels (NatureCNN)"` for S = 0, 1, 2, 1 torch thread each, in parallel. 1 frame plus speed. Phase 1 base PPO settings (8 cars, 256 steps per update, else SB3 defaults). Held-out eval every 50k steps. Compared with Phase 1 E1 (rays, same settings), re-run here: it reproduced 95%, 100%, 100% exactly. Wall clock 1.5–1.7 h per pixel run (≈340 steps/s against ≈3,300 for rays).
- **Result:** [report](https://claude.ai/artifact/7yPRhvTsiU6JGNTySK3Emf).
  - **At 300k steps: 0%, 0%, 60% held-out** (seeds 0, 1, 2). Rays: 95–100% by 50–75k.
  - **At 2M steps: 0/20 on every seed.** 55 of 60 attempts left the road, 5 timed out, 7–14% of the lap on average. Best moment of any seed: 70% (seed 2 at 250k), then back to 0%. Seeds 0 and 1 never passed 15%.
  - **Training got worse over time.** Mean return per training episode at 100k: 50, 37, 64; at 2M: 6, 10, 9. Early: about 8 m/s and nearly always crashing. Later, seeds 0 and 2 often sat still (uncrashed episodes averaged < 0.1 m/s); seed 1 crashed within about 4 s every episode.
  - **Not a bug (agent's checks).** A pipeline check passed: camera, frame handling and lap evaluation work end to end (details held back until E3, since they give away its answer). A diagnostic with separate actor and critic CNNs (SB3 shares one by default) did no better: 0–20% through 900k steps, seed 0. Gradient check on the final models: the critic's gradient is small, so it isn't swamping the policy at the end.
- **Explanation (owner, after the run):** Rays give the actual answer for where you are relative to the road. Pixels don't: the system has to learn position, speed, etc. in addition to just the controls. But why does it get worse? That normally points to overfitting or too high a learning rate. Something is exploding instead of converging. Could also be that the reward needs tuning.
- **Takeaway:** From pixels the network has to learn to see before it can learn to drive. With Phase 1's PPO settings that never happened in 2M steps (about 7× the ray budget, 60× the wall clock): learning peaked early and then fell apart. The owner's open hypotheses for the collapse: learning rate too high, overfitting, reward.
- **Decision (owner, after E1):** test the learning-rate hypothesis (E1b), then go on to behavior cloning (E3) and image shifts (E4). Frame stacking (E2) waits until a pixel policy learns.

### E1b — Does a smaller learning rate stop the collapse?

- **Prediction (owner, before the run):** I think it will help, but I expect other changes are required to get really good results. So yes, it stops the collapse; no, not driving.
- **Setup:** E1 setup with `--set learning_rate=LR`, LR ∈ {1e-4, 3e-5}, seeds 0–1, 1M steps, 4 in parallel on 1 thread each. The 3e-4 arm is E1's first 1M steps. About 50 min.
- **Result:** [report](https://claude.ai/artifact/CUHqJz1nDQCFo2R9QNcQ6k).
  - **Training return** (mean per episode, 100k-step windows, 200k → 1M): 3e-4 (E1): 64 → 35 and 28 → 12, falling. 1e-4: 55 → 54 (dip to 28 at 800k) and 56 → 93 (peak 113). **3e-5: 25 → 77 and 24 → 73, rising in every window.**
  - **Held-out at 1M:** 3e-4: 0%, 5%. 1e-4: 0%, 10%. 3e-5: 0%, 5%. Best eval of any run: 30% (1e-4, seed 1, 850k). None drives reliably.
  - **Driving style in training:** at 3e-5 the cars hold about 8 m/s and still crash in 95% of episodes, but get farther each time. At 3e-4 they slowed to 1–4 m/s.
  - Caveat: 2 seeds, 1M steps. The steady climb is clearest at 3e-5, the slowest learner.
- **Explanation (owner, after the run):** Smaller step allows it to learn details without overshooting. First it needs to learn how to recognize the track, and judge position and speed, then how to drive, then how to drive well. Too big of a step means the early stuff gets blown out rather than locked in. Slow to learn for the same reason: there's a lot more to learn.
- **Takeaway:** A pixel network has to build its perception while the policy on top of it is changing. A step size that was fine for 12 ray numbers wrecks that half-built perception; a 10× smaller step keeps it and learns steadily, but slowly. The learning rate fixes the collapse, not the speed of learning.

### E3 — Copy the expert, or learn from scratch?

- **Prediction (owner, before the run):** My intuition is limited here, but I think A will struggle with what to do if it isn't on the ideal line, B will pick up bad driving, and C will work.
- **Setup:** `sdc/phase2/bc.py`. Expert: Phase 1 SB3 PPO on rays (E1 seed 1: 100% held-out, 0.94× pure pursuit). 16 cars × 4,000 steps on random training tracks = 64k camera images (+ speed) labelled with the expert's action. **A:** `--noise 0` (the expert drives). **B:** `--noise 0.3` (Gaussian noise of std 0.3 added to the executed steering and throttle; labels stay clean). Network: SB3's PPO camera policy (NatureCNN + actor MLP), trained on mean-squared error to the expert action, Adam 3e-4, batch 256, 20 epochs; seeds 0–2. **C:** `python -m sdc.phase2.train --init runs/p2-e3-noisy-sS/model.zip --set learning_rate=3e-5 --steps 500000` (exploration noise starts at std 0.5; the critic starts untrained). Wall clock: A and B about 5 min each (15 s recording, the rest training); C about 28 min each.
- **Result:** [report](https://claude.ai/artifact/CJfoJeg5U1qNGBAeTzJ47G).
  - **A, clean copy: 20%, 10%, 25% held-out.** All failures were crashes, often late in the lap (up to 97%).
  - **B, copy with wobbles: 95%, 90%, 90% at 0.95× pure pursuit** (expert: 100% at 0.94×). Same data size, network and training time as A.
  - A's copy error ended *lower* than B's (MSE 0.007 vs. 0.009), yet A drives far worse.
  - **C, B + PPO: 25%, 90%, 60%**, worse than B. Held-out completion fell from 95/90/90% to 45/65/70% within the first 50k steps, then recovered only partly. Early training episodes crashed 96–99% of the time (exploration noise std 0.5, untrained critic). Training return still rose (59 → 148, 68 → 290, 112 → 313).
  - The pipeline check held back in E1 was a small version of B (noise 0.3, 25 epochs, a slightly different CNN head): 20/20 held-out.
  - B already meets the gate's completion bar (≥ 18/20) on all seeds, using SB3's CNN. The gate counts only the owner's encoder (E5).
- **Explanation (owner, after the run):** (1) Variety. It learns what to do in diverse situations. (2) Not sure. Feels like something about the PPO settings, like the learning rate is too fast or the amount it's allowed to change the weights is too much.
- **Agent's note on (2):** the learning rate was already 10× below E1's (3e-5), and the clip was PPO's usual ±20%. The bigger suspects: the critic starts untrained, so the first advantages are noise; and training adds exploration noise (std 0.5) that makes the cloned car crash in 96–99% of early episodes. PPO then learns from crashes the copy never made on its own. Common fixes: train the critic first with the policy frozen, and start with smaller exploration noise.
- **Takeaway:** Copying an expert from pixels works fast, but only if the data covers the expert's *recoveries*, not just its perfect laps. A clean copy drifts into states it has never seen, and its errors compound (distribution shift). Low training error doesn't measure that. Handing a good copy to RL can undo it at first, because RL's exploration and untrained critic start from scratch.

### E4 — Does shaking the picture help?

- **Prediction (owner, before the run):** Yes to both (shifts help PPO from scratch, and they rescue the clean copy).
- **Setup:** DrQ-style random shift (`camera.random_shift`): each training image slid by a random −3…+3 pixels in x and y (±1.9 m), edges filled by repeating edge pixels; lap evaluation never shifts. **A:** PPO from scratch, `--set learning_rate=0.00003 --env shift=3`, seeds 0–2, 1M steps; compared with E1b's learning rate 3e-5 runs (seed 2 added for this comparison). **B:** E3's clean copy with `--shift 3` (a fresh shift every time an image is used), seeds 0–2; compared with E3 A. Wall clock: A about 57 min per run; B about 7 min.
- **Result:** [report](https://claude.ai/artifact/HiYRCWVqQtofHQ6oWrxBYt). **Shifts hurt both.**
  - **PPO from scratch:** 0%, 0%, 0% held-out at 1M with shifts vs. 0%, 5%, 5% without. Training return (100k-step windows, 200k → 1M): without 25 → 77, 24 → 73, 21 → 77; with 19 → 39, 21 → 34, 21 → 37. Shifted cars crashed in essentially every training episode.
  - **Clean copy:** 0%, 0%, 0% with shifts vs. 20%, 10%, 25% without. Copy error ended about 10× higher (MSE 0.075–0.078 vs. 0.007): it couldn't fit the expert even on its own training images.
- **Explanation (owner, after the run):** Is it that there's nothing to learn because the shift is too big relative to the track size? What you'd do in those situations is entirely different, so nothing is learned?
- **Agent's note:** yes, and the size of the shift is only part of it. The car isn't drawn: it always sits at one fixed pixel, so *where the road is in the image* is the car's position on the road. Shifting the image 3 pixels shows a car 1.9 m to one side (a third of the road width) while the label is still the expert's action for the true position. Identical images then carry contradictory labels, so the copy error can't drop, and PPO gets equally mixed-up feedback. In DrQ's original tasks the important things move together inside the image, so a shift doesn't change the right answer. Here the image frame itself is the ruler.
- **Takeaway:** Augmentation must preserve the answer. A shift is harmless when what matters is how things in the image relate to each other, and harmful when the absolute pixel position *is* the information. Before using a data-augmentation trick, ask: would the expert's action change if the input changed this way?

### E2 — Can the car tell its speed from pictures alone?

- **Prediction (owner, before the run):** 1 picture struggles with braking, 4 pictures does fine.
- **Setup:** run with behavior cloning (E3 B: noise 0.3, 64k samples, 20 epochs) because PPO from pixels doesn't learn yet. `python -m sdc.phase2.bc --noise 0.3 --no-speed --frames F` for F ∈ {1, 4}, seeds 0–2. The stack holds the last F images (0.1 s apart), restarted at each new episode. Baseline: E3 B (speed number + 1 image). About 5 min (1 image) and 8 min (4 images) per run.
- **Result:** [report](https://claude.ai/artifact/AmZQVX3gUAKxii7Xetpoc2).
  - **Speed + 1 image (E3 B): 95%, 90%, 90%. No speed, 1 image: 0%, 5%, 0%. No speed, 4 images: 50%, 50%, 60%**, at 0.95–0.97× pure pursuit on finished laps.
  - 1 image, no speed (seed 0): 3 attempts never got going (top speed 0–10 m/s, then stopped, timed out); others crashed at 21–25 m/s without braking. Mean speed on held-out attempts 1.6 m/s (14.1 with the speed number). Copy error 0.047–0.053.
  - 4 images, no speed: normal pace (13.8 m/s mean), and it brakes. Every failure was a crash after slowing from 18–25 to 7–17 m/s: braked, but not enough or too late. Copy error 0.018 (0.009 with speed).
  - Motion between frames: 2.5 m (4 pixels) at 25 m/s, about 1 pixel at 7 m/s.

## Gate check

## Handoff to next phase
