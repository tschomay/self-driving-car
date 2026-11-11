# Self-driving RC car — learning roadmap

Status: **confirmed 2026-11-04** ([Confirm learning style, budget, and phase plan](https://github.com/tschomay/self-driving-car/issues/2)).

## Working agreements

- **Learning depth: hybrid.** Phase 0 uses Stable-Baselines3 to validate the pipeline. In Phase 1 you write a minimal PPO yourself and benchmark it against the library. Phase 2 uses a CNN encoder you write.
- **Budget: $0 to start.** Phases 0–1 run on the cloud container's CPU. Modal (free $30/month) gets set up at the start of Phase 2, and only if run lengths need it.
- **Cadence: ~2–4 h/week.** Every experiment gets a 1–2 sentence written prediction before the run and a short explanation after it.
- **Gates** as listed below. Phase 4 is an explicit go/no-go decision at the Phase 3 exit.
- **Decisions deferred to phase exits:**
  - Phase 2 entry: how long runs survive ephemeral sessions (checkpoint-and-push vs. detached Modal jobs).
  - Phase 3 exit: whether to do Phase 4, and if so which 3D engine (MuJoCo/MJX, Isaac Lab, Genesis, ...). **Decided (2026-11-11):** go to Phase 4 after a Phase 3.5 speed detour; the engine choice and Modal setup move to Phase 4 entry.
  - Phase 3.5 entry: the gate (reliability plus a lap-time target).

## Purpose

Learn CNNs and reinforcement learning (RL) by building toward a car that races itself, starting in cheap simulation. The project is a learning vehicle. Lap times are secondary.

## Feasibility findings (2026-11-04)

### 1. Compute from a phone

- **Phases 0–2 need no GPU.** The Claude Code cloud container (4 CPU, 15 GB RAM, no GPU) can run training directly. I measured Gymnasium `CarRacing-v3`, a pixel-based 2D racer, at about 167 env steps/s per core. That makes 1M steps of pixel training an hours-scale CPU job. A custom vectorized 2D sim with ray sensors runs 10–100× faster, so training takes minutes.
- **Catch: the container is ephemeral.** Long runs must checkpoint and push results. Multi-hour runs belong on detached compute.
- **Recommended GPU/detached compute: [Modal](https://modal.com/pricing).** It's driven from a Python SDK, so I can launch jobs from here and you never open a laptop. The Starter plan is $0 with **$30/month free credit**. Prices: T4 $0.59/h, L4 $0.80/h, A100-40GB $2.10/h, CPU $0.047/core-h. The network from this container reaches `api.modal.com`. Setup is one account plus one API token saved as an environment secret.
- **Free fallback: Kaggle**, which gives roughly 30 GPU-h/week on P100/T4 and can be driven through `kaggle kernels push`. It's clunkier, and the quota figures come from [third-party sources](https://www.gmicloud.ai/ja/blog/best-free-gpu-cloud-options-for-ai-startups-and-researchers).
- **Viewing from a phone:** each training run publishes a small report page with learning curves and a GIF of rollouts.

### 2. Is it reasonable?

- **2D and pixel phases: yes, well-trodden.** Small-scale cars trained in sim are a standard hobby and research setup: DonkeyCar sim ([L2D benchmark](https://arxiv.org/pdf/2008.00715)), AWS DeepRacer, and [F1TENTH](https://neurips.cc/virtual/2019/demonstration/15447).
- **Sim-to-real for 1/10 cars works on flat tracks.** Examples: zero-shot multi-car racing ([Werner et al., CoRL 2023](https://proceedings.mlr.press/v229/werner23a.html)), residual RL ([RLPP](https://arxiv.org/pdf/2501.17311)), and drifting ([arXiv 2507.23339](https://web3.arxiv.org/pdf/2507.23339)).
- **Off-road with jumps is frontier territory.** I found no paper doing sim-to-real RL for RC jumps. [FastRLAP](https://arxiv.org/pdf/2304.09831.pdf), the closest work, trains aggressive off-road driving on a real RC car. It does that *in the real world* precisely because simulating off-road terrain is hard. The project is reasonable if the real car is the long-term stretch goal and the simulator ladder is the main event.

### 3. Will you learn, or just watch me build a sim?

The sim is cheap: a few hundred lines for 2D. The learning is in the parts that need your judgement:

- **MDP design**: what the car observes, what counts as reward, when an episode ends.
- **Reward hacking**: you *will* watch the car exploit your reward.
- **RL debugging**: why curves collapse, which hyperparameters matter.
- **CNN representations**: what the network actually looks at.
- **Generalization**: does it drive tracks it never saw?
- **The sim-to-real gap**: domain randomization, latency, noise.

The risk is real if I build everything and you only read summaries. Every phase therefore includes **predict → run → explain** exercises. Before each experiment you write down a prediction. Afterwards you explain the gap between the prediction and the result.

## Phases

Each phase has a goal, the concepts it teaches, and a **gate**: a measurable benchmark that must pass before moving on. Gates are judged on **held-out tracks** (procedurally generated, never trained on).

### Phase 0 — Environment and loop (CPU, this container)
- **Goal:** a reproducible Python env and a published training report, built on PPO on a toy task.
- **Learn:** agent/env loop, returns, policy gradient intuition, what a learning curve tells you.
- **Gate:** a single command trains PPO to solve `CartPole-v1` (avg return ≥ 475) in under 10 min, and a report page with curve + GIF is viewable on your phone.

### Phase 1 — 2D top-down racer with ray sensors (CPU)
- **Goal:** our own minimal vectorized sim. It has kinematic bicycle-model physics with perfect traction, procedural closed tracks, 7–15 ray-cast "lidar" distances plus speed as the observation, and an MLP policy.
- **Learn:** reward shaping (progress vs. speed vs. penalties), reward hacking, episode design, PPO hyperparameters, overfitting to one track vs. generalizing.
- **Baseline:** a hand-written pure-pursuit controller.
- **Gate:** the RL policy completes ≥ 95% of laps on 20 held-out tracks with mean lap time ≤ 1.1× the pure-pursuit baseline.

### Phase 2 — Pixels → CNN (CPU; Modal optional)
- **Goal:** replace rays with a low-res camera image (top-down crop first, then ego/chase view) through a small CNN encoder.
- **Learn:** convolutions, frame stacking, why pixels are sample-hungry, data augmentation (DrQ-style random shifts), imitation learning (behavior cloning from the Phase 1 expert) vs. RL from scratch, saliency maps.
- **Gate:** the CNN policy reaches ≥ 90% of the Phase 1 policy's held-out lap completion. A saliency/attention visualization shows the network keying on track edges, not artifacts.

### Phase 3 — Realistic dynamics and robustness (CPU/Modal)
- **Goal:** dynamic bicycle model with tire slip, throttle/brake, low-grip "dirt" patches, actuation latency, sensor noise, and **domain randomization** of friction, mass, and latency.
- **Learn:** vehicle dynamics, why perfect-traction policies fail, domain randomization as sim-to-real insurance, robustness evaluation; drifting may emerge on its own.
- **Gate:** on held-out tracks, with physics parameters drawn from a range *wider* than training, completion is ≥ 85% and the robustness curve is published.
- **Go/no-go:** after this phase, decide whether to go 3D (cost, interest, time).

### Phase 3.5 — Speed (CPU) — added at the Phase 3 exit
- **Goal:** close the lap-time gap to the hand-coded driver (Phase 3's reliable car laps in about 34–42 s vs. 25.4 s) without losing robustness.
- **Learn:** the speed vs. reliability trade-off, multi-objective model selection, reward design for speed, what knowing the track (dirt locations) is worth.
- **Gate:** set at Phase 3.5 entry. Provisionally: ≥ 85% gate-range completion on a fresh test set, with mean lap ≤ about 1.15× the hand-coded driver.

### Phase 4 — 3D terrain with jumps (Modal GPU)
- **Goal:** a 3D rigid-body sim with heightfield terrain, ramps, and an IMU (accelerometer + gyro) in the observation. Candidate engines: MuJoCo/MJX, Isaac Lab, Genesis; to be decided at Phase 4 entry.
- **Learn:** 3D dynamics, partial observability (the car is airborne, so it uses memory/recurrence), using throttle to control pitch mid-air, massively parallel GPU simulation.
- **Gate (provisional):** ≥ 80% of jumps landed wheels-down on a held-out jump track, within the $30/month Modal budget.

### Phase 5 — Real car (future, not planned in this map)
Hardware bill of materials (1/10 off-road chassis, onboard compute like a Raspberry Pi 5 or Jetson, camera, IMU), deployment, and real-world fine-tuning. Sketched here so the earlier phases keep sim-to-real in mind, for example by keeping camera resolution and control rate realistic.
