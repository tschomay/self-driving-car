# Beyond the teacher: literature review for a sensors-only RC racer

2026-11-20. Background literature review for the Phase 4.6/4.7 question: how do we get past a hand-coded teacher (91% finish, ~30.5–31 s laps) to "a good racer", on 4 CPU cores?

Every claim below was checked against the primary source's text (arXiv or publisher PDF), except where it is marked **[unverified]**. Step counts marked *(derived)* are my own arithmetic from numbers the paper gives. nature.com could be read for Swift. GT Sophy's full text came from the author-hosted PDF linked from the paper's co-author's page.

## TL;DR

- Every system that beat its baseline used **RL on a dense progress reward, with the RL policy (or its critic) given privileged state**. Hand-coded or imitation teachers set a ceiling. In LbC, the teacher itself was cloned from an autopilot, and the student only matched it.
- The budgets are **about 100× ours**. Swift: 1e8 PPO steps. F1TENTH RL racer: 1.2e8. RMA: 1.2e9. GT Sophy: about 1.6e9 *(derived)*. Every PPO fine-tune in this repo has used 1e6 steps with a 2,048-step batch. On 4 cores, about 2e8 steps a day is enough for a privileged teacher at the Swift/F1TENTH scale.
- "Curse of chaos" (PIPPS) is mainly an argument against **pathwise (differentiable-sim) gradients**, not against PPO. PIPPS finds likelihood-ratio gradients robust. Our PPO problem is return variance, and the literature's tools for that are dense rewards, a critic, large batches and reset distributions spread over the whole track (an "initial-state buffer").
- There is no published RL work on RC-car *jumps*. The closest work (DART, 2026; Pokhrel et al., 2025) finds that in-air authority is small and **the decisive lever is the speed before takeoff**.

## 1. State-of-the-art agile control and transfer to sensor-limited policies

| System | Who learns what | Teacher / privileged inputs | Reward | Samples and compute | Distillation |
|---|---|---|---|---|---|
| **Lee et al. 2020**, quadruped, *Science Robotics* ([arXiv 2010.11251](https://arxiv.org/abs/2010.11251)) | Teacher by RL (TRPO), student by imitation | Privileged *x_t*: terrain heights at 9 points around each foot, contact states and forces, friction, external disturbances. Encoded to a latent *l̄_t* | Rewards advancing faster toward the goal, plus smoothness terms (details in supp. S4) | **[unverified]**: Table S7 is in the supplement, which I did not read | TCN student on proprioceptive history. Loss = (teacher action − student action)² **+ (teacher latent − student latent)²**. DAgger: the student drives, the teacher labels. Adaptive terrain curriculum (particle filter) in both stages |
| **Learning by Cheating**, Chen et al. 2019 ([arXiv 1912.12294](https://arxiv.org/abs/1912.12294)) | Privileged agent **imitates an expert (autopilot)**. No RL | Bird's-eye-view map with ground-truth layout and traffic | none | — | Student imitates the privileged agent on-policy (DAgger-like). The teacher is a "white box", queried for *all* command branches. The teacher is bounded by the autopilot, which is **structurally the same as our pipeline** |
| **RMA**, Kumar et al. 2021 ([arXiv 2107.04034](https://arxiv.org/abs/2107.04034)) | Base policy + env-factor encoder by PPO. Adaptation module by supervised learning | Mass, CoM, friction, motor strength, terrain height → extrinsics *z_t* | Bioenergetics-inspired: forward progress capped at 0.35 m/s, energy and smoothness penalties | PPO, 15,000 iterations × 80,000-step batches, **1.2 billion steps, about 24 h** on a desktop with 1 GPU. Adaptation module: 80 M steps, 3 h | Regress *ẑ_t* from 50 steps of state-action history. **On-policy**: the base policy is rolled out with the *predicted* *ẑ_t*, to cover off-nominal states |
| **Swift**, Kaufmann et al. 2023, *Nature* ([s41586-023-06419-4](https://www.nature.com/articles/s41586-023-06419-4)) | Policy by PPO directly on the *deployable* observation (VIO estimate + gate corners + last action). **No teacher** | **Asymmetric critic**: "the value network … can access privileged information about the environment that is not accessible to the policy" | Dense: progress toward the next gate + perception (keep the gate in view) − body-rate and Δaction penalties − crash (5.0) | **100 parallel agents, 1,500-step episodes, 1e8 steps in 50 min** (i9 + RTX 3090). Fine-tune 2e7 steps | Instead of distilling, it fine-tunes in a sim augmented with residual dynamics (k-NN) and observation-noise (GP) models fitted to about 50 s of real flight |
| **GT Sophy**, Wurman et al. 2022, *Nature* ([s41586-021-04357-7](https://www.nature.com/articles/s41586-021-04357-7); [author PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)) | QR-SAC (distributional SAC, 32 quantiles), off-policy, asynchronous | Full game state through an API, plus 60 course points along each edge and the centreline spanning about the **next 6 s** of travel. No sensor student (it plays the game) | Course progress + off-course, wall (∝ speed²) and tyre-slip penalties (+ racing terms) | Time trial: **10 PS4s × up to 20 cars, about 8 days**. Laps at default-AI level after about 8 h, faster than 95% of humans in 1–2 days, then 9+ more days (>45,000 driving hours) "shaving off tenths of seconds". At 10 Hz that is ~1.6e9 steps *(derived)* | — |
| **Fuchs et al. 2021**, GT Sport ([arXiv 2008.07971](https://arxiv.org/abs/2008.07971)) | SAC | Policy sees velocity, acceleration, heading vs centreline, 180° rangefinders (100 m), previous steering, a wall-contact flag and **N curvature samples ahead** | "Course-progress proxy reward" + wall-contact penalty ∝ speed². They say lap time is "a very sparse reward signal" | 4 PS4s × 20 cars at 10 Hz. Beat the best human after **56–73 h** (946,453 km driven); ~1.6–2.1e8 steps *(derived)*. Warm-starting from behaviour cloning of human data **did not improve the final performance** | — |
| **Song et al. 2021**, drone racing ([arXiv 2103.08624](https://arxiv.org/abs/2103.08624)) | PPO | Full state | Path-progress reward + optional safety reward + terminal crash penalty | Up to 100 parallel envs. **Distributed initialisation**: start at all path segments, later from states sampled from previous trajectories. Random-track curriculum | Within **5.2% of the time-optimal (CPC) lap** |
| **Song et al. 2023**, "Reaching the limit", *Sci. Robotics* ([arXiv 2310.10943](https://arxiv.org/abs/2310.10943)) | PPO (SB3, Flightmare) | Full state (motion capture) | Gate progress − 0.01‖ω‖, −10 per collision, +10 at the finish | "Hundreds of environments", **about 10 min** on a workstation. **Initial-state buffer** of successful states "greatly improves sample efficiency" | — (state policy) |

**Key finding (Song 2023):** "RL does not outperform OC because RL optimizes its objective better… RL outperforms OC because it optimizes a better objective". RL maximises task progress directly instead of tracking a planned trajectory. On a realistic (mismatched) drone model, time-optimal trajectory tracking and contouring MPC had success rates of 0% and 20%. RL had 100%, with the best lap time. This maps directly onto our hand-coded teacher, which plans a speed profile and tracks it.

## 2. Asymmetric actor-critic as an alternative to distillation

- **Pinto et al. 2017** ([arXiv 1710.06542](https://arxiv.org/abs/1710.06542)): "the critic is trained on full states while the actor … gets rendered images". This "significantly improves performance" in simulation and transfers to a real robot.
- **Swift** (above) uses exactly this. Its actor sees only the deployable observation, and its critic gets privileged information in addition. A world-champion-level result came **without any teacher**.
- **DreamWaQ** (Nahrendra et al. 2023, [arXiv 2301.10602](https://arxiv.org/abs/2301.10602)): proprioceptive quadruped. Its finding: "the interplay between the policy and value networks in actor-critic algorithms is sufficient for learning a robust locomotion policy that could implicitly imagine the privileged observations". The critic gets privileged observations, and an estimator network supplies context.
- **When it breaks: Baisero & Amato 2022** ([arXiv 2105.11674](https://arxiv.org/abs/2105.11674)). A *state-only* critic is theoretically flawed under partial observability: it is biased and "limit[s] its ability to address partial observability". Their unbiased variant conditions the critic on **history and state** together, and it converged to better policies or faster on strongly partially observable tasks. Practical reading: give the critic the privileged state *and* the actor's GRU input/history. Asymmetry helps most when the missing information is either inferable from history (grip, mass) or irrelevant to the best action.
- **Why teacher-student loses something** (the "imitation gap" or "realizability gap"):
  - Weihs et al. 2021, ADVISOR ([arXiv 2007.12173](https://arxiv.org/abs/2007.12173)): when the teacher uses privileged information, "this information is marginalized during imitation learning", which can give poor results. ADVISOR weights the imitation and RL losses per state.
  - Fu et al. 2022, ROA ([arXiv 2210.10044](https://arxiv.org/abs/2210.10044)): the teacher "may provide supervision that is impossible for the student network to predict, resulting in a realizability gap". ROA trains the policy and the history-based estimator jointly by RL, regularising the privileged latent toward the estimable one.
  - Both describe the ~4-point student-teacher gap seen here. They also explain why a better-landing but more map-dependent teacher gave a worse student (Phase 4.6 retune). That last reading is my interpretation.
- **Teacher-guided PPO with an advantage gate: TADPO** (Wu et al. 2026, [arXiv 2603.05995](https://arxiv.org/abs/2603.05995)), off-road RL on a full-scale vehicle. It extends PPO with a clipped loss on teacher rollouts that applies "only when … the teacher rollout's return exceeds the student policy's expected return". The teacher may use privileged observations, and the teacher was MPPI-based. Self-Imitation Learning (Oh et al. 2018, [arXiv 1806.05635](https://arxiv.org/abs/1806.05635)) is the same idea applied to the agent's own good episodes. **This is a principled replacement for the fixed "anchor loss" tried in Phase 4.7.**

## 3. RL for RC cars, off-road and jumps

- **FastRLAP** (Stachowicz et al. 2023, [arXiv 2304.09831](https://arxiv.org/abs/2304.09831)), 1/10 scale, real world:
  - Algorithm: RLPD, i.e. SAC-style with a critic ensemble and a high update-to-data ratio. Each batch is **50% from one slow demonstration lap and 50% online**, which they "found … to be critical".
  - Reward: velocity toward the next checkpoint, with collision and stuck penalties.
  - Practice is autonomous between checkpoints, with automatic recovery.
  - Result: aggressive driving "comparable to a human expert within 20 minutes" of practice.
  - It builds on pre-trained visual representations and offline pre-training.
- **RLPD** (Ball et al. 2023, [arXiv 2302.02948](https://arxiv.org/abs/2302.02948)) is the underlying recipe: symmetric 50/50 sampling, LayerNorm in the critic, large ensembles. This is how to use *teacher data* inside off-policy RL.
- **Czechmanowski et al. 2025, F1TENTH** ([arXiv 2504.02420](https://arxiv.org/abs/2504.02420)), IROS 2025:
  - Setup: PPO with 400 parallel sims, 1,024-step rollouts at 0.05 s, **120 M steps**. Reward = centreline progress, −1 for leaving the track, with a reset to a random centreline position. Observations include curvature and width at N points ahead.
  - Result: it beat MPC and expert human RC drivers on real hardware.
  - Relevant to us: wider domain randomisation gave **slower, more conservative policies**, "since the policy has no access to action history or recurrent memory to infer the underlying system dynamics". Accurate system identification beat broad randomisation.
- **Werner et al., CoRL 2023** ([PMLR v229](https://proceedings.mlr.press/v229/werner23a.html)), 1/10-scale multi-team racing:
  - Hierarchical: high-level goals trained with DecSARSA, low-level control with PPO.
  - Self-play against checkpoints and pure-pursuit opponents, with a dominant sparse team-rank reward plus overtaking and progress terms.
  - Compute: **2,048 parallel envs on one RTX 2080 Ti, about 8 h**. Low-level inference at 20 Hz on CPU.
- **Evans et al. 2023, Trajectory-aided learning** ([arXiv 2306.07003](https://arxiv.org/abs/2306.07003)), F1TENTH sim, TD3, 100k steps. The reward is the distance between the agent's action and a pure-pursuit tracker following a minimum-curvature, minimum-time trajectory (Heilmeier et al.). It gives higher completion at high speed. In effect it is imitation-shaped RL, so it is bounded by that planner.
- **TC-Driver** (Ghignone et al. 2022, [arXiv 2205.09370](https://arxiv.org/abs/2205.09370)): RL conditioned on a reference trajectory for robust racing (downloaded; not studied in depth).
- **Off-road and jumps:**
  - **Xu et al. 2024** ([arXiv 2409.02383](https://arxiv.org/abs/2409.02383)): PPO + a terrain-difficulty curriculum for steep, rocky terrain. Reward = progress + a penalty above a roll/pitch threshold + a timeout penalty.
  - **Pokhrel et al. 2025, "Dom, cars don't fly!"** ([arXiv 2503.19140](https://arxiv.org/abs/2503.19140)): in-air attitude control of a small ground robot using only throttle and steering, with a *sampling-based planner* on a learned hybrid kinodynamic model targeting the landing pose. Not RL.
  - **DART** (Hu et al. 2026, [arXiv 2607.29011](https://arxiv.org/abs/2607.29011)), full-scale simulation only:
    - "the airborne phase is barely controllable". The wheel angular-momentum budget caps recoverable pitch rate, so "the decisive leverage lies before takeoff".
    - A certified feasible-takeoff set plus pre-takeoff speed shaping raised on-target landings from 0/30 to 30/30.
  - **No RL paper on RC-car jumps or tabletops/doubles was found.** A citation in DART to Zhao et al., "Learning when to jump for off-road navigation" (RSS 2026) **[unverified, not located]**.

## 4. Handling chaotic dynamics and high-variance returns

- **What PIPPS actually says** (Parmas et al. 2018, [arXiv 1902.01240](https://arxiv.org/abs/1902.01240)): with long chains of nonlinear dynamics, "the direction of the gradients becomes essentially random". It shows "reparameterization gradients suffer from the problem, while likelihood ratio gradients are robust". PPO is a likelihood-ratio method. So chaos does not invalidate PPO. It makes single-episode outcomes a high-variance signal, which is a sample-size and credit-assignment problem.
- **Dense progress rewards:**
  - Fuchs: progress proxy with discounting "to reduce the variance".
  - GT Sophy: course progress; it also filtered network-delay transitions "to reduce variance in time-sensitive rewards".
  - Swift, Song 2021/2023 and F1TENTH all use progress too.
  - **No system relies on a sparse crash/finish signal alone.** Crash penalties are small (Swift 5.0, Song −10, F1TENTH −1) relative to the accumulated progress reward.
- **Reset distributions** (start states spread over the whole task):
  - Song 2021/2023: initial-state buffer of states from successful trajectories.
  - DeepMimic, Peng et al. 2018 ([arXiv 1804.02717](https://arxiv.org/abs/1804.02717)): "reference state initialization and early termination" are "critical for achieving highly dynamic" skills.
  - Swift: reset at a random gate, perturbed around previously observed states.
  - Phase 4.7's drills were this idea. The difference is the budget: 1M steps, against 1e8.
- **Large batches and many environments:**
  - Rudin et al. 2021 ([arXiv 2109.11978](https://arxiv.org/abs/2109.11978)): batch size is "a crucial hyper-parameter". They use thousands of robots, much larger mini-batches "stabiliz[e] the learning process", and a game-inspired terrain curriculum.
  - RMA used 80,000-step batches, Swift 150,000 (100 agents × 1,500 steps).
  - Ours, 2,048 per update, is 40–75× smaller.
- **Smoothness penalties:** body-rate and Δaction (Swift), body rate (Song 2023), energy and jerk (RMA). These regularise the policy, which reduces how much action noise amplifies.
- **Differentiable simulation:**
  - SHAC (Xu et al. 2022, [arXiv 2204.07137](https://arxiv.org/abs/2204.07137)) handles non-smoothness with "a smooth critic function" and "a truncated learning window".
  - Song et al. 2024 ([arXiv 2403.14864](https://arxiv.org/abs/2403.14864)) gets gradients from a **simplified surrogate model** while a non-differentiable simulator runs the forward dynamics, because with chaotic contact "these gradients can become unmanageable".
  - Both rely on GPU-parallel, differentiable simulation. Jumps and landings are the most discontinuous part of our task. *Assessment: poor fit for MuJoCo on 4 CPUs.*
- **Optimal control as the teacher:**
  - Foehn et al. 2021, time-optimal (CPC, [arXiv 2108.04537](https://arxiv.org/abs/2108.04537)) is the reference for minimum-time planning. Per Song 2023, full-dynamics time-optimal planning "requires hours of computation" offline, and tracking it crashed under model mismatch unless the thrust limit was lowered (6.12 s against an RL lap of about 5.1 s).
  - Evans et al. shows the use as a reward guide.
  - Takeaway: OC makes a good **initialiser or reward guide**, but it is brittle as the final controller under randomised physics.

## 5. Sample efficiency on a small CPU budget

| Work | Algorithm | Steps to good or superhuman |
|---|---|---|
| Song 2023 drone | PPO | about 10 min of workstation training (step count not stated in the main text) |
| Swift | PPO | 1e8 (+2e7 fine-tune) |
| F1TENTH (Czechmanowski 2025) | PPO | 1.2e8 |
| RMA | PPO | 1.2e9 (+8e7 adaptation) |
| Werner 2023 | PPO (+ DecSARSA) | 2,048 envs × 8 h (total steps **[unverified]**) |
| Fuchs 2021 | SAC | ~1.6–2.1e8 *(derived)*, 56–73 h |
| GT Sophy | QR-SAC | ~1.6e9 *(derived)*. Competent laps after about 8 h ≈ 6e7 *(derived)* |
| Evans 2023 (F1TENTH sim) | TD3 | 1e5, but imitation-shaped reward and an easier task |
| FastRLAP (real) | RLPD (SAC + ensemble, high UTD) | about 20 min of real driving, with pre-training and a demo lap |

- **Is ~1–2e8 steps a day on 4 cores enough?** Yes, plausibly, for a privileged-state teacher. It is the same order as Swift (1e8), F1TENTH (1.2e8) and Fuchs (~2e8). Two caveats:
  - Our task adds jumps and wide domain randomisation, which RMA needed 1e9 steps for (on legged locomotion).
  - All four cores go to one run, so there are no parallel seeds.
- **Is off-policy (SAC/DroQ/REDQ/TD-MPC2) much more sample-efficient?**
  - Within *simulated* racing, the evidence does not show a large gap. Fuchs and GT Sophy (both SAC-family) used ~1e8–1e9 steps, the same order as PPO racers.
  - REDQ ([arXiv 2101.05982](https://arxiv.org/abs/2101.05982)) and DroQ ([arXiv 2110.02034](https://arxiv.org/abs/2110.02034)) do show big gains on MuJoCo benchmarks through a high update-to-data ratio. But REDQ is "much less computationally efficient", which is why DroQ exists.
  - On CPU, a high update-to-data ratio turns cheap sim steps into expensive gradient steps. When the simulator is fast and compute-bound, as ours is, sample efficiency is the wrong metric. Wall-clock time is what counts.
  - TD-MPC2 ([arXiv 2310.16828](https://arxiv.org/abs/2310.16828)) was evaluated on 104 control tasks with one set of hyperparameters. **No car-racing result was found [unverified for racing]**, and its planning makes training and inference heavier than PPO.
  - The strongest off-policy argument for us is **RLPD-style reuse of teacher/DAgger data**, as FastRLAP did. Raw sample counts are not the argument.

## Implications for this project (ranked)

**1. A privileged RL teacher trained on progress, at the literature's budget.** Highest expected impact; feasible. This is the only route in the literature to *exceed* a hand-designed driver (Song 2023, GT Sophy, Fuchs, F1TENTH).

- **Inputs:** true state, randomised physics parameters and the track preview the hand-coded teacher uses (curvature ahead, jump kind and distance, as GT Sophy's 6 s preview does).
- **Reward:** metres of progress per step, with a crash penalty of about a few seconds' worth of progress, plus small Δaction and body-rate penalties.
- **Start:** a BC copy of the current teacher, because Phase 3.5 and 4 E1 showed PPO from scratch never found speed or air control. Fuchs is a counter-example in GT Sport, so a from-scratch control arm is cheap insurance.
- **Resets:** the Song/DeepMimic reset distribution. Start everywhere on the loop, especially before takeoffs, from states the teacher or the current policy actually visited.
- **Batch:** ≥ 50k steps per update.
- **Budget:** 5e7–2e8 steps, about 6–24 h using all 4 cores.
- **Then:** distil with the existing DAgger pipeline, which already transfers ~96% of teacher reliability. Add a latent-matching loss (Lee 2020) and keep the teacher's privileged inputs to what the GRU could plausibly infer (ROA realizability gap).

**2. Asymmetric actor-critic fine-tuning of the sensor student, with an advantage-gated teacher loss.** High feasibility (a change to the existing PPO); medium-high impact.

- Give the critic the true state and track preview *plus* the GRU features (Baisero & Amato). This cuts advantage variance far more than any return trick, and it is how Swift trained a sensor-level policy directly.
- Replace the fixed anchor loss with TADPO or SIL-style imitation that applies only where the teacher's (or the policy's own) return beats the critic's value. The policy is then free to leave the teacher where it is better.
- Run this after #1 if possible, using the RL teacher as the guide.

**3. Fix landings before takeoff, not in the air.** Medium impact on reliability; high feasibility. Both off-road jump papers (DART, Pokhrel) put the leverage in takeoff speed and attitude, given limited in-air authority. The Phase 4.6 physics review also found in-air throttle control weak at speed.

- Add a takeoff-speed and pitch-rate term (or a learned "feasible takeoff" check) to the hand-coded teacher, then re-run DAgger.
- In #1, oversample resets 1–2 s before each jump.
- This is the cheapest way to move the 91% ceiling. It will not move lap time much.

**4. Off-policy RL with teacher data (RLPD-style SAC/DroQ).** Medium impact; lower feasibility on CPU.

- It is attractive *only* to reuse the large store of teacher and DAgger transitions (50/50 symmetric sampling, LayerNorm critics).
- On 4 CPUs the update-to-data ratio is compute-bound, and the sim-racing literature does not show it beating PPO on sample count.
- Try it after #1 if PPO stalls.

**Not recommended now:**

- Differentiable simulation (SHAC, Song 2024): it needs a differentiable or surrogate model, and contact-rich landings are where the gradients break.
- A minimum-time OC teacher as the final controller: brittle under the randomised physics, though useful as a speed-profile initialiser.
- Feeding the student a map again: the imitation-gap literature predicts it helps only if the student can use it as well as the teacher does.

## Sources not fully read

- Lee 2020 supplementary (sample counts, reward weights).
- GT Sophy Extended Data Table 1 (reward weights) and Supplementary Information.
- Werner 2023 Appendix C (total step count).
- TC-Driver (downloaded, skimmed only).
- Zhao et al., RSS 2026 (not located).
