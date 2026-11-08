# Self-driving RC car (simulated first)

A hobby project for learning **CNNs and reinforcement learning** by teaching a car to race itself. It starts as a simple 2D top-down sim and climbs a ladder of more realistic simulators. A real off-road RC car with jumps is the long-term stretch goal.

The project is a learning vehicle. Lap times are secondary.

## Approach

- **Simulate first, cheaply.** The early phases run on the CPU of a Claude Code cloud session, which costs nothing and needs no GPU. Modal's free $30/month GPU tier comes in only when a phase actually needs it.
- **Phone-first workflow.** All work is driven from a phone through Claude Code sessions. Every training run publishes a report page with learning curves and a GIF of the car driving.
- **Learn by predicting.** Before each experiment the owner writes a 1–2 sentence prediction. After the run, they explain any gap between the prediction and the result. This is where most of the learning happens.
- **Hybrid depth.** A library (Stable-Baselines3) validates the pipeline first. Then the owner builds their own PPO (Phase 1) and CNN encoder (Phase 2) with the agent guiding, and benchmarks them against the library.
- **Gates, not vibes.** Each phase ends with a measurable benchmark, judged on **held-out tracks** the policy never trained on. You move on only when the gate passes.

Full plan, feasibility research, and sources: [`docs/roadmap.md`](docs/roadmap.md).

## Phases

| Phase | What we build | Gate to move on | Status |
|---|---|---|---|
| 0 | Python setup on the cloud CPU, plus the report pages | A toy RL task (CartPole) solved in under 10 minutes, with a report readable on a phone | Gate passed |
| 1 | Our own 2D racer with distance sensors and perfect grip | Finishes at least 95% of laps on 20 new tracks, no slower than 1.1× a hand-coded driver | Gate passed |
| 2 | Camera images fed through a CNN | Reaches at least 90% of Phase 1's lap completion, with a map showing it watches the track edges | Not started |
| 3 | Tire slip, dirt patches, delays, sensor noise, randomized physics | Finishes at least 85% of laps with physics outside its training range; then decide whether to go 3D | Not started |
| 4 | 3D terrain with jumps and an accelerometer, on a Modal GPU | Lands at least 80% of jumps wheels-down, within $30 a month | Go/no-go after Phase 3 |
| 5 | The real car | Not planned yet | Future |

Each phase's log (experiments, predictions, results, gate verdict) lives in [`docs/phases/`](docs/phases/).

## Starting a session

Open a new Claude Code session on this repo and say **"start phase N"** or **"continue phase N"**. The agent follows the protocol in [`AGENTS.md`](AGENTS.md).
