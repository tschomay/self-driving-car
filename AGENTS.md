## Agent skills

### Issue tracker

Issues live as GitHub Issues on this repo, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root, created lazily as terms and decisions resolve. See `docs/agents/domain.md`.

## Project workflow: "start phase N" / "continue phase N"

This repo is a learning project that runs phase by phase. When the owner says **start phase N** or **continue phase N**:

1. **Load context.** Read `README.md`, then `docs/roadmap.md` (the working agreements and phase N's goal, concepts, and gate), then every log in `docs/phases/`, especially the previous phase's handoff section.
2. **Check the previous gate.** If phase N-1's log doesn't record a passing gate, say so and ask before continuing.
3. **Settle phase-entry decisions** listed in the roadmap's "Decisions deferred to phase exits", using `/grilling`: one round, numbered questions, a recommended answer for each. Example: long-run strategy at Phase 2 entry.
4. **Create or update the phase log** at `docs/phases/phase-N.md` from `docs/phases/TEMPLATE.md`. Set the README's status column for the phase to "In progress".
5. **Plan experiments** that teach the phase's concepts. A short plan (3–6 experiments) goes into the log before any long run.
6. **Predict → run → explain, for every experiment.**
   - Ask the owner for a 1–2 sentence prediction, and wait for it.
   - Run the experiment and publish a report page as an artifact: learning curves, a rollout GIF, key numbers.
   - Ask the owner to explain the result.
   - Record all three in the log. Never write the owner's prediction or explanation for them.
7. **Respect learner-owned code.** Some components belong to the owner: their own PPO (Phase 1) and CNN encoder (Phase 2). For those:
   - Scaffold the file with TODOs, explain each piece, and have the owner choose or dictate the core logic. They work from a phone, so short exchanges.
   - Don't silently write these pieces yourself.
   - Everything else (sim, plumbing, evaluation, reports) is the agent's to build.
8. **Survive ephemeral containers.** Checkpoint long runs and commit and push code, logs, and small results often. Don't commit large binaries; publish those through report artifacts instead.
9. **Gate check.** Run the gate evaluation on held-out tracks and record the verdict in the log. Update the README status column ("Gate passed" or "In progress, gate failed"), then fill in the handoff section.
10. **Stay low touch.** Do research yourself. Bring the owner only predictions, explanations, learner-owned code, and decisions that are genuinely theirs: money, time, scope, interest.

Compute: the cloud container has 4 CPU, 15 GB RAM, and no GPU, which is enough for Phases 0–1. Modal (free $30/month) needs an account and an API token, requested via the credential form when Phase 2 needs it. Budget ceiling: $0 until then, $30/month after.
