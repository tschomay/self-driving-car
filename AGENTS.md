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

Compute: the cloud container has 4 CPU, 15 GB RAM, and no GPU. Run at most 4 training runs at once: Phase 3 runs use about 3 GB each, and 12 at once got one killed for memory (Phase 3 E8). Check every run produced its results; a killed run leaves an empty log. The CPU is enough for Phases 0–1. Modal (free $30/month) needs an account and an API token, requested via the credential form when Phase 2 needs it. Budget ceiling: $0 until then, $30/month after.

## Lessons from working with the owner

Collected as they come up (first batch: Phase 0, E1–E2). Apply them in every phase.

- **Give context before asking for a prediction.** Before the first experiment on a new environment or method, explain in plain terms what the task is, what the agent sees and controls, what ends an episode, how reward and return are counted, and how the experiment is set up. Include rules that change the answer (CartPole ends at 12°, not when the pole falls flat). The prediction should test intuition about learning, not knowledge of hidden rules. Never reveal the results.
- **Explain the mechanism, not just the name.** The owner wants to know *how* the algorithm works (how PPO decides an action was good, how it generalizes to unseen states). Expand every acronym the first time it appears and link it to the concept being taught. Offer this up front for each new method.
- **Make every visual self-explanatory.** GIFs carry overlays: who is driving, which episode or seed, step out of the cap, return so far, the last action, the quantity that ends the episode, a progress bar, and an end card saying why it ended and that it loops (`sdc/gifs.py`). Report captions say what one GIF covers: one episode, which seed, real speed.
- **Flag surprises before the owner finds them.** When a result looks odd (an untrained policy scoring 416), check it and explain it in the report instead of leaving the owner to wonder.
- **Keep runs reproducible.** Pin the torch thread count (default 1) and record seeds. A different thread count changes results even with the same seed.
- **Show controls visually, not as numbers.** Numbers are hard to read while a GIF plays. Lap GIFs draw steering and pedal as gauges under the map, and skid marks for tire slip (owner's request, Phase 3 E1).
- **Phone-sized messages.** Short sections, tables for prediction vs. result, one question at a time where possible.

## Git workflow: merge when a body of work is done

The owner has given standing permission: **when you complete a body of work, open a pull request to `main` and merge it without asking.** A body of work is a finished phase step, for example:

- a completed experiment with its log entry
- a gate check
- docs or plan updates
- a working feature

Before merging:

- Run the repo's fast checks (tests or lint, once they exist).
- Don't merge broken or half-finished work; push it to the branch and say what's left.

Don't ask "should I open a PR / merge?" in these cases. Just do it and report the merged PR.

PRs are squash-merged, so afterwards bring `main` into the working branch with `git merge origin/main` (not rebase plus force-push, which the session's permissions block).
