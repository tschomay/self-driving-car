#!/usr/bin/env bash
# Phase 4.7 E2: harder, calibrated drills with a drill exam. Needs runs/p47-e2-prep.json (python -m sdc.phase47.e2_prep)
# and waits for the code review to pass (runs/p47-e2-review.ok). Then the fresh-loop check of kept and final checkpoints.
set -u
while [ ! -f runs/p47-e2-prep.json ] || [ ! -f runs/p47-e2-review.ok ]; do sleep 60; done
for s in 0 1; do
  python -m sdc.phase47.train_residual --seed "$s" --out "runs/p47-e2-s$s" --set drills=2 --set eval_procs=2 \
    > "runs/p47-e2-s$s.log" 2>&1 &
done
wait
python -m sdc.phase47.check --out runs/p47-e2-check.json --cars base runs/p47-e2-s0/model.pt runs/p47-e2-s1/model.pt \
  runs/p47-e2-s0/ckpt/1001472.pt runs/p47-e2-s1/ckpt/1001472.pt > runs/p47-e2-check.log 2>&1
echo done > runs/p47-e2.done
