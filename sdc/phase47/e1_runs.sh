#!/usr/bin/env bash
# Phase 4.7 E1 (seeds 0-1, one core each) and, beside it, the Phase 4.6 gate rerun on the rebuilt car (two cores).
# Waits for sdc/phase46/rebuild.sh to finish first.
set -u
while [ ! -f runs/rebuild.done ]; do sleep 60; done
for s in 0 1; do
  python -m sdc.phase47.train_residual --seed "$s" --out "runs/p47-e1-s$s" > "runs/p47-e1-s$s.log" 2>&1 &
done
python -m sdc.phase46.gate --runs runs/p46-e10-b-s0 runs/p46-e10-b-s1 --out runs/p46-gate3 > runs/p46-gate3.log 2>&1
wait
echo done > runs/p47-e1.done
