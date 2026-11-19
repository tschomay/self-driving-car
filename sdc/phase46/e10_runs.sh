#!/usr/bin/env bash
# Phase 4.6 E10: copy the re-tuned teacher on the truggy physics; arm A no expo, arm B steering expo 0.4. Seeds 0-1.
set -u
for s in 0 1; do
  python -m sdc.phase46.bc_loop --truggy --rounds 16 --seed "$s" --out "runs/p46-e10-a-s$s" > "runs/p46-e10-a-s$s.log" 2>&1 &
  python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" --out "runs/p46-e10-b-s$s" > "runs/p46-e10-b-s$s.log" 2>&1 &
done
wait
for arm in a b; do
  python -m sdc.phase46.ensemble --runs "runs/p46-e10-$arm-s0" "runs/p46-e10-$arm-s1" --out "runs/p46-e10-$arm-avg.json" --procs 4 \
    > "runs/p46-e10-$arm-avg.log" 2>&1
done
