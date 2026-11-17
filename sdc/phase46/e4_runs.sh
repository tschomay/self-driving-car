#!/usr/bin/env bash
# Phase 4.6 E4 (owner, 2026-11-17: "Let's increase training time"): E3 with 16 copying rounds instead of 8.
# Both arms (sensors only; + out-lap map), seeds 0-1, four runs at once. About 2.2 h on 4 cores.
set -u
for s in 0 1; do
  python -m sdc.phase46.bc_loop --out "runs/p46-e4-map-s$s" --seed "$s" --map --rounds 16 > "runs/p46-e4-map-s$s.log" 2>&1 &
  python -m sdc.phase46.bc_loop --out "runs/p46-e4-nomap-s$s" --seed "$s" --rounds 16 > "runs/p46-e4-nomap-s$s.log" 2>&1 &
done
wait
