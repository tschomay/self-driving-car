#!/usr/bin/env bash
# Phase 4.6 E5 (owner, 2026-11-17): E4's sensors-only arm copying a teacher with margin (a_lat 2.5, a_brake 2.5).
# Seeds 0-2. E6 (average two E4 students per arm) runs alongside as the fourth process. About 2.2 h on 4 cores.
set -u
for s in 0 1 2; do
  python -m sdc.phase46.bc_loop --out "runs/p46-e5-s$s" --seed "$s" --rounds 16 --a-lat 2.5 --a-brake 2.5 \
    > "runs/p46-e5-s$s.log" 2>&1 &
done
( python -m sdc.phase46.ensemble --runs runs/p46-e4-nomap-s0 runs/p46-e4-nomap-s1 --out runs/p46-e6-nomap.json \
    > runs/p46-e6-nomap.log 2>&1
  python -m sdc.phase46.ensemble --map --runs runs/p46-e4-map-s0 runs/p46-e4-map-s1 --out runs/p46-e6-map.json \
    > runs/p46-e6-map.log 2>&1 ) &
wait
