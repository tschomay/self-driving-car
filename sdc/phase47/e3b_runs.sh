#!/usr/bin/env bash
# Phase 4.7 E3b: E3 again, with complete maps (every failed out-lap driven again: full_maps.py). Waits for the code
# review (runs/p47-overnight-review.ok). Scores the averaged pair: 5 noisy passes of the 150 selection runs.
set -u
cd "$(dirname "$0")/../.."
while [ ! -f runs/p47-overnight-review.ok ]; do sleep 60; done
python -m sdc.phase47.full_maps trainval > runs/p47-e3b-maps.log 2>&1 || { echo maps > runs/p47-e3b.failed; exit 1; }
for s in 0 1; do python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --map --seed "$s" --procs 1 \
  --train-map runs/p47-maps-train-full.json --val-map runs/p47-maps-val-full.json --out "runs/p47-e3b-s$s" \
  > "runs/p47-e3b-s$s.log" 2>&1 & done
wait
for s in 0 1; do [ -f "runs/p47-e3b-s$s/metrics.json" ] || { echo "seed $s failed" > runs/p47-e3b.failed; exit 1; }; done
python -m sdc.phase47.where e3b > runs/p47-e3b-where.log 2>&1 || { echo where > runs/p47-e3b.failed; exit 1; }
echo done > runs/p47-e3b.done
