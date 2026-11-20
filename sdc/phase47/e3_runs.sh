#!/usr/bin/env bash
# Phase 4.7 E3: the student gets the out-lap map of the turns (truggy maps), copying the teacher that plans from the
# same map. Waits for the maps (python -m sdc.phase47.e3_maps) and the code review (runs/p47-e3-review.ok).
# Then scores the averaged pair and the map-following teacher: 5 noisy passes of the 150 selection runs.
set -u
cd "$(dirname "$0")/../.."
while [ ! -f runs/p47-maps-train.json ] || [ ! -f runs/p47-e3-review.ok ]; do sleep 60; done
for s in 0 1; do
  python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --map --seed "$s" --procs 2 \
    --train-map runs/p47-maps-train.json --val-map runs/p47-maps-val.json --out "runs/p47-e3-s$s" \
    > "runs/p47-e3-s$s.log" 2>&1 &
done
wait
for s in 0 1; do [ -f "runs/p47-e3-s$s/metrics.json" ] || { echo "seed $s failed" > runs/p47-e3.failed; exit 1; }; done
python -m sdc.phase47.where e3 > runs/p47-e3-where.log 2>&1
echo done > runs/p47-e3.done
