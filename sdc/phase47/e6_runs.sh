#!/usr/bin/env bash
# Phase 4.7 E6: the sensors-only base recipe copies E6a, the RL racer that sees only 4 m of track
# (runs/p47-e6a, 6M steps: 124.0 finished per pass at 25.87 s, race 541.7 s), 2 seeds, then where.py scores it.
set -u
cd "$(dirname "$0")/../.."
R=runs/p47-e6a/ckpt/6029312.pt
for s in 0 1; do python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" --procs 2 \
  --teacher-racer "$R" --teacher-preview 4.0 --out "runs/p47-e6-s$s" > "runs/p47-e6-s$s.log" 2>&1 & done
wait
for s in 0 1; do [ -f "runs/p47-e6-s$s/metrics.json" ] || { echo "seed $s failed" > runs/p47-e6.failed; exit 1; }; done
python -m sdc.phase47.where e6 > runs/p47-e6-where.log 2>&1 || { echo where > runs/p47-e6.failed; exit 1; }
echo done > runs/p47-e6.done
