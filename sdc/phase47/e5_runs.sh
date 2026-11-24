#!/usr/bin/env bash
# Phase 4.7 E5: the sensors-only base recipe (E10 arm B: DAgger, 16 rounds, truggy, expo 0.4) copies the privileged
# RL racer (runs/p47-racer-s1, 3M steps: 132.8 finished per pass at 25.9 s), 2 seeds, then where.py scores the racer
# and the averaged students on the 150 selection runs, 5 noisy passes.
set -u
cd "$(dirname "$0")/../.."
R=runs/p47-racer-s1/ckpt/3014656.pt
for s in 0 1; do python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" --procs 2 \
  --teacher-racer "$R" --out "runs/p47-e5-s$s" > "runs/p47-e5-s$s.log" 2>&1 & done
wait
for s in 0 1; do [ -f "runs/p47-e5-s$s/metrics.json" ] || { echo "seed $s failed" > runs/p47-e5.failed; exit 1; }; done
python -m sdc.phase47.where e5 > runs/p47-e5-where.log 2>&1 || { echo where > runs/p47-e5.failed; exit 1; }
echo done > runs/p47-e5.done
