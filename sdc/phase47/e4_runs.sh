#!/usr/bin/env bash
# Phase 4.7 E4: re-tune the teacher for landings (e4_tune.py, on train-split loops), score it on the selection runs,
# then the sensors-only base recipe (E10 arm B: DAgger, 16 rounds, truggy, expo 0.4) copies it. Waits for the code
# review (runs/p47-overnight-review.ok). If the current teacher is kept, there is nothing new to copy: stops there.
set -u
cd "$(dirname "$0")/../.."
while [ ! -f runs/p47-overnight-review.ok ]; do sleep 60; done
python -m sdc.phase47.e4_tune --procs 2 > runs/p47-e4-tune.log 2>&1 || { echo tune > runs/p47-e4.failed; exit 1; }
if grep -q '"_name": "current"' runs/p47-e4-teacher.json; then echo "current teacher kept" > runs/p47-e4.done; exit 0; fi
python -m sdc.phase47.where e4teacher > runs/p47-e4teacher-where.log 2>&1 || { echo where-teacher > runs/p47-e4.failed; exit 1; }
# second check, on the selection runs (independent of the tuning loops): the re-tuned teacher must beat the current one
python -c "import json,sys; d=json.load(open('runs/p47-e4teacher-where.json')); m=lambda w: sum(d[w]['finished_per_pass'])/len(d[w]['finished_per_pass']); sys.exit(0 if m('teacher_plus') > m('teacher') else 1)" \
  || { echo "re-tuned teacher not better on the selection runs" > runs/p47-e4.done; exit 0; }
for s in 0 1; do python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" --procs 1 \
  --teacher-json runs/p47-e4-teacher.json --out "runs/p47-e4-s$s" > "runs/p47-e4-s$s.log" 2>&1 & done
wait
for s in 0 1; do [ -f "runs/p47-e4-s$s/metrics.json" ] || { echo "seed $s failed" > runs/p47-e4.failed; exit 1; }; done
python -m sdc.phase47.where e4 > runs/p47-e4-where.log 2>&1 || { echo where > runs/p47-e4.failed; exit 1; }
echo done > runs/p47-e4.done
