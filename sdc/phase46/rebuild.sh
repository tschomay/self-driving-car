#!/usr/bin/env bash
# Rebuild Phase 4.6's gitignored inputs and its best car (E10 arm B) in a fresh container. About 2.5-3 h on 4 CPUs.
set -eu
mkdir -p runs
[ -f runs/p46-maps-train.json ] || python -m sdc.phase46.maps > runs/p46-maps.log 2>&1
[ -f runs/p46-teacher-val-gate-truggy.pkl ] || python -m sdc.phase46.teacher_ref > runs/p46-teacher-ref.log 2>&1
for s in 0 1; do
  [ -f "runs/p46-e10-b-s$s/metrics.json" ] || python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" \
    --procs 2 --out "runs/p46-e10-b-s$s" > "runs/p46-e10-b-s$s.log" 2>&1 &
done
wait
python -m sdc.phase46.ensemble --runs runs/p46-e10-b-s0 runs/p46-e10-b-s1 --out runs/p46-e10-b-avg.json --procs 4 \
  > runs/p46-e10-b-avg.log 2>&1
echo done > runs/rebuild.done
