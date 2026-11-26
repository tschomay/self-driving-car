#!/usr/bin/env bash
# Full-size truggy (SDC_REAL=truggy2) rebuild, after runs/p48-racer ends: pick its best checkpoint by race time,
# copy it onto sensors as in E5 (2 seeds), run E7 on seed 0, score each with where.py p48 (5 noisy passes).
set -u
cd "$(dirname "$0")/../.."
export SDC_REAL=truggy2 PYTHONPATH=$PWD
best() {   # best.pt in run dir $1, chosen by 20 x lap + 20 x crash rate x 7 s from the 2-pass evals in log $2
  python - "$1" "$2" <<'PY'
import ast, shutil, sys
run, log = sys.argv[1:]
ev = [ast.literal_eval(l[5:]) for l in open(log) if l.startswith("eval ")]
race = lambda e: 20 * e["lap_s"] + 20 * (1 - e["mean"] / 150) * 7
b = min(ev, key=race)
print("best", b, round(race(b), 1))
shutil.copy(f"{run}/ckpt/{b['steps']}.pt", f"{run}/ckpt/best.pt")
PY
}
best runs/p48-racer runs/p48-racer.log > runs/p48-best.log || { echo best > runs/p48.failed; exit 1; }
for s in 0 1; do python -m sdc.phase46.bc_loop --truggy --rounds 16 --expo 0.4 --seed "$s" --procs 2 \
  --teacher-racer runs/p48-racer/ckpt/best.pt --out "runs/p48-e5-s$s" > "runs/p48-e5-s$s.log" 2>&1 & done
wait
for s in 0 1; do [ -f "runs/p48-e5-s$s/metrics.json" ] || { echo "e5 seed $s" > runs/p48.failed; exit 1; }; done
python -m sdc.phase47.where p48 p48racer p48e5 > runs/p48-where-e5.log 2>&1 || { echo where-e5 > runs/p48.failed; exit 1; }
python -m sdc.phase47.e7 --out runs/p48-e7 --seed 0 --hours 9 --set actor=runs/p48-e5-s0 \
  --set critic=runs/p48-racer/ckpt/best.pt > runs/p48-e7.log 2>&1 || { echo e7 > runs/p48.failed; exit 1; }
best runs/p48-e7 runs/p48-e7.log >> runs/p48-best.log
cp runs/p48-e7/ckpt/best.pt runs/p48-e7/model.pt && cp runs/p48-e5-s0/metrics.json runs/p48-e7/metrics.json
python -m sdc.phase47.where p48 p48e7 > runs/p48-where-e7.log 2>&1 || { echo where-e7 > runs/p48.failed; exit 1; }
echo done > runs/p48.done
