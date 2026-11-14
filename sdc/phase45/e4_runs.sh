#!/usr/bin/env bash
# Phase 4.5 E4: what does the student need? Copies of the steady expert (E3b's recipe: parts-list sensors, training-range
# physics, scored at gate range; no fine-tuning). Baseline = E3b's copies (runs/p45-steady-bc-s*).
#   a: bigger network (256-unit layers)   c: + the owner's filter (pitch, roll) as inputs   d: both
set -u
EXPERT='{"plan": false, "v_target": 7.4, "kd": 1.0}'
ENV=(--env 'real="all"' --env 'sensors="parts"' --env 'dr="train"' --env 'val_dr="gate"')
run() {   # arm, seed, extra args...
  local arm=$1 s=$2; shift 2
  local out="runs/p45-e4-$arm-s$s"
  [ -f "$out/model.pt" ] || python -m sdc.phase4.bc --out "$out" --seed "$s" "${ENV[@]}" --expert "$EXPERT" "$@" > "$out.log" 2>&1
}
run a 0 --hidden '[256, 256]' & run c 0 --env attitude=true & run d 0 --hidden '[256, 256]' --env attitude=true &
run a 1 --hidden '[256, 256]' &
wait
run c 1 --env attitude=true & run d 1 --hidden '[256, 256]' --env attitude=true &
wait
