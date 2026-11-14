#!/usr/bin/env bash
# Phase 4.5 E3b: E3's recipe (arm B: training-range physics, parts-list sensors, scored at gate range) but copying the
# steady-speed expert (7.4 m/s into every jump), whose speed choice the student can copy. Seeds 0-1.
set -u
EXPERT='{"plan": false, "v_target": 7.4, "kd": 1.0}'
run() {
  local s=$1 bc="runs/p45-steady-bc-s$1" ft="runs/p45-e3b-s$1"
  local env=(--env 'real="all"' --env 'sensors="parts"' --env 'dr="train"' --env 'val_dr="gate"')
  while pgrep -f "out $bc " >/dev/null; do sleep 30; done   # seed 0's copy may already be training
  if [ ! -f "$bc/model.pt" ]; then
    python -m sdc.phase4.bc --out "$bc" --seed "$s" "${env[@]}" --expert "$EXPERT" > "$bc.log" 2>&1 || return 1
  fi
  python -m sdc.phase4.train --seed "$s" --out "$ft" "${env[@]}" --env finish_bonus=100 --env steer_cost=0.03 \
    --set init="\"$bc/model.pt\"" --set 'std_cap=[0.15,0.05]' --set gamma=0.997 --label "steady copy + PPO" > "$ft.log" 2>&1
}
run 0 & run 1 &
wait
