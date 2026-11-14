#!/usr/bin/env bash
# Phase 4.5 E3: copy the 4.5 expert into the owner's network on the parts-list sensors, then fine-tune with the
# owner's PPO + finish bonus. Arm A trains at nominal physics, arm B on the training range; both are scored and
# checkpoint-selected at gate-range physics on the validation courses.
#   bash sdc/phase45/e3_runs.sh                 # all four runs (A, B x seeds 0-1), 4 at once
#   ROUNDS=1 BC_STEPS=800 FT_STEPS=4096 EVAL=4096 OUTDIR=/tmp/x bash sdc/phase45/e3_runs.sh   # smoke test
set -u
OUTDIR=${OUTDIR:-runs}
ROUNDS=${ROUNDS:-6}; BC_STEPS=${BC_STEPS:-50000}; FT_STEPS=${FT_STEPS:-1000000}; EVAL=${EVAL:-50000}
SEEDS=${SEEDS:-"0 1"}
EXPERT='{"plan": true, "plan_offset": 0.3, "kd": 1.0}'
run() {   # arm, train range, seed
  local arm=$1 range=$2 s=$3
  local bc="$OUTDIR/p45-e3-$arm-bc-s$s" ft="$OUTDIR/p45-e3-$arm-s$s"
  local env=(--env 'real="all"' --env 'sensors="parts"' --env "dr=$range" --env 'val_dr="gate"')
  if [ ! -f "$bc/model.pt" ]; then
    python -m sdc.phase4.bc --out "$bc" --seed "$s" --rounds "$ROUNDS" --steps "$BC_STEPS" "${env[@]}" \
      --expert "$EXPERT" > "$bc.log" 2>&1 || return 1
  fi
  python -m sdc.phase4.train --seed "$s" --steps "$FT_STEPS" --eval-every "$EVAL" --out "$ft" "${env[@]}" \
    --env finish_bonus=100 --env steer_cost=0.03 --env fan=false --env servo=false \
    --set init="\"$bc/model.pt\"" --set 'std_cap=[0.15,0.05]' --set gamma=0.997 --label "copy + PPO ($arm)" \
    > "$ft.log" 2>&1
}
for s in $SEEDS; do
  run A null "$s" &
  run B '"train"' "$s" &
done
wait
