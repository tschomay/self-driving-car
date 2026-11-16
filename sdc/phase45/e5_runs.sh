#!/usr/bin/env bash
# Phase 4.5 E5: the best student so far (owner's GRU + owner's filter inputs, copying the steady expert) on the ROLLER
# courses (doubles instead of gaps), then fine-tuned with the owner's PPO + finish bonus. The first 100k fine-tune steps
# train only the critic (the copy has none). Training-range physics; scored and kept at gate range. Seeds 0-1.
set -u
EXPERT='{"plan": false, "v_target": 7.4, "kd": 1.0}'
ENV=(--env 'real="all"' --env 'sensors="parts"' --env 'dr="train"' --env 'val_dr="gate"' --env attitude=true
     --env 'kinds=["double", "table", "drop"]')
run() {
  local s=$1 bc="runs/p45-e5-bc-s$1" ft="runs/p45-e5-s$1"
  [ -f "$bc/model.pt" ] || python -m sdc.phase45.bc_rnn --out "$bc" --seed "$s" "${ENV[@]}" --expert "$EXPERT" > "$bc.log" 2>&1 || return 1
  python -m sdc.phase4.train_rnn --seed "$s" --out "$ft" "${ENV[@]}" --env finish_bonus=100 --env steer_cost=0.03 \
    --set init="\"$bc/model.pt\"" --set 'std_cap=[0.15,0.05]' --set gamma=0.997 --set critic_warmup=100000 \
    --label "GRU copy + PPO (rollers)" > "$ft.log" 2>&1
}
run 0 & run 1 &
wait
