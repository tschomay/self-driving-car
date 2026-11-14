#!/usr/bin/env bash
# Phase 4.5 E4(b): the owner's GRU memory (sdc/phase4/my_gru.py) with the owner's filter inputs, copying the steady expert.
# Same setup as E4 (c): parts-list sensors + filter pitch/roll, training-range physics, scored at gate range. Seeds 0-1.
set -u
EXPERT='{"plan": false, "v_target": 7.4, "kd": 1.0}'
ENV=(--env 'real="all"' --env 'sensors="parts"' --env 'dr="train"' --env 'val_dr="gate"' --env attitude=true)
for s in 0 1; do
  out="runs/p45-e4-b-s$s"
  [ -f "$out/model.pt" ] || python -m sdc.phase45.bc_rnn --out "$out" --seed "$s" "${ENV[@]}" --expert "$EXPERT" > "$out.log" 2>&1 &
done
wait
