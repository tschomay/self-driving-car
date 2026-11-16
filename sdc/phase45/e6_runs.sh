#!/usr/bin/env bash
# Phase 4.5 E6: E5's copy (owner's GRU + filter, rollers) plus the walk-through out-lap map as two more inputs (distance to
# the next mapped lip, its mapped speed). Teacher: the steady expert aiming at the map's speed + 0.6 m/s for the next jump.
# Training on practice courses 100000-100299 (their crawl maps, cross-fitted), training-range physics; scored at gate
# range on validation with the validation crawl maps. Seeds 0-1. Maps: python -m sdc.phase45.maps
set -u
EXPERT='{"plan": false, "v_target": 7.4, "kd": 1.0, "use_map": true, "plan_offset": 0.6}'
COURSES=$(python -c "print(list(range(100000, 100300)))")
ENV=(--env 'real="all"' --env 'sensors="parts"' --env 'dr="train"' --env 'val_dr="gate"' --env attitude=true
     --env 'kinds=["double", "table", "drop"]' --env 'track_map="runs/p45-maps-train.json"'
     --env 'val_map="runs/p45-maps-val.json"' --env "courses=$COURSES")
for s in 0 1; do
  out="runs/p45-e6-s$s"
  [ -f "$out/model.pt" ] || python -m sdc.phase45.bc_rnn --out "$out" --seed "$s" "${ENV[@]}" --expert "$EXPERT" > "$out.log" 2>&1 &
done
wait
