#!/bin/bash
# E4 launcher: behaviour clone, then PPO fine-tune, per seed. Safe to re-run after a restart.
cd /home/user/self-driving-car
OPTS="--env fan=true --env servo=true --env steer_cost=0.03"
seed() {
  s=$1; bc=runs/p4-e4-bc-s$s; ft=runs/p4-e4-ft-s$s
  [ -f $bc/metrics.json ] || python -m sdc.phase4.bc --out $bc --seed $s > $bc.log 2>&1
  [ -f $ft/metrics.json ] || python -m sdc.phase4.train --seed $s --out $ft $OPTS \
      --set init=\"$bc/model.pt\" --set std_cap=[0.15,0.05] --set gamma=0.997 --label "clone + my PPO" >> $ft.log 2>&1
}
seed 0 & seed 1 & seed 2 &
wait
