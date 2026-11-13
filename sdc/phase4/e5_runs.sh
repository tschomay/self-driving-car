#!/bin/bash
# E5 launcher: E4's fine-tune plus a finish bonus of 100. Safe to re-run after a restart.
cd /home/user/self-driving-car
OPTS="--env fan=true --env servo=true --env steer_cost=0.03 --env finish_bonus=100"
seed() {
  s=$1; bc=runs/p4-e4-bc-s$s; ft=runs/p4-e5-fb-s$s
  [ -f $ft/metrics.json ] || python -m sdc.phase4.train --seed $s --out $ft $OPTS \
      --set init=\"$bc/model.pt\" --set std_cap=[0.15,0.05] --set gamma=0.997 --label "clone + my PPO + finish bonus" >> $ft.log 2>&1
  [ -f $ft/rescore.json ] || python -m sdc.phase4.rescore $ft > $ft.rescore.log 2>&1
}
seed 0 & seed 1 & seed 2 &
wait
