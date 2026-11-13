#!/bin/bash
# E3 launcher: safe to re-run after a container restart (runs resume; finished runs are skipped)
cd /home/user/self-driving-car
OPTS="--env fan=true --env servo=true --env steer_cost=0.03 --set std_cap=[0.5,0.1]"
run() { # module out seed label
  [ -f $2/metrics.json ] && return
  python -m $1 --seed $3 --out $2 $OPTS --label "$4" >> $2.log 2>&1
}
(run sdc.phase4.train_rnn runs/p4-e3-gru-s0 0 "my PPO + my GRU"; run sdc.phase4.train_rnn runs/p4-e3-gru-s2 2 "my PPO + my GRU") &
(run sdc.phase4.train_rnn runs/p4-e3-gru-s1 1 "my PPO + my GRU"; run sdc.phase4.train runs/p4-e3-mlp-s2 2 "my PPO, no memory") &
(run sdc.phase4.train runs/p4-e3-mlp-s0 0 "my PPO, no memory") &
(run sdc.phase4.train runs/p4-e3-mlp-s1 1 "my PPO, no memory") &
wait
