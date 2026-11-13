#!/bin/bash
cd /home/user/self-driving-car
rs() { [ -f $1/rescore.json ] || python -m sdc.phase4.rescore $1 $2 > $1.rescore.log 2>&1; }
wait_for() { until [ -f $1/metrics.json ]; do sleep 30; done; }
(rs runs/p4-e3-mlp-s0; rs runs/p4-e3-gru-s0 --rnn; wait_for runs/p4-e3-mlp-s2; rs runs/p4-e3-mlp-s2) &
(rs runs/p4-e3-mlp-s1; rs runs/p4-e3-gru-s1 --rnn; wait_for runs/p4-e3-gru-s2; rs runs/p4-e3-gru-s2 --rnn) &
wait
