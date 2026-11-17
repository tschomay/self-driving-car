#!/usr/bin/env bash
# Phase 4.6 E3: copy the loop teacher into the owner's GRU (filter inputs, parts-list sensors + two angled range
# sensors), with and without the out-lap map of the turns. Practice loops at training-range physics; scored on the
# validation loops at gate range. Seeds 0-1, four runs at once (about 3 GB each).
# First: python -m sdc.phase46.maps   (out-lap maps; also writes nothing else)
set -u
for s in 0 1; do
  python -m sdc.phase46.bc_loop --out "runs/p46-e3-map-s$s" --seed "$s" --map > "runs/p46-e3-map-s$s.log" 2>&1 &
  python -m sdc.phase46.bc_loop --out "runs/p46-e3-nomap-s$s" --seed "$s" > "runs/p46-e3-nomap-s$s.log" 2>&1 &
done
wait
