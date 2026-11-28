# Race replay viewer (issue #86)

A 3D replay of a recorded 20-lap race, built as a Claude artifact. The car's race is recorded in the sim
(`sdc/race/record.py`) and played back here; nothing is re-simulated.

```sh
# record (from the repo root): one race per car and loop, into runs/race/
python -m sdc.race.record --car models/e7 --dr nominal --loops 10001 10002 10003 --out runs/race8n
python -m sdc.race.record --car teacher --sim truggy2 --dr nominal --loops 10001 10002 10003 --out runs/race8n
python -m sdc.race.record --car runs/p47-e5-s0 runs/p47-e5-s1 --name e5   # any student runs, averaged

# build (from web/viewer/; run npm install once in web/): dist/race.html + dist/data/
node build.mjs --data ../../runs/race8n --main e7  # --serve: local preview on http://localhost:8124
```

The main car is `--main`. Every other car recorded on all the loops can be picked as the rival. Car labels live in
`build.mjs` (`CARS`).

| File | What |
|---|---|
| `src/main.js`, `src/page.html` | Playback, ghost of the best lap, rival gap, slow-motion replays, HUD, end card |
| `../shared/` | The renderer, shared with the game (`web/shared/`) |
