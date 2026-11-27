# Race replay viewer (issue #86)

A 3D replay of a recorded 20-lap race, built as a Claude artifact. The car's race is recorded in the sim
(`sdc/race/record.py`) and played back here; nothing is re-simulated.

```sh
# record (from the repo root): one race per car and loop, into runs/race/
python -m sdc.race.record --car e4 --loops 10001 10002 10003
python -m sdc.race.record --car teacher --loops 10001 10002 10003
python -m sdc.race.record --car runs/p47-e5-s0 runs/p47-e5-s1 --name e5   # any student runs, averaged

# build (from viewer/): dist/race.html + dist/data/
npm install
node build.mjs --data ../runs/race --main e7      # --serve: local preview on http://localhost:8124
```

The main car is `--main`. Every other car recorded on all the loops can be picked as the rival. Car labels live in
`build.mjs` (`CARS`).

| File | What |
|---|---|
| `src/main.js` | Playback, ghost of the best lap, rival gap, slow-motion replays, HUD, end card |
| `src/track.js` | The recorded loop as the renderer's circuit (sim terrain heights, track coordinates per cell) |
| `src/render.js`, `src/terrainart.js`, `src/carmesh.js` | three.js view, from the owner's rc-racer game (`tschomay/sandbox`, `projects/rc-racer`), with TV cameras and a truggy body |
| `src/page.html` | HUD and controls (rc-racer's pit-board look) |
