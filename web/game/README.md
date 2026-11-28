# rc-racer — RC Dirt Circuit

An RC truggy racing game for the phone that runs the **same physics as the
[self-driving-car](https://github.com/tschomay/self-driving-car) project's training sim**: the same engine
(MuJoCo 4.0.1, here as WebAssembly) stepping the same car on its "truggy2" physics (the full-size 1/8 truggy), on the project's own Phase 4.6
loops. Rendering is the project's race replay viewer. Time trial with a ghost of your best lap, or a race against
three AI cars ported from the project's teacher.

**Status:** prototyping

## Run

```sh
cd .. && npm install && cd game   # one install in web/ for the game and the viewer
node build.mjs --serve     # http://localhost:8123, rebuilds on change
node build.mjs   # dist/: index.html, game.js, mujoco.wasm, loops/ (any static host), artifact.html (Claude artifact)
# refresh the loops from the training repo:
python tools/export_loops.py --out loops
```

Controls: a Bluetooth or USB gamepad (left stick steers; right trigger drives, left trigger brakes and reverses;
or the right stick up/down like a stick radio), touch (left thumb steers, right thumb throttle and brake), or
the keyboard (arrows / WASD, R reset, C camera, Esc pause).

## Checks (headless, in Node, on the real physics)

```sh
node test/loops.mjs      # the JS heightfield rebuild matches the sim's, for every loop
node test/race.mjs       # full four-car races
node test/feel.mjs [grip] [py.json]   # launch, braking, cornering, lift-off; side by side with tools/py_feel.py
node test/parity.mjs     # one scripted run against a recorded sim trajectory (diverges by chaos after ~1 s)
```

## Layout

| File | What |
|---|---|
| `src/carmodel.js`, `src/sim.js` | The truggy as MuJoCo XML and its actuators, ported from `sdc/jumps/car.py`, `env.py`, `real.py` |
| `src/loop.js`, `loops/`, `tools/export_loops.py` | Phase 4.6 loops (`sdc/jumps/loop.py`) and their exact heightfield |
| `src/driver.js` | AI: `sdc/phase46/teacher.py` LoopExpert with the TRUGGY and Phase 4.7 racer settings |
| `src/race.js` | Laps, timing, crashes and the marshal (Phase 4.7 race rules), opponents, ghost |
| `../shared/` | The renderer, shared with the race replay viewer (`web/shared/`); cars drawn at the physics car's geometry |
| `src/input.js`, `src/audio.js`, `src/main.js`, `src/page.html` | Controls, sound, HUD and menu |

Docs: [`docs/physics.md`](docs/physics.md) — what is the same as the training sim, what differs, and measured
performance of the car.
