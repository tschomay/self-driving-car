// Parity with the training sim: the same scripted commands on the same loop, compared with a trajectory recorded
// from sdc.jumps.env (the champion sim, nominal physics, no command delay) by tools/py_parity.py. Usage:
//   python web/game/tools/py_parity.py > /tmp/py.json && node test/parity.mjs /tmp/py.json [seed]   (also tests/test_export.py)
import fs from 'fs';
import loadMujoco from '@mujoco/mujoco';
import { LoopCourse, START_X } from '../src/loop.js';
import { World } from '../src/sim.js';
const [file, seedArg] = process.argv.slice(2);
const seed = +(seedArg || 10001), py = JSON.parse(fs.readFileSync(file));
const mj = await loadMujoco();
const course = new LoopCourse(JSON.parse(fs.readFileSync(`loops/${seed}.json`)));
const w = new World(mj, course, 1), car = w.cars[0];
w.setPhys([{ delay: 0 }], 1.0);
car.expo = 0;
car.place(START_X);
const [X, Y] = course.pose(START_X);
w.d.qpos[car.qadr + 2] = course.height(START_X) + 0.17;      // env.reset: the centreline height + 0.17
mj.mj_forward(w.m, w.d);
for (let k = 0; k < 100; k++) mj.mj_step(w.m, w.d);           // settle 0.2 s
const cmd = (t) => (t < 1.5 ? [0, 1] : t < 2.0 ? [0, -1] : t < 3.5 ? [0.6, 0.7] : t < 4.0 ? [0.6, 0] : [-0.3, 1]);
let worst = 0;
for (let k = 0; k < py.length; k++) {
  for (let j = 0; j < 10; j++) w.substep(j === 0 ? () => car.command(cmd(k * 0.02)) : null);
  const s = car.update(), p = py[k];
  const e = Math.hypot(s.X - p[0], s.Y - p[1], s.z - p[2]);
  worst = Math.max(worst, e);
  if (k % 25 === 24) console.log(`t ${((k + 1) * 0.02).toFixed(1)} s  js v ${s.vFwd.toFixed(3)} wheels ${s.wheelSpeed.toFixed(2)} | py v ${p[3].toFixed(3)} wheels ${p[4].toFixed(2)} | position gap ${(e * 1000).toFixed(2)} mm`);
}
console.log(`largest position gap ${(worst * 1000).toFixed(2)} mm over ${(py.length * 0.02).toFixed(1)} s`);
