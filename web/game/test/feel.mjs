// Behaviour tests of the truggy car on flat ground: launch, braking, cornering, lift-off at full lock. Same protocol
// as tools/py_feel.py on the training sim; with a file from it, prints both side by side.
//   node test/feel.mjs [grip] [py_feel.json]
import fs from 'fs';
import loadMujoco from '@mujoco/mujoco';
import { LoopCourse, ZMAX } from '../src/loop.js';
import { World } from '../src/sim.js';
import * as cm from '../src/carmodel.js';
const grip = +(process.argv[2] || 1), pyFile = process.argv[3];
const mj = await loadMujoco();
const course = new LoopCourse(JSON.parse(fs.readFileSync('loops/10001.json')));
const FLAT = 0.5;
function make(nudge) {
  const w = new World(mj, course, 1), car = w.cars[0];
  w.m.hfield_data.fill(FLAT / ZMAX);
  w.setPhys([{}], grip);
  car.expo = 0;
  const q = w.d.qpos;
  q.set(w.m.qpos0); w.d.qvel.fill(0);
  q[0] = nudge; q[1] = 0; q[2] = FLAT + 0.17; q[3] = 1; q[4] = q[5] = q[6] = 0;
  mj.mj_forward(w.m, w.d);
  for (let k = 0; k < 100; k++) mj.mj_step(w.m, w.d);
  return w;
}
function st(w) {
  const q = w.d.qpos, v = w.d.qvel, x = q[4], y = q[5], z = q[6], qw = q[3];
  const R0 = 1 - 2 * (y * y + z * z), R3 = 2 * (x * y + qw * z), R6 = 2 * (x * z - qw * y);
  return { x: q[0], y: q[1], v: R0 * v[0] + R3 * v[1] + R6 * v[2], yaw: Math.atan2(R3, R0), yr: v[5], sp: Math.hypot(v[0], v[1]), pitch: Math.atan2(R6, Math.hypot(R0, R3)) };
}
const hold = (vt, v) => Math.max(-1, Math.min(1, 0.8 * (vt - v) + 0.06 * vt));
function run(w, fn, secs) {
  const car = w.cars[0], out = [];
  for (let k = 0; k < Math.round(secs / cm.DT); k++) {
    const s = st(w);
    for (let j = 0; j < cm.SUBSTEPS; j++) w.substep(j === 0 ? () => car.command(fn(k * cm.DT, s)) : null);
    out.push(st(w));
  }
  return out;
}
function air(nudge, u) {     // free flight: level, 7 m/s, wheels at ground speed, 2 m up; command u for 0.5 s
  const w = make(nudge), car = w.cars[0], q = w.d.qpos, v = w.d.qvel;
  q[2] = FLAT + 2.0; v.fill(0); v[0] = 7.0;
  for (const j of car.spinDof) v[j] = 7.0 / cm.WHEEL_R;
  mj.mj_forward(w.m, w.d);
  const p0 = st(w).pitch, tr = run(w, () => [0, u], 0.5), out = (tr.at(-1).pitch - p0) * 180 / Math.PI;
  w.dispose();
  return out;
}
const angle = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
function tests(nudge) {
  const r = {};
  let w = make(nudge), tr = run(w, () => [0, 1], 3.0), v = tr.map((s) => s.v);
  r.t6 = (v.findIndex((x) => x >= 6) + 1) * cm.DT; r.t10 = (v.findIndex((x) => x >= 10) + 1) * cm.DT; r.v3 = v.at(-1); w.dispose();
  w = make(nudge); run(w, (t, s) => [0, hold(8, s.v)], 3.0);
  const s0 = st(w); tr = run(w, () => [0, -1], 2.5);
  let stop = tr.findIndex((s) => s.sp < 0.1); if (stop < 0) stop = tr.length - 1;
  r.brake_v0 = s0.v; r.brake_m = Math.hypot(tr[stop].x - s0.x, tr[stop].y - s0.y);
  r.brake_yaw = Math.max(...tr.slice(0, stop + 1).map((s) => Math.abs(angle(s.yaw - s0.yaw)))) * 180 / Math.PI; w.dispose();
  for (const [name, steer, vt] of [['c5', 0.5, 5], ['c8', 0.5, 8]]) {
    w = make(nudge); tr = run(w, (t, s) => [steer, hold(vt, s.v)], 6.0).slice(-100);
    const yr = tr.reduce((a, s) => a + s.yr, 0) / tr.length, sp = tr.reduce((a, s) => a + s.sp, 0) / tr.length;
    r[name + '_R'] = sp / Math.abs(yr); r[name + '_g'] = sp * Math.abs(yr) / 9.81; w.dispose();
  }
  w = make(nudge); run(w, (t, s) => [0, hold(7, s.v)], 3.0);
  const l0 = st(w); tr = run(w, () => [1, 0], 2.0);
  let tot = 0, prev = l0.yaw; for (const s of tr) { tot += Math.abs(angle(s.yaw - prev)); prev = s.yaw; }
  r.lift_yaw = tot * 180 / Math.PI; r.lift_v = tr.at(-1).sp; w.dispose();
  r.air_thr = air(nudge, 1); r.air_brk = air(nudge, -1); r.air_neu = air(nudge, 0);
  return r;
}
const res = [0, 1, 2, 3, 4].map((k) => tests(k * 1e-6));
const js = Object.fromEntries(Object.keys(res[0]).map((k) => [k, res.map((x) => x[k])]));
const py = pyFile ? JSON.parse(fs.readFileSync(pyFile)) : null;
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const label = { t6: '0-6 m/s (s)', t10: '0-10 m/s (s)', v3: 'speed after 3 s (m/s)', brake_v0: 'brake from (m/s)', brake_m: 'stopping distance (m)',
  brake_yaw: 'heading change braking (deg)', c5_R: 'half lock at 5 m/s: radius (m)', c5_g: '  lateral (g)', c8_R: 'half lock at 8 m/s: radius (m)',
  c8_g: '  lateral (g)', air_thr: 'in the air, 0.5 s full throttle: pitch (deg)', air_brk: '  0.5 s full brake: pitch (deg)', air_neu: '  0.5 s neutral: pitch (deg)', lift_yaw: 'lift at full lock from 7 m/s: rotation (deg)', lift_v: '  speed after 2 s (m/s)' };
console.log(`grip ${grip}, mean of 5 runs nudged by 1 µm` + (py ? ' | training sim (Python) | JS port' : ''));
for (const k of Object.keys(label)) {
  const j = mean(js[k]);
  console.log(`${label[k].padEnd(44)} ${py ? mean(py[k].filter((x) => x !== null)).toFixed(2).padStart(7) + ' |' : ''} ${j.toFixed(2).padStart(7)}`);
}
