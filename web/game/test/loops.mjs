// The JS heightfield rebuild against the training sim's own (checksums written by tools/export_loops.py).
import fs from 'fs';
import { LoopCourse, N, ZMAX, BASE, BANK_H } from '../src/loop.js';
const seeds = process.argv.slice(2).map(Number);
const index = JSON.parse(fs.readFileSync('loops/index.json'));
let worst = 0, bad = 0;
for (const { seed } of seeds.length ? seeds.map((seed) => ({ seed })) : index) {
  const c = new LoopCourse(JSON.parse(fs.readFileSync(`loops/${seed}.json`)));
  const t0 = performance.now(), hf = c.hfield(), ms = performance.now() - t0;
  let sum = 0, lane = 0;
  for (const v of hf) { sum += v; if (v * ZMAX < BASE + BANK_H - 1e-4) lane++; }
  let probe = 0;
  for (const [i, j, v] of c.check.probe) probe = Math.max(probe, Math.abs(hf[i * N + j] - v) * ZMAX);
  const dSum = Math.abs(sum - c.check.sum) * ZMAX, dLane = lane - c.check.n_lane;
  worst = Math.max(worst, probe);
  // a few bank-slope cells can differ by a centimetre or two (which splat sample lands last after the export rounds to 1 µm)
  const ok = probe < 1e-4 && Math.abs(dLane) <= 20 && dSum < 0.2;
  if (!ok) bad++;
  console.log(seed, ok ? 'ok ' : 'BAD', `max probe diff ${(probe * 1000).toFixed(4)} mm, sum diff ${dSum.toFixed(5)} m, lane cells ${dLane >= 0 ? '+' : ''}${dLane}, ${ms.toFixed(0)} ms`);
}
console.log(bad ? `${bad} loops differ` : `all match (worst probe ${(worst * 1000).toFixed(4)} mm)`);
process.exit(bad ? 1 : 0);
