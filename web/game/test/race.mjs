// Headless check: races on Phase 4.6 loops in the real physics. Your car is driven by the teacher too (preset by
// env ME=truggy|middle|racer). Prints lap times and crashes per car.  node test/race.mjs [seeds...]
import fs from 'fs';
import loadMujoco from '@mujoco/mujoco';
import { LoopCourse } from '../src/loop.js';
import { Race } from '../src/race.js';
import { LoopExpert, TRUGGY, MIDDLE, RACER } from '../src/driver.js';
const mj = await loadMujoco();
const seeds = process.argv.slice(2).map(Number), grip = Number(process.env.GRIP || 1), laps = Number(process.env.LAPS || 3);
const me = { truggy: TRUGGY, middle: MIDDLE, racer: RACER }[process.env.ME || 'truggy'];
for (const seed of seeds.length ? seeds : [10000, 10001, 10002, 10003]) {
  const c = new LoopCourse(JSON.parse(fs.readFileSync(`loops/${seed}.json`)));
  const race = new Race(mj, c, { mode: 'race', laps, grip }), drv = new LoopExpert(c, me);
  const t0 = performance.now();
  let t = 0;
  while (t < 60 * laps && race.cars.some((x) => x.finished === null)) {
    race.step(0.02, drv.act(race.player.sim)); t += 0.02;
    if (race.state === 'finished') race.state = 'racing';
  }
  console.log(seed, `${c.length.toFixed(0)} m`, race.cars.map((x) => `${x.name}: ${x.finished ? x.finished.toFixed(1) + ' s' : 'DNF'}, laps ${x.laps.map((l) => l.toFixed(1)).join('/')}${x.crashes.length ? ', ' + x.crashes.join(', ') : ''}`).join(' | '),
    `· ${(t / ((performance.now() - t0) / 1000)).toFixed(1)}x real time`);
  race.dispose();
}
