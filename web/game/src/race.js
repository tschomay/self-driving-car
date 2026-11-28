// Race logic on top of the physics world: grid, laps, timing, crashes and the marshal, AI opponents, the ghost.
// Crashes follow the Phase 4.7 race rules (sdc/race/record.py): rolled over, off the lane, stuck, or spun round
// (facing more than 120° from the track for 0.5 s). The marshal puts the car back on the centre line at the furthest
// point it reached, or 9 m before the kicker if it crashed on a jump. The game is gentler about "off the lane" than the
// sim (a second on the banks is allowed) and puts the car back after a short pause instead of a 7 s penalty.
import { World, TRAIN_RANGE } from './sim.js';
import { LoopExpert, TRUGGY, MIDDLE, RACER } from './driver.js';
import { DRIVE_W, START_X } from './loop.js';
import * as cm from './carmodel.js';
import SIM from '../../shared/generated/sim.json' with { type: 'json' };   // python -m sdc.export: the sim's numbers

export const GRIPS = [
  { name: 'Hard-pack', grip: 1.0, note: 'the sim’s nominal grip' },
  { name: 'Loose', grip: 0.75, note: 'the low end of the training range (0.7–1.1)' },
  { name: 'Dusty', grip: 0.55, note: 'the floor of the gate range (0.55–1.25)' },
];
export const RIVALS = [
  { name: 'Racer', preset: RACER, offset: 0.4, what: 'Phase 4.7’s faster teacher: corners at 3.5 m/s², doubles at 7.8 m/s' },
  { name: 'Teacher+', preset: MIDDLE, offset: -0.4, what: 'the teacher with corners at 3.0 m/s²' },
  { name: 'Teacher', preset: TRUGGY, offset: 0.2, what: 'the re-tuned teacher the sensor cars copy (teacher.TRUGGY)' },
];
const SPUN_DEG = SIM.race.spun_deg, SPUN_S = SIM.race.spun_s, RUNUP_M = SIM.race.runup_m, PAUSE_S = 1.0;

function rng32(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Race {
  constructor(mj, course, { mode = 'trial', laps = 3, grip = 1.0 } = {}) {
    this.c = course; this.mode = mode; this.totalLaps = laps;
    const n = mode === 'race' ? 1 + RIVALS.length : 1;
    this.world = new World(mj, course, n);
    const rng = rng32(course.seed * 31 + 5);
    // the AI cars each get their own physics draw from the training range; yours is nominal
    const phys = [{}];
    for (let i = 1; i < n; i++) { const p = {}; for (const k in TRAIN_RANGE) p[k] = TRAIN_RANGE[k][0] + rng() * (TRAIN_RANGE[k][1] - TRAIN_RANGE[k][0]); phys.push(p); }
    this.world.setPhys(phys, grip);
    this.cars = this.world.cars.map((sim, i) => ({
      sim, isPlayer: i === 0, name: i ? RIVALS[i - 1].name : 'You', lap: 0, lapStart: 0, laps: [], best: null, finished: null,
      far: 0, offTime: 0, flipTime: 0, stuckTime: 0, spunTime: 0, resets: 0, crashes: [], down: null,
      driver: i === 0 ? null : new LoopExpert(course, { ...RIVALS[i - 1].preset, offset: RIVALS[i - 1].offset }),
    }));
    // the grid: in a race you start at the back (rows 1.2 m apart, staggered); in a time trial START_X past the line
    const order = mode === 'race' ? [...this.cars.slice(1), this.cars[0]] : this.cars;
    order.forEach((car, k) => {
      const s0 = mode === 'race' ? START_X - 1.2 * k : START_X, d0 = mode === 'race' ? (k % 2 ? -0.5 : 0.5) : 0;
      const sl = ((s0 % course.length) + course.length) % course.length;
      car.sim.place(s0 < 0 ? sl - course.length : s0, d0);
      car.sim.sCont = s0;
      car.far = s0;
    });
    this.world.settle(0.3);
    this.time = 0; this.state = 'countdown'; this.countdown = 3.2;
    this.player = this.cars[0];
    this.events = [];                      // consumed by the UI each frame
    this.rec = []; this.ghost = null; this.ghostT = 0;
    this.input = [0, 0];
    this.tick = () => this._tick();
  }

  /** Advance by real time dt with the player's command [steer, throttle]. */
  step(dt, input) {
    this.input = input;
    if (this.state === 'countdown') {
      const before = Math.ceil(this.countdown);
      this.countdown -= dt;
      const after = Math.ceil(this.countdown);
      if (after !== before && after > 0 && after <= 3) this.events.push({ type: 'count', n: after });
      if (this.countdown <= 0) { this.state = 'racing'; this.events.push({ type: 'go' }); }
    }
    this.world.advance(dt, this.tick);
  }

  _tick() {
    const racing = this.state !== 'countdown', L = this.c.length;
    if (racing) this.time += cm.DT;
    for (const car of this.cars) {
      const sim = car.sim, st = sim.st;
      if (car.wasAir && !sim.airborne && sim.lastAir > 0.18) { sim.justLanded = true; if (car.isPlayer) this.events.push({ type: 'land', air: sim.lastAir, vz: car.vzPrev }); }
      car.wasAir = sim.airborne; car.vzPrev = st.vz;
      // a car sitting out a crash: hold it, then the marshal puts it back
      if (car.down) {
        sim.command([0, 0]);
        if ((car.down.t -= cm.DT) <= 0) this._marshal(car);
        continue;
      }
      car.far = Math.max(car.far, st.x);
      // laps: each time the track position passes another whole lap
      if (racing && !car.finished) {
        const lapsDone = Math.floor(st.x / L);
        if (lapsDone > car.lap) {
          const t = this.time - car.lapStart;
          car.lap = lapsDone; car.lapStart = this.time; car.laps.push(t);
          const isBest = car.best === null || t < car.best;
          if (isBest) car.best = t;
          if (car.isPlayer) {
            this.events.push({ type: 'lap', time: t, best: isBest, lap: lapsDone });
            if (isBest && this.rec.length) this.ghost = this.rec;
            this.rec = []; this.ghostT = 0;
          }
          if (this.mode === 'race' && lapsDone >= this.totalLaps) {
            car.finished = this.time;
            if (car.isPlayer) { this.state = 'finished'; this.events.push({ type: 'finish', place: this.placeOf(car) }); }
          }
        }
      }
      // crashes (the race rules), with a second's grace on the banks
      car.offTime = Math.abs(st.y) > DRIVE_W / 2 ? car.offTime + cm.DT : 0;
      car.flipTime = st.upZ < 0.2 && !sim.airborne ? car.flipTime + cm.DT : 0;
      const pushing = car.isPlayer ? Math.abs(this.input[1]) > 0.3 : true;
      car.stuckTime = racing && pushing && st.vFwd < 0.3 && !sim.airborne ? car.stuckTime + cm.DT : 0;
      car.spunTime = Math.abs(st.yaw) > SPUN_DEG * Math.PI / 180 && racing ? car.spunTime + cm.DT : 0;
      const reason = Math.abs(st.y) > DRIVE_W / 2 + 1.0 || car.offTime > 1.0 ? 'Off the lane'
        : car.flipTime > 0.4 ? 'Rolled over' : car.stuckTime > 2.0 ? 'Stuck' : car.spunTime > SPUN_S ? 'Spun round' : null;
      if (reason && racing && !car.finished) { this.crash(car, reason); continue; }
      // commands
      if (!racing) sim.command([0, 0]);
      else if (car.finished) sim.command(car.driver ? [car.driver.act(sim)[0], st.vFwd > 0.5 ? -0.5 : 0] : [0, st.vFwd > 0.5 ? -0.5 : 0]);
      else if (car.isPlayer) sim.command(this.input);
      else sim.command(car.driver.act(sim));
    }
    // ghost recording: the player's pose at every control tick of the lap
    if (racing && this.mode === 'trial') { this.rec.push(snapshot(this.player.sim)); this.ghostT++; }
  }

  /** A crash: the car sits for PAUSE_S (no throttle), then the marshal puts it back. */
  crash(car, reason) {
    car.down = { t: PAUSE_S, reason };
    car.crashes.push(reason);
    car.offTime = car.flipTime = car.stuckTime = car.spunTime = 0;
    if (car.isPlayer) this.events.push({ type: 'crash', reason });
  }

  /** Back on the centreline at the furthest point reached; on a jump, RUNUP_M before its kicker (record.py). */
  _marshal(car) {
    const L = this.c.length, total = car.far, k = this.c.jumpAt(total);
    let back = total;
    if (k !== null) { const lap0 = total - (((total % L) + L) % L); back = lap0 + this.c.jumps[k].start - RUNUP_M; }
    back = Math.max(back, total - 25.0);
    car.sim.place(back, 0);
    car.far = back;
    car.down = null; car.resets++;
    if (car.driver) car.driver.u0 = null;
    if (car.isPlayer) { this.events.push({ type: 'reset' }); this.rec = []; this.ghostT = 1e9; }
  }

  /** The player asked to be put back (reset button). */
  rescue() { if (this.state === 'racing' && !this.player.down) this.crash(this.player, 'Reset'); }

  placeOf(car) {
    const key = (x) => (x.finished !== null ? 1e9 - x.finished : x.sim.st.x);
    return 1 + this.cars.filter((o) => o !== car && key(o) > key(car)).length;
  }

  ghostFrame() { return this.ghost && this.ghostT < this.ghost.length ? this.ghost[this.ghostT] : null; }
  lapTime() { return this.time - this.player.lapStart; }
  dispose() { this.world.dispose(); }
}

/** The renderer's car sample (the fields sdc/race/record.py records), from a sim car. */
export function snapshot(sim, out = {}) {
  const s = sim.st;
  out.x = s.X; out.y = s.Y; out.z = s.z; out.qw = s.qw; out.qx = s.qx; out.qy = s.qy; out.qz = s.qz;
  [out.s1, out.s2, out.s3, out.s4] = s.susp;
  out.st = s.steer; out.v = s.vFwd; out.vx = s.wx; out.vy = s.wy; out.yaw = s.heading; out.s = s.s;
  out.cs = sim.lastAct[0]; out.ct = sim.lastAct[1]; out.air = sim.airborne ? 1 : 0;
  out.gnd = sim.wheelsDown.reduce((m, w, k) => m | (w ? 1 << k : 0), 0);
  return out;
}
