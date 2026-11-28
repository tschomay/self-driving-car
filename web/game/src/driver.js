// The AI drivers: a port of the self-driving-car project's loop teacher (sdc/phase46/teacher.py LoopExpert, on top of
// sdc/jumps/driver.py ExpertDriver). It reads the true state, as the teacher does.
//   steering   feed-forward for the curve just ahead plus feedback back to the centreline and along it
//   speed      the lower of the jump speed and the corner plan (lateral a_lat, braking a_brake), traction control
//   in the air the motor's reaction torque matches pitch to the landing slope (PD on pitch around the takeoff throttle)
// Presets: TRUGGY is the re-tuned teacher (teacher.TRUGGY); RACER is Phase 4.7's faster teacher (racer.TEACHER).
// The game adds a lane offset so the AI cars don't all queue on the centreline (0 near jumps).
import * as cm from './carmodel.js';
import SIM from '../../shared/generated/sim.json' with { type: 'json' };   // python -m sdc.export: the sim's numbers

const G = 9.81, WHEELBASE = 0.30;

export const TRUGGY = SIM.teacher.truggy;
export const RACER = SIM.teacher.racer;                      // racer.TEACHER in the champion's sim
export const MIDDLE = { ...TRUGGY, a_lat: 3.0, a_brake: 3.0 };

/** The command a whose expo-curved steering (1 - e) a + e a^3 equals s (env.expo_inverse). */
export function expoInverse(s, e) {
  if (!e) return s;
  let a = s;
  for (let k = 0; k < 20; k++) { const f = (1 - e) * a + e * a ** 3 - s; a -= f / ((1 - e) + 3 * e * a * a); }
  return Math.max(-1, Math.min(1, a));
}

/** Pitch of the terrain where the ballistic arc from state s meets the ground (driver.landing_slope, track coordinates). */
function landingSlope(course, s) {
  let x = s.x, z = s.z - 0.10, vz = s.vz;
  const vx = s.vx;
  for (let k = 0; k < 150; k++) {
    x += vx * 0.01; vz -= G * 0.01; z += vz * 0.01;
    if (z <= course.height(x) && vz < 0) break;
  }
  return Math.atan(course.slope(x));
}

export class LoopExpert {
  constructor(course, opts = {}) {
    const o = { a_lat: 3.5, a_brake: 3.0, v_target: 7.4, v_cruise: null, look: 0.15, k_ff: 1.0, jump_v: null, kv: 0.8,
      kv_ff: 0.06, tc: null, k_rate: 0.15, kp: 4.0, kd: 1.0, lead: 0.0, offset: 0.0, ...opts };
    Object.assign(this, o);
    if (this.v_cruise === null) this.v_cruise = this.v_target;
    this.c = course; this.u0 = null;
    // corner_speed: per centreline point, the speed limit from the corners at and ahead of it
    const n = course.n, v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.min(Math.sqrt(this.a_lat / Math.max(Math.abs(course.kappa[i]), 1e-6)), 20.0);
    for (let pass = 0; pass < 2; pass++)
      for (let i = n - 1; i >= 0; i--) { const nx = v[(i + 1) % n]; v[i] = Math.min(v[i], Math.sqrt(nx * nx + 2 * this.a_brake * course.ds)); }
    this.plan = v;
  }

  jumpSpeed(s) {
    const c = this.c, x = ((s.x % c.length) + c.length) % c.length;
    if (this.jump_v) {
      const j = c.jumps.find((jj) => jj.end >= x) ?? c.jumps[0];
      return this.jump_v[j.kind] ?? this.v_target;
    }
    return this.v_target;
  }

  act(car) {
    const s = car.st, c = this.c;
    if (car.airborne) {
      if (this.u0 === null) this.u0 = car.lastAct[1];
      const target = landingSlope(c, s);
      return [0, Math.max(-1, Math.min(1, this.u0 + this.kp * (target + this.lead - s.pitch) - this.kd * s.pitchRate))];
    }
    this.u0 = null;
    const v = Math.max(s.vFwd, 0.5), kap = c.curvature(s.x + this.look * v);
    const ff = this.k_ff * Math.atan(WHEELBASE * kap) / cm.STEER_MAX;
    const x = ((s.x % c.length) + c.length) % c.length;
    const nearJump = c.jumps.some((j) => j.start - 6.0 <= x && x <= j.end);
    const off = nearJump ? 0 : this.offset;
    let steer = Math.max(-1, Math.min(1, ff - 1.0 * (s.y - off) - 1.5 * s.yaw - this.k_rate * s.yawRate));
    const i = Math.floor(x / c.ds) % c.n;
    const target = Math.min(nearJump ? this.jumpSpeed(s) : this.v_cruise, this.plan[i]);
    let throttle = this.kv * (target - s.vFwd) + this.kv_ff * target;
    if (this.tc !== null && throttle > 0) {
      const spin = car.meanWheelSpeed() - s.vFwd;
      if (spin > this.tc) throttle *= Math.max(0, Math.min(1, 1 - (spin - this.tc) / this.tc));
    }
    if (car.expo) steer = expoInverse(steer, car.expo);
    return [steer, Math.max(-1, Math.min(1, throttle))];
  }
}
