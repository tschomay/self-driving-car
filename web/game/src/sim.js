// The physics world: MuJoCo 4.0.1 (WebAssembly) stepping the truggy2 car (full-size 1/8) at 2 ms on a Phase 4.6 loop, with the
// actuator models of sdc/jumps/env.py JumpCar._actuate and _physics: command delay queue, steering expo, speed-limited
// servo with a deadband, battery sag, the ESC (drive with back-EMF, a brake that only stops the wheels, a light drag
// brake at neutral, no reverse), and the sliding-grip drop. Track coordinates follow env.JumpCar.state(loop=True).
import * as cm from './carmodel.js';
import { START_X } from './loop.js';

// real.Phys, nominal
export const NOMINAL = {
  mass: 1.0, friction: 1.0, spring: 1.0, damping: 1.0, motor: 1.0, charge: 0.95, sag: 0.07,
  servo_s60: 0.12, deadband: 0.01, delay: 0.015,
};
// real.RANGES["train"] (friction is the surface choice, shared)
export const TRAIN_RANGE = {
  mass: [0.85, 1.15], spring: [0.8, 1.25], damping: [0.7, 1.4], motor: [0.85, 1.1],
  charge: [0.85, 1.0], servo_s60: [0.08, 0.16], delay: [0.005, 0.03],
};
export const BRAKE_CAP = 0.65;                            // env.BRAKE_CAP
export const SLIDE_S0 = 0.3, SLIDE_S1 = 1.5, SLIDE_DROP = 0.25;
export const EXPO = 0.4;                                  // steering expo of the Phase 4.6 gate car and after
const ON_GROUND = 0.01;                                   // env.ON_GROUND

export class Car {
  constructor(world, i) {
    const { mj, m } = world, p = `c${i}_`, O = mj.mjtObj;
    const id = (type, name) => {
      const k = mj.mj_name2id(m, type.value, p + name);
      if (k < 0) throw new Error(`missing ${p + name}`);
      return k;
    };
    this.world = world; this.index = i;
    const root = id(O.mjOBJ_JOINT, 'root');
    this.qadr = m.jnt_qposadr[root]; this.vadr = m.jnt_dofadr[root];
    this.body = id(O.mjOBJ_BODY, 'car');
    this.wheelBody = cm.WHEELS.map((w) => id(O.mjOBJ_BODY, `${w}_wheel`));
    this.spinDof = cm.WHEELS.map((w) => m.jnt_dofadr[id(O.mjOBJ_JOINT, `${w}_spin`)]);
    this.spinQ = cm.WHEELS.map((w) => m.jnt_qposadr[id(O.mjOBJ_JOINT, `${w}_spin`)]);
    this.suspJ = cm.WHEELS.map((w) => id(O.mjOBJ_JOINT, `${w}_susp`));
    this.suspQ = this.suspJ.map((j) => m.jnt_qposadr[j]);
    this.suspDof = this.suspJ.map((j) => m.jnt_dofadr[j]);
    this.steerQ = ['fl', 'fr'].map((w) => m.jnt_qposadr[id(O.mjOBJ_JOINT, `${w}_steer`)]);
    this.steerDof = ['fl', 'fr'].map((w) => m.jnt_dofadr[id(O.mjOBJ_JOINT, `${w}_steer`)]);
    this.tires = cm.WHEELS.map((w) => id(O.mjOBJ_GEOM, `${w}_tire`));
    this.chassis = id(O.mjOBJ_GEOM, 'chassis');
    this.ctrl = id(O.mjOBJ_ACTUATOR, 'motor');          // motor, fl_servo, fr_servo are consecutive
    this.bodies = [this.body];
    for (let b = 0; b < m.nbody; b++) if (m.body_rootid[b] === this.body && b !== this.body) this.bodies.push(b);
    this.phys = { ...NOMINAL };
    this.mu0 = [1, 1, 1, 1];
    this.expo = EXPO;
    this.st = {}; this.si = null; this.sCont = 0;
    this.wheelsDown = [true, true, true, true];
    this.clear();
  }

  clear() {
    this.queue = []; this.applied = [0, 0]; this.servoSet = 0; this.lastAct = [0, 0];
    this.airborne = false; this.airTime = 0; this.lastAir = 0;
  }

  /** A new command [steer, throttle] in [-1, 1], read at a 50 Hz control tick; it reaches the actuators after the delay. */
  command(a) {
    const w = this.world, delay = this.phys.delay;
    const jitter = delay > 0 ? (Math.random() * 2 - 1) * 0.001 : 0;
    const lag = Math.max(0, Math.round((delay + jitter) / cm.TIMESTEP));
    const act = [Math.max(-1, Math.min(1, a[0])), Math.max(-1, Math.min(1, a[1]))];
    this.queue.push([w.sub + lag, act]);
    this.lastAct = act;
  }

  meanWheelSpeed() {
    const v = this.world.d.qvel;
    let s = 0;
    for (const j of this.spinDof) s += v[j];
    return s / 4 * cm.WHEEL_R;
  }

  /** Per tire: how fast its contact patch (the wheel's bottom point) slides over the ground, m/s (env._slip_speeds). */
  slipSpeeds(out) {
    const { mj, m, d, vel } = this.world;
    for (let k = 0; k < 4; k++) {
      mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY.value, this.wheelBody[k], vel, 0);
      const r = vel.GetView();
      out[k] = Math.hypot(r[3] + r[1] * -cm.WHEEL_R, r[4] - r[0] * -cm.WHEEL_R);
    }
    return out;
  }

  actuate() {
    const w = this.world, ctrl = w.d.ctrl, ph = this.phys;
    while (this.queue.length && this.queue[0][0] <= w.sub) this.applied = this.queue.shift()[1];
    let a0 = this.applied[0];
    if (this.expo) a0 = (1 - this.expo) * a0 + this.expo * a0 * a0 * a0;
    const target = a0 * cm.STEER_MAX, err = target - this.servoSet;
    if (Math.abs(err) > ph.deadband) {
      const step = (Math.PI / 3) / ph.servo_s60 * cm.TIMESTEP;
      this.servoSet += Math.max(-step, Math.min(step, err));
    }
    const raw = this.applied[1];
    let u = raw * ph.charge * (1.0 - ph.sag * Math.abs(raw));            // battery
    const wn = this.meanWheelSpeed() / cm.TOP_SPEED;                        // ESC: motor speed / no-load speed
    if (raw > 0.05) u = u - wn;                                             // drive, back-EMF
    else if (raw < -0.05) u = -BRAKE_CAP * Math.abs(u) * Math.tanh(wn * cm.TOP_SPEED / 0.3);   // brake, no reverse
    else u = -0.1 * wn;                                                     // neutral: light drag brake
    u = Math.max(-1, Math.min(1, u));
    // sliding grip: friction falls by up to 25% as the contact patch slides from 0.3 to 1.5 m/s
    const slip = this.slipSpeeds(w.slipTmp), gf = w.m.geom_friction;
    for (let k = 0; k < 4; k++) {
      const f = Math.min(1, Math.max(0, (slip[k] - SLIDE_S0) / (SLIDE_S1 - SLIDE_S0)));
      gf[3 * this.tires[k]] = this.mu0[k] * (1 - SLIDE_DROP * f);
    }
    ctrl[this.ctrl] = u; ctrl[this.ctrl + 1] = this.servoSet; ctrl[this.ctrl + 2] = this.servoSet;
    this.u = u;
  }

  /** Put the car standing on the track at lap position s (counting laps), offset d (left +), at rest, as env.reset. */
  place(s, d = 0) {
    const { d: data, course } = this.world, q = data.qpos, v = data.qvel;
    const [X, Y, th] = course.pose(s, d), a = this.qadr;
    q[a] = X; q[a + 1] = Y; q[a + 2] = course.groundAt(X, Y) + 0.17;
    q[a + 3] = Math.cos(th / 2); q[a + 4] = 0; q[a + 5] = 0; q[a + 6] = Math.sin(th / 2);
    for (let k = 0; k < 6; k++) v[this.vadr + k] = 0;
    for (const j of this.suspQ) q[j] = 0;
    for (const j of this.suspDof) v[j] = 0;
    for (const j of this.steerQ) q[j] = 0;
    for (const j of this.steerDof) v[j] = 0;
    for (const j of this.spinDof) v[j] = 0;
    this.si = course.project(X, Y).i; this.sCont = s;
    this.clear();
  }

  /** Oracle state as env.JumpCar.state() on a loop: x = metres along the track (keeps counting past the line),
   *  y = metres left of the centreline, yaw relative to the track; plus world pose and wheel contact. */
  update() {
    const { mj, m, d: data, course } = this.world, q = data.qpos, v = data.qvel, a = this.qadr, b = this.vadr;
    const w = q[a + 3], x = q[a + 4], y = q[a + 5], z = q[a + 6];
    const R0 = 1 - 2 * (y * y + z * z), R3 = 2 * (x * y + w * z), R6 = 2 * (x * z - w * y);
    const R1 = 2 * (x * y - w * z), R4 = 1 - 2 * (x * x + z * z), R7 = 2 * (y * z + w * x), R8 = 1 - 2 * (x * x + y * y);
    const s = this.st, L = course.length;
    s.X = q[a]; s.Y = q[a + 1]; s.z = q[a + 2];
    s.qw = w; s.qx = x; s.qy = y; s.qz = z;
    s.pitch = Math.atan2(R6, Math.hypot(R0, R3));
    s.roll = Math.atan2(R7, R8);
    s.heading = Math.atan2(R3, R0);
    s.upZ = R8;
    s.wx = v[b]; s.wy = v[b + 1]; s.vz = v[b + 2];
    s.vFwd = R0 * s.wx + R3 * s.wy + R6 * s.vz;
    s.vLat = R1 * s.wx + R4 * s.wy + R7 * s.vz;
    s.speed = Math.hypot(s.wx, s.wy, s.vz);
    s.pitchRate = -v[b + 4];
    const p = course.project(s.X, s.Y, this.si);
    this.si = p.i;
    this.sCont += ((p.s - this.sCont + L / 2) % L + L) % L - L / 2;
    const kap = course.curvature(p.s), vs = Math.cos(p.th) * s.wx + Math.sin(p.th) * s.wy;
    s.x = this.sCont; s.s = p.s; s.y = p.d; s.kappa = kap; s.vx = vs;
    s.yaw = ((s.heading - p.th + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    s.yawRate = v[b + 5] - vs * kap;
    s.wheelSpeed = this.meanWheelSpeed();
    s.steer = (q[this.steerQ[0]] + q[this.steerQ[1]]) / 2;
    s.susp = this.suspQ.map((j) => q[j]);
    let sp = 0;
    for (const j of this.spinQ) sp += q[j];
    s.spin = sp / 4;
    // wheels on the ground: tire within 1 cm of the terrain (env.contacts, mj_geomDistance)
    let any = false;
    for (let k = 0; k < 4; k++) {
      this.wheelsDown[k] = mj.mj_geomDistance(m, data, this.tires[k], this.world.ground, 0.05, null) < ON_GROUND;
      any = any || this.wheelsDown[k];
    }
    if (!any) { this.airTime = this.airborne ? this.airTime + cm.DT : 0; this.airborne = true; }
    else { if (this.airborne) this.lastAir = this.airTime; this.airborne = false; }
    return s;
  }
}

export class World {
  constructor(mj, course, nCars) {
    this.mj = mj; this.course = course;
    this.m = mj.MjModel.from_xml_string(cm.modelXml(nCars));
    this.d = new mj.MjData(this.m);
    this.m.hfield_data.set(course.hfield());
    this.vel = new mj.DoubleBuffer(6);
    this.slipTmp = new Float64Array(4);
    this.base = {
      body_mass: Float64Array.from(this.m.body_mass), body_inertia: Float64Array.from(this.m.body_inertia),
      jnt_stiffness: Float64Array.from(this.m.jnt_stiffness), dof_damping: Float64Array.from(this.m.dof_damping),
      actuator_gear: Float64Array.from(this.m.actuator_gear), geom_friction: Float64Array.from(this.m.geom_friction),
    };
    this.ground = mj.mj_name2id(this.m, mj.mjtObj.mjOBJ_GEOM.value, 'ground');
    this.cars = [];
    for (let i = 0; i < nCars; i++) this.cars.push(new Car(this, i));
    this.sub = 0; this.acc = 0;
    mj.mj_forward(this.m, this.d);
  }

  /** Per-car physical parameters (env._apply_phys); grip scales tires and ground (the surface choice). */
  setPhys(physList, grip = 1.0) {
    const { m, base, mj } = this;
    this.cars.forEach((car, i) => {
      const p = car.phys = { ...NOMINAL, ...(physList[i] || {}), friction: grip };
      for (const b of car.bodies) {
        m.body_mass[b] = base.body_mass[b] * p.mass;
        for (let k = 0; k < 3; k++) m.body_inertia[3 * b + k] = base.body_inertia[3 * b + k] * p.mass;
      }
      for (const j of car.suspJ) m.jnt_stiffness[j] = base.jnt_stiffness[j] * p.spring;
      for (const j of car.suspDof) m.dof_damping[j] = base.dof_damping[j] * p.damping;
      m.actuator_gear[6 * car.ctrl] = base.actuator_gear[6 * car.ctrl] * p.motor;
      car.tires.forEach((g, k) => { m.geom_friction[3 * g] = base.geom_friction[3 * g] * grip; car.mu0[k] = m.geom_friction[3 * g]; });
    });
    m.geom_friction[3 * this.ground] = base.geom_friction[3 * this.ground] * grip;
    this.grip = grip;
    const scratch = new mj.MjData(m);
    mj.mj_setConst(m, scratch);
    scratch.delete();
  }

  /** Let every car settle on its suspension (env.reset settles 0.2 s). */
  settle(seconds = 0.3) {
    for (let k = 0; k < seconds / cm.TIMESTEP; k++) this.mj.mj_step(this.m, this.d);
    for (const c of this.cars) c.update();
  }

  /** One 2 ms physics step. On every 10th (a 50 Hz control tick) onTick() runs first so drivers can command. */
  substep(onTick) {
    if (this.sub % cm.SUBSTEPS === 0) {
      for (const c of this.cars) c.update();
      if (onTick) onTick();
    }
    for (const c of this.cars) c.actuate();
    this.mj.mj_step(this.m, this.d);
    this.sub++;
  }

  /** Advance by real time dt (seconds), in whole 2 ms steps. */
  advance(dt, onTick) {
    this.acc += Math.min(dt, 0.1);
    let n = 0;
    while (this.acc >= cm.TIMESTEP) { this.substep(onTick); this.acc -= cm.TIMESTEP; n++; }
    return n;
  }

  dispose() { this.vel.delete(); this.d.delete(); this.m.delete(); }
}

export { START_X };
