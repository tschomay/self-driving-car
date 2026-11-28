// The self-driving-car project's car on its "truggy2" physics, as MuJoCo XML: a port of sdc/jumps/car.py
// model_xml(real="truggy2", terrain=loop.TERRAIN) with the same numbers (branch ccr-e2fca4fb-g9rode at 7b1cd69):
//   all     wide tires (capsules), soft suspension (30% sag) + battery on top, one motor through diffs
//   esc     back-EMF moves from the tendon damper into the ESC model (sim.js); no reverse
//   lsd     limited-slip diffs: damped tendons front (fl-fr), rear (rl-rr) and centre (front axle - rear axle)
//   power   0.6 N·m per wheel, 16 m/s no-load
//   heavy   wheel + drivetrain spin inertia (armature 1.5e-4 on each wheel's spin joint)
//   elliptic MuJoCo's elliptic friction cone (grip the same in every direction)
//   slide   (sim.js) tires lose up to 25% of their grip as the contact patch slides; tires get contact priority 1
//   wheels  each wheel's spin inertia about its axle set explicitly (a tyre carries its mass at the tread)
//   groundroll (race.js) "rolled over" counts only on the ground
//   size8   the full-size 1/8 truggy: 380 mm wheelbase, 143 mm tyres, 4.0 kg, 34 mm ride height (car.GEOM["size8"])
// Differences, none of which change how one car drives: several cars share one world (names get a c<i>_ prefix);
// chassis boxes also collide with each other (collision bit 2); no cameras, rays, textures or lights.
import { BOX, N, ZMAX } from './loop.js';
import SIM from '../../shared/generated/sim.json' with { type: 'json' };   // python -m sdc.export: the sim's numbers

// Every number below comes from the sim via web/shared/generated/sim.json (car.GEOM, car.motor, car.LSD, ...).
const C = SIM.car;
export const TIMESTEP = C.timestep;
export const SUBSTEPS = C.substeps;    // 50 Hz control
export const DT = TIMESTEP * SUBSTEPS;
export const WHEEL_R = C.wheel_r;
export const TAU_MAX = C.tau_max;      // N·m per wheel
export const TOP_SPEED = C.top_speed;  // m/s no-load
export const STEER_MAX = C.steer_max;  // rad
export const WHEELS = ['fl', 'fr', 'rl', 'rr'];
export const CORNER = { fl: [C.corner_x, C.corner_y], fr: [C.corner_x, -C.corner_y], rl: [-C.corner_x, C.corner_y], rr: [-C.corner_x, -C.corner_y] };
export const MOUNT_Z = C.mount_z;
export const TIRE_W = C.tire_w;
export const SOFT_K = C.spring_k, SOFT_C = C.spring_c;
export const CHASSIS_M = C.chassis_m, BATTERY_M = C.battery_m, BATTERY_Z = C.battery_z;
export const CORNER_M = C.corner_m, WHEEL_M = C.wheel_m, AXLE_I = C.axle_i;
export const LSD = C.lsd;
export const WHEEL_ARMATURE = C.wheel_armature;
export const CHASSIS_HALF = C.box;

function carXml(i) {
  const p = `c${i}_`;
  const corners = WHEELS.map((w) => {
    const [cx, cy] = CORNER[w];
    const steer = w[0] === 'f'
      ? `<joint name="${p}${w}_steer" type="hinge" axis="0 0 1" range="-${STEER_MAX} ${STEER_MAX}" damping="0.05" armature="0.0005"/>` : '';
    return `
      <body name="${p}${w}" pos="${cx} ${cy} ${MOUNT_Z}">
        <inertial pos="0 0 0" mass="${CORNER_M}" diaginertia="2e-5 2e-5 2e-5"/>
        <joint name="${p}${w}_susp" type="slide" axis="0 0 1" range="-0.03 0.05" stiffness="${SOFT_K}" springref="-0.03" damping="${SOFT_C}" limited="true" solreflimit="0.005 1" armature="0.01"/>
        ${steer}
        <body name="${p}${w}_wheel">
          <inertial pos="0 0 0" mass="${WHEEL_M}" diaginertia="${AXLE_I / 2} ${AXLE_I} ${AXLE_I / 2}"/>
          <joint name="${p}${w}_spin" type="hinge" axis="0 1 0" damping="0.000010" armature="${WHEEL_ARMATURE}"/>
          <geom name="${p}${w}_tire" type="capsule" euler="1.5708 0 0" contype="1" conaffinity="0" priority="1" size="${WHEEL_R} ${TIRE_W / 2}" mass="${WHEEL_M}" friction="1.0 0.005 0.0001"/>
        </body>
      </body>`;
  }).join('');
  const body = `
    <body name="${p}car" pos="${i * 2} 0 1">
      <freejoint name="${p}root"/>
      <geom name="${p}chassis" type="box" contype="3" conaffinity="2" size="${CHASSIS_HALF.join(' ')}" mass="${CHASSIS_M - BATTERY_M}"/>
      <geom name="${p}battery" type="box" contype="0" conaffinity="0" pos="0 0 ${BATTERY_Z}" size="0.07 0.035 0.015" mass="${BATTERY_M}"/>
      ${corners}
    </body>`;
  const actuators = `
    <motor name="${p}motor" tendon="${p}drive" gear="${4 * TAU_MAX}" ctrlrange="-1 1"/>
    <position name="${p}fl_servo" joint="${p}fl_steer" kp="3" ctrlrange="-${STEER_MAX} ${STEER_MAX}"/>
    <position name="${p}fr_servo" joint="${p}fr_steer" kp="3" ctrlrange="-${STEER_MAX} ${STEER_MAX}"/>`;
  const J = (w, c) => `<joint joint="${p}${w}_spin" coef="${c}"/>`;
  const tendon = `<fixed name="${p}drive" damping="0.000000">${WHEELS.map((w) => J(w, 0.25)).join('')}</fixed>`
    + `<fixed name="${p}dfront" damping="${LSD.front}">${J('fl', 1)}${J('fr', -1)}</fixed>`
    + `<fixed name="${p}drear" damping="${LSD.rear}">${J('rl', 1)}${J('rr', -1)}</fixed>`
    + `<fixed name="${p}dcentre" damping="${LSD.centre}">${J('fl', 0.5)}${J('fr', 0.5)}${J('rl', -0.5)}${J('rr', -0.5)}</fixed>`;
  return { body, actuators, tendon };
}

export function modelXml(nCars) {
  const cars = [];
  for (let i = 0; i < nCars; i++) cars.push(carXml(i));
  return `<mujoco model="rc-truggy-loop">
  <compiler angle="radian"/>
  <option timestep="${TIMESTEP}" integrator="implicitfast" cone="elliptic"/>
  <asset>
    <hfield name="terrain" nrow="${N}" ncol="${N}" size="${BOX / 2} ${BOX / 2} ${ZMAX} 0.3"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" contype="0" conaffinity="1" hfield="terrain" pos="0 0 0" friction="1.0 0.005 0.0001"/>
    ${cars.map((c) => c.body).join('')}
  </worldbody>
  <actuator>${cars.map((c) => c.actuators).join('')}
  </actuator>
  <tendon>${cars.map((c) => c.tendon).join('')}</tendon>
</mujoco>`;
}
