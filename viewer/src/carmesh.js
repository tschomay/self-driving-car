// The truggy: a 1/8-style racing truggy (low cab-forward shell, big pin-tread tires on dish wheels, long A-arms,
// coil-over shocks, a tall wing), built around the sim car's dimensions (sdc/jumps/car.py: chassis 46 x 18 x 7 cm,
// 5.5 cm wheels at x ±15 cm, y ±14 cm, mounted at z -5 cm on 8 cm of slide travel). Body frame: x forward, y left, z up.
//
// The wheels are children of the body: setSuspension() puts each one at its recorded suspension travel, steers the
// front knuckles, spins the tires, and moves the arms, driveshafts and shocks to match.
import * as THREE from 'three';
import { canvasTexture } from './terrainart.js';
import { WHEEL_R, CORNER, MOUNT_Z, WHEELS } from './carmodel.js';

const TIRE_W = 0.052;           // looks: a truggy's wide tire (the sim's capsule tread is 3 cm plus round shoulders)
const RIM_R = 0.034;
const HUB_Y = 0.112;            // |y| of the hub carrier's inner face (wheel centre at 0.14)

let shared = null;
function sharedAssets() {
  if (shared) return shared;
  // mini-pin tread: a grid of round pins, as a bump map (bright = raised)
  const pins = canvasTexture(256, 64, (g, w, h) => {
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    const grd = (x, y) => { const r = g.createRadialGradient(x, y, 0, x, y, 3.4); r.addColorStop(0, '#fff'); r.addColorStop(0.7, '#bbb'); r.addColorStop(1, 'rgba(0,0,0,0)'); return r; };
    for (let i = 0; i < 32; i++) for (let j = 0; j < 8; j++) { const x = i * 8 + (j % 2) * 4 + 4, y = j * 8 + 4; g.fillStyle = grd(x, y); g.fillRect(x - 4, y - 4, 8, 8); }
  }, false);
  pins.wrapS = pins.wrapT = THREE.RepeatWrapping;
  pins.repeat.set(5, 1);
  // tire: a lathe profile (radius, y) with rounded shoulders and a little sidewall bulge
  const prof = [[0.031, -0.024], [0.04, -0.0262], [0.0485, -0.0262], [0.0535, -0.023], [0.055, -0.016], [0.055, 0.016], [0.0535, 0.023], [0.0485, 0.0262], [0.04, 0.0262], [0.031, 0.024]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const tireGeo = new THREE.LatheGeometry(prof, 36);
  // the lathe's u runs around the wheel and v across the profile: tread pins wrap around the tire
  const uv = tireGeo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i), uv.getY(i));
  shared = {
    tireGeo,
    tireMat: new THREE.MeshStandardMaterial({ color: 0x1a1a1c, roughness: 0.93, bumpMap: pins, bumpScale: 2.2 }),
    carbon: new THREE.MeshStandardMaterial({ color: 0x1c1d20, roughness: 0.5, metalness: 0.35 }),
    alu: new THREE.MeshStandardMaterial({ color: 0x9aa2ad, roughness: 0.32, metalness: 0.9 }),
    gun: new THREE.MeshStandardMaterial({ color: 0x5b5f66, roughness: 0.35, metalness: 0.85 }),
    plastic: new THREE.MeshStandardMaterial({ color: 0x232427, roughness: 0.7 }),
    shockBody: new THREE.MeshStandardMaterial({ color: 0xc9ced6, roughness: 0.22, metalness: 0.95 }),
    shaft: new THREE.MeshStandardMaterial({ color: 0xe8ecf2, roughness: 0.1, metalness: 1 }),
    unitRod: new THREE.CylinderGeometry(1, 1, 1, 8).translate(0, 0.5, 0),           // radius 1, from y=0 to 1
    spring: helix(),
  };
  return shared;
}

/** A coil spring of unit length along +y (scaled per frame), 1 cm across. */
function helix() {
  const pts = [], turns = 7, n = 140;
  for (let i = 0; i <= n; i++) { const t = i / n, a = t * turns * Math.PI * 2; pts.push(new THREE.Vector3(0.0105 * Math.cos(a), t, 0.0105 * Math.sin(a))); }
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 280, 0.0016, 5, false);
}

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const Y_UP = V(0, 1, 0);
/** A rod mesh spanning a -> b (unit-rod geometry: scale x/z = radius, y = length). */
function setRod(m, a, b, r) {
  const d = b.clone().sub(a), len = d.length();
  m.position.copy(a);
  m.quaternion.setFromUnitVectors(Y_UP, d.divideScalar(len || 1));
  m.scale.set(r, len, r);
}
function box(mat, sx, sy, sz, x, y, z, cast = true) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  m.position.set(x, y, z); m.castShadow = cast;
  return m;
}

/** The shell: cross-sections lofted along x, each a squared-off arch (superellipse) open at the bottom. */
function shellGeometry() {
  // x, half-width, bottom z, top z (body frame): rear deck, cab, windscreen, long sloping nose
  const S = [[-0.215, 0.06, -0.012, 0.03], [-0.17, 0.072, -0.02, 0.04], [-0.105, 0.08, -0.025, 0.048], [-0.065, 0.082, -0.026, 0.076],
    [-0.005, 0.083, -0.026, 0.088], [0.035, 0.081, -0.026, 0.083], [0.075, 0.077, -0.026, 0.058], [0.12, 0.071, -0.026, 0.042],
    [0.17, 0.063, -0.026, 0.03], [0.215, 0.05, -0.022, 0.018], [0.245, 0.03, -0.012, 0.006]];
  const nx = 60, na = 28, pos = [], uv = [], idx = [];
  const lerp = (k, t) => S[k].map((v, i) => v + (S[k + 1][i] - v) * t);
  for (let i = 0; i <= nx; i++) {
    const f = (i / nx) * (S.length - 1), k = Math.min(S.length - 2, Math.floor(f));
    const [x, w, z0, z1] = lerp(k, f - k);
    for (let j = 0; j <= na; j++) {
      const a = (j / na) * Math.PI, c = Math.cos(a), s = Math.sin(a);
      const y = w * Math.sign(c) * Math.pow(Math.abs(c), 0.55), z = z0 + (z1 - z0) * Math.pow(s, 0.45);
      pos.push(x, y, z); uv.push(i / nx, j / na);
    }
  }
  for (let i = 0; i < nx; i++) for (let j = 0; j < na; j++) {
    const a = i * (na + 1) + j, b = a + na + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Livery on the shell's (u along the car rear->nose, v across from the right side over the roof to the left). */
function livery(color, accent, number) {
  const col = '#' + new THREE.Color(color).getHexString(), acc = '#' + new THREE.Color(accent).getHexString();
  const tex = canvasTexture(1024, 512, (g, W, H) => {
    // canvas x: body x from rear (-0.215) to nose (0.245); canvas y: across, one side's skirt (top) over the roof to the other (bottom)
    const U = (x) => ((x + 0.215) / 0.46) * W, Vy = (v) => v * H;
    g.fillStyle = col; g.fillRect(0, 0, W, H);
    for (const side of [0, 1]) {
      const v = (t) => Vy(side ? 1 - t : t);
      g.fillStyle = '#15161a'; g.fillRect(0, v(0), W, v(0.035) - v(0));                    // dark skirt
      g.fillStyle = '#f3f1ea';                                                              // white wedge, widening to the nose
      g.beginPath(); g.moveTo(U(-0.215), v(0.06)); g.lineTo(U(0.245), v(0.05)); g.lineTo(U(0.245), v(0.2)); g.lineTo(U(-0.215), v(0.09)); g.fill();
      g.fillStyle = acc;
      g.beginPath(); g.moveTo(U(-0.215), v(0.09)); g.lineTo(U(0.245), v(0.2)); g.lineTo(U(0.245), v(0.225)); g.lineTo(U(-0.215), v(0.105)); g.fill();
      g.fillStyle = '#10151c';                                                              // side window
      g.beginPath(); g.moveTo(U(-0.055), v(0.27)); g.lineTo(U(0.03), v(0.27)); g.lineTo(U(0.05), v(0.34)); g.lineTo(U(-0.065), v(0.34)); g.fill();
    }
    g.fillStyle = '#10151c';                                                                // windscreen
    g.beginPath(); g.moveTo(U(0.03), Vy(0.3)); g.lineTo(U(0.072), Vy(0.25)); g.lineTo(U(0.072), Vy(0.75)); g.lineTo(U(0.03), Vy(0.7)); g.fill();
    g.fillStyle = acc; g.fillRect(U(-0.215), Vy(0.46), U(0.03) - U(-0.215), Vy(0.08));     // roof stripe
    g.fillRect(U(0.08), Vy(0.47), U(0.245) - U(0.08), Vy(0.06));
    g.fillStyle = '#f3f1ea'; g.beginPath(); g.arc(U(-0.03), Vy(0.5), 30, 0, 7); g.fill();
    g.fillStyle = '#15161a'; g.font = '700 46px "Barlow Condensed", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.save(); g.translate(U(-0.03), Vy(0.5)); g.rotate(Math.PI / 2); g.fillText(String(number), 0, 3); g.restore();
    g.fillStyle = '#fff4c8'; for (const t of [0.4, 0.6]) { g.beginPath(); g.ellipse(U(0.232), Vy(t), 8, 14, 0, 0, 7); g.fill(); }
  });
  tex.anisotropy = 8;
  return tex;
}

export function makeCar(color, { ghost = false, number = 1, wheelColor = 0xd9ee2c, accent = 0x15161a } = {}) {
  const A = sharedAssets();
  const body = new THREE.Group();
  const ghostMat = ghost ? new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, depthWrite: false }) : null;
  const M = (m) => ghostMat ?? m;
  const paint = ghostMat ?? new THREE.MeshPhysicalMaterial({ map: livery(color, accent, number), roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.12, side: THREE.DoubleSide });
  const wheelMat = M(new THREE.MeshStandardMaterial({ color: wheelColor, roughness: 0.45 }));

  const shell = new THREE.Mesh(shellGeometry(), paint);
  shell.castShadow = !ghost;
  body.add(shell);

  // wing: two uprights from the rear of the chassis to a flat wing with big endplates
  const wingMat = M(new THREE.MeshStandardMaterial({ color: 0x141518, roughness: 0.4, side: THREE.DoubleSide }));
  const wing = new THREE.Group();
  wing.add(box(wingMat, 0.11, 0.22, 0.004, 0, 0, 0));
  wing.add(box(wingMat, 0.09, 0.2, 0.003, 0.005, 0, 0.022));       // the raised centre deck of a racing wing
  for (const sy of [1, -1]) {
    const ep = box(M(new THREE.MeshStandardMaterial({ color: wheelColor, roughness: 0.5, side: THREE.DoubleSide })), 0.1, 0.003, 0.034, -0.008, sy * 0.111, 0.008);
    wing.add(ep);
  }
  wing.position.set(-0.235, 0, 0.135); wing.rotation.y = 0.1;
  body.add(wing);
  if (!ghost) for (const sy of [1, -1]) body.add(box(A.carbon, 0.022, 0.004, 0.11, -0.218, sy * 0.032, 0.08).rotateY(-0.18));

  if (!ghost) {
    // chassis plate, bumpers and shock towers
    const plate = new THREE.Mesh(new THREE.ExtrudeGeometry(new THREE.Shape([[-0.22, -0.04], [0.17, -0.055], [0.235, -0.03], [0.235, 0.03], [0.17, 0.055], [-0.22, 0.04]].map(([x, y]) => new THREE.Vector2(x, y))), { depth: 0.004, bevelEnabled: false }), A.alu);
    plate.position.z = -0.038; plate.castShadow = true; body.add(plate);
    body.add(box(A.plastic, 0.03, 0.15, 0.022, 0.25, 0, -0.022));
    body.add(box(A.plastic, 0.02, 0.12, 0.02, -0.232, 0, -0.022));
    for (const sx of [0.15, -0.15]) body.add(box(A.carbon, 0.006, 0.2, 0.022, sx + (sx > 0 ? -0.03 : 0.03), 0, 0.036));
  }

  // per corner: wheel (steered knuckle + spinning tire), lower and upper arms, driveshaft, shock
  const corners = WHEELS.map((w) => {
    const [cx, cy] = CORNER[w], sy = Math.sign(cy), front = w[0] === 'f';
    const wheel = new THREE.Group(), spin = new THREE.Group();
    const tire = new THREE.Mesh(A.tireGeo, M(A.tireMat)); tire.castShadow = !ghost; spin.add(tire);
    if (!ghost) {
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(RIM_R, RIM_R, TIRE_W * 0.9, 28, 1, true), wheelMat); spin.add(rim);
      const dish = new THREE.Mesh(new THREE.CircleGeometry(RIM_R, 28), wheelMat);
      dish.position.y = sy * TIRE_W * 0.2; dish.rotation.x = sy * -Math.PI / 2; spin.add(dish);          // the outer face, set in
      const hex = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.012, 6), A.gun);
      hex.position.y = sy * (TIRE_W * 0.2 + 0.004); spin.add(hex);
      for (let k = 0; k < 5; k++) {                                                                       // dish ribs, so the spin shows
        const rib = box(M(new THREE.MeshStandardMaterial({ color: new THREE.Color(wheelColor).multiplyScalar(0.7), roughness: 0.5 })), 0.003, 0.004, RIM_R * 1.6, 0, sy * TIRE_W * 0.2, 0, false);
        rib.rotation.y = (k * Math.PI) / 5; spin.add(rib);
      }
      const knuckle = box(A.gun, 0.02, 0.01, 0.03, 0, -sy * (0.14 - HUB_Y - 0.005), 0);
      wheel.add(knuckle);
    }
    wheel.add(spin);
    body.add(wheel);
    const part = { w, cx, cy, sy, front, wheel, spin };
    if (!ghost) {
      part.lowerF = new THREE.Mesh(A.unitRod, A.plastic); part.lowerR = new THREE.Mesh(A.unitRod, A.plastic);
      part.upper = new THREE.Mesh(A.unitRod, A.gun); part.shaft = new THREE.Mesh(A.unitRod, A.alu);
      part.shockBody = new THREE.Mesh(A.unitRod, A.shockBody); part.shockRod = new THREE.Mesh(A.unitRod, A.shaft);
      part.spring = new THREE.Mesh(A.spring, new THREE.MeshStandardMaterial({ color: wheelColor, roughness: 0.4, metalness: 0.3 }));
      for (const k of ['lowerF', 'lowerR', 'upper', 'shaft', 'shockBody', 'shockRod', 'spring']) { part[k].castShadow = true; body.add(part[k]); }
    }
    return part;
  });

  const car = { body, wheels: corners.map((c) => c.wheel), corners, paint, ghost, spinAngle: 0 };
  setSuspension(car, [-0.002, -0.002, -0.002, -0.002], 0, 0);
  return car;
}

const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Vector3();
/** Pose the running gear: susp = slide travel per wheel (m, + = compressed), steer (rad), spin (rad). */
export function setSuspension(car, susp, steer, spin) {
  for (let k = 0; k < 4; k++) {
    const c = car.corners[k], zc = MOUNT_Z + susp[k];
    c.wheel.position.set(c.cx, c.cy, zc);
    c.wheel.rotation.set(0, 0, c.front ? steer : 0);
    c.spin.rotation.y = spin;
    if (car.ghost) continue;
    const sy = c.sy, hubY = sy * HUB_Y;
    // lower A-arm: two rods from the chassis pivots (front and rear of the arm) to the hub's lower ball
    const ball = tmpA.set(c.cx, hubY, zc - 0.014);
    setRod(c.lowerF, V(c.cx + 0.028, sy * 0.035, -0.034), ball, 0.0045);
    setRod(c.lowerR, V(c.cx - 0.028, sy * 0.035, -0.034), ball, 0.0045);
    // upper link (turnbuckle) to the top of the hub carrier
    setRod(c.upper, V(c.cx, sy * 0.045, 0.018), tmpB.set(c.cx, hubY, zc + 0.022), 0.0025);
    // driveshaft from the diff to the hub
    setRod(c.shaft, V(c.cx, sy * 0.03, -0.022), tmpB.set(c.cx, hubY, zc), 0.0035);
    // coil-over: from the shock tower to a mount 70% out along the lower arm
    const top = V(c.cx + (c.front ? -0.03 : 0.03), sy * 0.09, 0.042);
    const bot = tmpC.set(c.cx, sy * 0.035, -0.034).lerp(ball, 0.85);
    const len = top.distanceTo(bot);
    setRod(c.shockBody, top, top.clone().lerp(bot, Math.min(0.62, 0.046 / len)), 0.0085);
    setRod(c.shockRod, bot, top.clone().lerp(bot, 0.3), 0.0026);
    setRod(c.spring, top.clone().lerp(bot, 0.08), bot.clone().lerp(top, 0.06), 1);
    c.spring.scale.x = c.spring.scale.z = 1;
  }
}

export { WHEEL_R };
