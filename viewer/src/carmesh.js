// Proportions follow a real 1/8 truggy (TLR 8IGHT-XT: 373 mm wheelbase, 421 mm wide, 185 mm tall, 40 mm ground
// clearance) scaled to the sim's 300 mm wheelbase. The sim's own collision box (46 x 18 x 7 cm) rides higher than a real
// chassis (72 mm clearance vs about 32 mm scaled), so the visible chassis plate sits 2 cm below the box's bottom.
//
// The truggy: a 1/8-style racing truggy (low cab-forward shell, big pin-tread tires on dish wheels, long A-arms,
// coil-over shocks, a tall wing), built around the sim car's dimensions (sdc/jumps/car.py: chassis 46 x 18 x 7 cm,
// 5.5 cm wheels at x ±15 cm, y ±14 cm, mounted at z -5 cm on 8 cm of slide travel). Body frame: x forward, y left, z up.
//
// The wheels are children of the body: setSuspension() puts each one at its recorded suspension travel, steers the
// front knuckles, spins the tires, and moves the arms, driveshafts and shocks to match.
import * as THREE from 'three';
import { canvasTexture } from './terrainart.js';
import { WHEEL_R, CORNER, MOUNT_Z, WHEELS } from './carmodel.js';

const TIRE_W = 0.06;            // looks: a truggy's wide tire (the sim's capsule tread is 3 cm plus round shoulders)
const RIM_R = 0.035;
const HUB_Y = 0.106;            // |y| of the hub carrier's inner face (wheel centre at 0.14)

let shared = null;
function sharedAssets() {
  if (shared) return shared;
  // the tire's surface, unwrapped: u around the wheel (1024 px), v across the profile from one bead to the other (256 px).
  // Height: mini pins on the tread, raised lettering on both sidewalls. Colour: rubber, a little lighter on the sidewalls.
  const tread = tireTextures();
  // tire: a lathe profile (radius, y) with rounded shoulders and a little sidewall bulge
  const prof = [[0.033, -0.028], [0.043, -0.0305], [0.0505, -0.0302], [0.0542, -0.0265], [0.0552, -0.019], [0.0552, 0.019], [0.0542, 0.0265], [0.0505, 0.0302], [0.043, 0.0305], [0.033, 0.028]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const tireGeo = new THREE.LatheGeometry(prof, 64);
  // the lathe's u runs around the wheel and v across the profile: tread pins wrap around the tire
  shared = {
    tireGeo,
    tireMat: new THREE.MeshStandardMaterial({ map: tread.map, normalMap: tread.normal, normalScale: new THREE.Vector2(1.4, 1.4), roughness: 0.9 }),
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

function tireTextures() {
  const W = 1024, H = 256;
  const hc = document.createElement('canvas'); hc.width = W; hc.height = H;
  const g = hc.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  // pins: 64 around x 9 across, staggered, on the tread band (v 0.24-0.76)
  for (let i = 0; i < 64; i++) for (let j = 0; j < 9; j++) {
    const x = (i + (j % 2) * 0.5) * W / 64, y = H * (0.25 + 0.5 * (j + 0.5) / 9);
    const r = g.createRadialGradient(x, y, 0, x, y, 7.5);
    r.addColorStop(0, '#fff'); r.addColorStop(0.55, '#eee'); r.addColorStop(0.8, '#555'); r.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = r; g.fillRect(x - 8, y - 8, 16, 16);
    if (x < 8) { g.fillStyle = r; g.save(); g.translate(W, 0); g.fillRect(x - 8, y - 8, 16, 16); g.restore(); }
  }
  // sidewall lettering, raised
  g.font = '700 26px "Barlow Condensed", Arial, sans-serif'; g.textBaseline = 'middle'; g.fillStyle = '#9a9a9a';
  for (const [y, flip] of [[H * 0.1, false], [H * 0.9, true]]) for (let k = 0; k < 3; k++) {
    g.save(); g.translate(k * W / 3 + 40, y); if (flip) g.scale(1, -1); g.fillText('SDC  MINI-PIN  1/8 TRUGGY', 0, 0); g.restore();
  }
  // colour map: rubber, sidewalls a shade lighter, lettering grey
  const map = canvasTexture(W, H, (c) => {
    c.fillStyle = '#19191b'; c.fillRect(0, 0, W, H);
    c.fillStyle = '#222225'; c.fillRect(0, 0, W, H * 0.2); c.fillRect(0, H * 0.8, W, H * 0.2);
    c.globalCompositeOperation = 'lighter'; c.globalAlpha = 0.18; c.drawImage(hc, 0, 0);
  });
  // normal map from the height canvas
  const src = g.getImageData(0, 0, W, H).data, nc = document.createElement('canvas'); nc.width = W; nc.height = H;
  const nx = nc.getContext('2d'), img = nx.createImageData(W, H), d = img.data, ht = (x, y) => src[4 * (((y + H) % H) * W + ((x + W) % W))] / 255;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dx = (ht(x - 1, y) - ht(x + 1, y)) * 2.5, dy = (ht(x, y - 1) - ht(x, y + 1)) * 2.5, l = Math.hypot(dx, dy, 1), o = 4 * (y * W + x);
    d[o] = (dx / l * 0.5 + 0.5) * 255; d[o + 1] = (dy / l * 0.5 + 0.5) * 255; d[o + 2] = (1 / l * 0.5 + 0.5) * 255; d[o + 3] = 255;
  }
  nx.putImageData(img, 0, 0);
  const normal = new THREE.CanvasTexture(nc);
  normal.anisotropy = 8;
  return { map, normal };
}

/** A racing dish wheel face: the car's wheel colour, a darker bead ring, vent holes, a hex hub. */
function dishTexture(color) {
  const c = '#' + new THREE.Color(color).getHexString(), dk = '#' + new THREE.Color(color).multiplyScalar(0.55).getHexString();
  return canvasTexture(256, 256, (g, W) => {
    g.fillStyle = c; g.fillRect(0, 0, W, W);
    const R = W / 2;
    g.translate(R, R);
    const sh = g.createRadialGradient(0, 0, R * 0.2, 0, 0, R);            // the dish: lighter centre, shaded towards the rim
    sh.addColorStop(0, 'rgba(255,255,255,0.18)'); sh.addColorStop(0.7, 'rgba(0,0,0,0)'); sh.addColorStop(0.92, 'rgba(0,0,0,0.35)'); sh.addColorStop(1, 'rgba(0,0,0,0.6)');
    g.fillStyle = sh; g.beginPath(); g.arc(0, 0, R, 0, 7); g.fill();
    g.strokeStyle = dk; g.lineWidth = 7; g.beginPath(); g.arc(0, 0, R * 0.86, 0, 7); g.stroke();
    g.fillStyle = '#111';
    for (let k = 0; k < 8; k++) { const a = (k / 8) * Math.PI * 2; g.beginPath(); g.ellipse(Math.cos(a) * R * 0.6, Math.sin(a) * R * 0.6, 12, 7, a, 0, 7); g.fill(); }
    g.fillStyle = dk; g.beginPath(); g.arc(0, 0, R * 0.3, 0, 7); g.fill();
    g.fillStyle = '#6d7178'; g.beginPath();
    for (let k = 0; k < 6; k++) { const a = (k / 6) * Math.PI * 2; g[k ? 'lineTo' : 'moveTo'](Math.cos(a) * R * 0.16, Math.sin(a) * R * 0.16); } g.fill();
    g.fillStyle = '#2a2c30'; g.beginPath(); g.arc(0, 0, R * 0.07, 0, 7); g.fill();
  });
}

/** The ghost: a hologram (fresnel rim, scan lines), additive, shared clock in holoTime. */
export const holoTime = { value: 0 };
let holoMat = null;
export function hologram() {
  return holoMat ??= new THREE.ShaderMaterial({
    uniforms: { time: holoTime, tint: { value: new THREE.Color(0x7fe4ff) } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    vertexShader: `varying vec3 vN; varying vec3 vV; varying float vZ;
      void main(){ vec4 wp = modelMatrix * vec4(position, 1.0); vZ = wp.z; vec4 mv = viewMatrix * wp;
        vN = normalize(mat3(modelMatrix) * normal); vV = normalize(cameraPosition - wp.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform float time; uniform vec3 tint; varying vec3 vN; varying vec3 vV; varying float vZ;
      void main(){ float f = pow(1.0 - abs(dot(normalize(vN), vV)), 2.2);
        float scan = 0.65 + 0.35 * sin(vZ * 260.0 - time * 9.0);
        float flick = 0.92 + 0.08 * sin(time * 37.0);
        gl_FragColor = vec4(tint * (0.08 + 0.9 * f) * scan * flick, 1.0); }`,
  });
}

/** A coil spring of unit length along +y (scaled per frame), 2.8 cm across: a big-bore truggy shock's spring. */
function helix() {
  const pts = [], turns = 8, n = 200;
  for (let i = 0; i <= n; i++) { const t = i / n, a = t * turns * Math.PI * 2; pts.push(new THREE.Vector3(0.0135 * Math.cos(a), t, 0.0135 * Math.sin(a))); }
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 400, 0.0021, 6, false);
}

const V = (x, y, z) => new THREE.Vector3(x, y, z);
const Y_UP = V(0, 1, 0);
/** Point a +y-length-1 part from a to b: position at a, +y towards b, length scaled. */
function setRod(m, a, b, r) {
  const d = b.clone().sub(a), len = d.length();
  m.position.copy(a);
  m.quaternion.setFromUnitVectors(Y_UP, d.divideScalar(len || 1));
  m.scale.set(r, len, r);
}
function aim(g, a, b) {
  const d = b.clone().sub(a), len = d.length();
  g.position.copy(a);
  g.quaternion.setFromUnitVectors(Y_UP, d.divideScalar(len || 1));
  return len;
}
function box(mat, sx, sy, sz, x, y, z, cast = true) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  m.position.set(x, y, z); m.castShadow = cast;
  return m;
}
/** A flat plate from a 2D outline: 'xz' = (x, z) points, thickness along y centred on y0; 'yz' = (y, z) points at x = y0;
 *  'xy' = (x, y) points at z = y0. */
function plate(mat, pts, t, plane = 'xz', y0 = 0) {
  const g = new THREE.ExtrudeGeometry(new THREE.Shape(pts.map(([a, b]) => new THREE.Vector2(a, b))), { depth: t, bevelEnabled: true, bevelThickness: 0.001, bevelSize: 0.001, bevelSegments: 1 });
  if (plane === 'xz') { g.rotateX(Math.PI / 2); g.translate(0, y0 + t / 2, 0); }
  else if (plane === 'yz') { g.rotateX(Math.PI / 2); g.rotateZ(Math.PI / 2); g.translate(y0 - t / 2, 0, 0); }
  else g.translate(0, 0, y0 - t / 2);
  const m = new THREE.Mesh(g, mat); m.castShadow = true; m.receiveShadow = true;
  return m;
}

// The shell: a small truck cab on a narrow body (about half the car's width), lofted from cross-sections along x,
// each a squared-off arch open at the bottom. The chassis, the front and rear ends and the shocks stay exposed.
const SHELL = [  // x, half-width, bottom z, top z
  [-0.15, 0.052, -0.046, -0.012], [-0.12, 0.066, -0.05, -0.002], [-0.082, 0.073, -0.052, 0.004], [-0.064, 0.074, -0.052, 0.024],
  [-0.025, 0.075, -0.052, 0.034], [0.018, 0.074, -0.052, 0.032], [0.06, 0.071, -0.052, 0.006], [0.105, 0.066, -0.052, -0.006],
  [0.155, 0.058, -0.05, -0.016], [0.198, 0.046, -0.048, -0.027], [0.225, 0.03, -0.045, -0.035]];
const SX0 = SHELL[0][0], SX1 = SHELL[SHELL.length - 1][0];
function shellGeometry() {
  const nx = 90, na = 36, pos = [], uv = [], idx = [];
  const at = (f) => { const k = Math.min(SHELL.length - 2, Math.floor(f)), t = f - k; return SHELL[k].map((v, i) => v + (SHELL[k + 1][i] - v) * (t * t * (3 - 2 * t) * 0.35 + t * 0.65)); };
  for (let i = 0; i <= nx; i++) {
    const [x, w, z0, z1] = at((i / nx) * (SHELL.length - 1));
    // texture u follows x; v follows arc length around the section, so the livery keeps its shape on sides and roof
    const ring = [];
    for (let j = 0; j <= na; j++) {
      const a = (j / na) * Math.PI, c = Math.cos(a), s = Math.sin(a);
      ring.push([w * Math.sign(c) * Math.pow(Math.abs(c), 0.3), z0 + (z1 - z0) * Math.pow(s, 0.25)]);
    }
    const arc = [0];
    for (let j = 1; j <= na; j++) arc.push(arc[j - 1] + Math.hypot(ring[j][0] - ring[j - 1][0], ring[j][1] - ring[j - 1][1]));
    for (let j = 0; j <= na; j++) { pos.push(x, ring[j][0], ring[j][1]); uv.push((x - SX0) / (SX1 - SX0), arc[j] / arc[na]); }
  }
  for (let i = 0; i < nx; i++) for (let j = 0; j < na; j++) { const a = i * (na + 1) + j, b = a + na + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Livery: a two-tone fade with flames licking back from the nose, a dark cab glasshouse and a race number.
 *  Canvas x = body x (rear to nose); canvas y = across, one side's skirt (top) over the roof to the other (bottom). */
function livery(scheme, number) {
  const hex = (c) => '#' + new THREE.Color(c).getHexString();
  const tex = canvasTexture(2048, 1024, (g, W, H) => {
    const U = (x) => ((x - SX0) / (SX1 - SX0)) * W;
    const fade = g.createLinearGradient(U(SX0), 0, U(SX1), 0);
    fade.addColorStop(0, hex(scheme.b)); fade.addColorStop(0.45, hex(scheme.b)); fade.addColorStop(0.62, hex(scheme.a)); fade.addColorStop(1, hex(scheme.a));
    g.fillStyle = fade; g.fillRect(0, 0, W, H);
    // flames: tongues from the front quarter sweeping back along each side and over the hood
    const flame = (y0, dir, n, len, amp) => {
      for (let k = 0; k < n; k++) {
        const yb = y0 + dir * k * amp * 0.55, tip = U(0.11) - len * (0.55 + 0.45 * Math.sin(k * 2.3 + 1)) * W, wav = amp * 0.6;
        const grd = g.createLinearGradient(U(0.13), 0, tip, 0);
        grd.addColorStop(0, hex(scheme.flame)); grd.addColorStop(0.7, hex(scheme.flame2)); grd.addColorStop(1, hex(scheme.flame2));
        g.fillStyle = grd; g.strokeStyle = '#f6f4ee'; g.lineWidth = 6;
        g.beginPath(); g.moveTo(U(0.13), yb - amp * 0.5);
        g.bezierCurveTo((U(0.13) + tip) / 2, yb - amp * 0.9 - wav, (U(0.13) + tip) / 2, yb + wav, tip, yb + dir * amp * 0.25);
        g.bezierCurveTo((U(0.13) + tip) / 2 + 40, yb + amp * 0.2, (U(0.13) + tip) / 2, yb + amp * 0.9, U(0.13), yb + amp * 0.5);
        g.closePath(); g.stroke(); g.fill();
      }
    };
    flame(H * 0.11, 1, 4, 0.42, H * 0.06);
    flame(H * 0.89, -1, 4, 0.42, H * 0.06);
    g.fillStyle = hex(scheme.flame);
    g.fillRect(U(0.12), H * 0.3, U(SX1) - U(0.12), H * 0.4);                    // hood panel
    g.fillStyle = hex(scheme.flame2); g.fillRect(U(0.12), H * 0.46, U(SX1) - U(0.12), H * 0.08);
    // the glasshouse: windscreen, side windows, a rear window, with a black window line around them
    g.fillStyle = '#0d1117'; g.strokeStyle = '#05070a'; g.lineWidth = 10;
    g.beginPath(); g.moveTo(U(0.025), H * 0.31); g.lineTo(U(0.068), H * 0.35); g.lineTo(U(0.068), H * 0.65); g.lineTo(U(0.025), H * 0.69); g.closePath(); g.fill(); g.stroke();
    for (const [a, b] of [[0.2, 0.3], [0.8, 0.7]]) {
      g.beginPath(); g.moveTo(U(-0.052), H * a); g.lineTo(U(0.018), H * a); g.lineTo(U(0.04), H * b); g.lineTo(U(-0.052), H * b); g.closePath(); g.fill(); g.stroke();
    }
    g.beginPath(); g.moveTo(U(-0.06), H * 0.36); g.lineTo(U(-0.05), H * 0.36); g.lineTo(U(-0.05), H * 0.64); g.lineTo(U(-0.06), H * 0.64); g.fill();
    // skirt, door numbers and a roof number
    g.fillStyle = '#121316'; g.fillRect(0, 0, W, H * 0.03); g.fillRect(0, H * 0.97, W, H * 0.03);
    g.font = '800 120px "Barlow Condensed", Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    for (const [y, flip] of [[H * 0.14, true], [H * 0.86, false]]) {
      g.save(); g.translate(U(-0.015), y); if (flip) g.scale(1, -1);
      g.lineWidth = 12; g.strokeStyle = '#111'; g.strokeText(String(number), 0, 6); g.fillStyle = '#f6f4ee'; g.fillText(String(number), 0, 6); g.restore();
    }
    g.fillStyle = '#f6f4ee'; g.beginPath(); g.arc(U(-0.02), H * 0.5, 62, 0, 7); g.fill();
    g.fillStyle = '#111'; g.save(); g.translate(U(-0.02), H * 0.5); g.rotate(Math.PI / 2); g.fillText(String(number), 0, 6); g.restore();
    // headlights
    g.fillStyle = '#fff6d0'; for (const v of [0.38, 0.62]) { g.beginPath(); g.ellipse(U(0.205), H * v, 14, 30, 0, 0, 7); g.fill(); }
  });
  tex.anisotropy = 8;
  return tex;
}

const SCHEMES = {
  1: { a: 0x37d34b, b: 0x1d4fd8, flame: 0xff7418, flame2: 0xffc21a },     // main car: green to blue, orange flames
  2: { a: 0xf3f1ea, b: 0xc81e2b, flame: 0x15161a, flame2: 0x4a4d55 },     // rival: white to red, black flames
};

/** The sim's car geometry (record.py writes it into each recording): the old 1/10-size car by default. */
export const SMALL = { wheel_r: WHEEL_R, corner_x: CORNER.fl[0], corner_y: CORNER.fl[1], mount_z: MOUNT_Z, box: [0.23, 0.09, 0.035] };

export function makeCar(color, { ghost = false, number = 1, wheelColor = 0xf3f2ee, accent = 0x15161a, geom = SMALL } = {}) {
  const A = sharedAssets();
  const body = new THREE.Group();
  // The body kit (shell, chassis plate, towers, wing) is drawn at the old car's size and scaled to the sim's wheelbase.
  // Its chassis plate sits at the sim's collision-box bottom when that is a realistic ride height (the full-size
  // truggy: about 3 cm), else 5 cm above the ground (the old car's box rode 7 cm up). Wheels and links follow the sim.
  const sc = geom.corner_x / 0.15, kr = geom.wheel_r / WHEEL_R;
  const ground = geom.mount_z - 0.002 - geom.wheel_r, boxBot = -geom.box[2];
  const plateZ = boxBot - ground <= 0.045 ? boxBot : ground + 0.052;
  const kitZ = plateZ + 0.055 * sc;
  const kit = new THREE.Group(); kit.scale.setScalar(sc); kit.position.z = kitZ; body.add(kit);
  const K = (x, y, z) => new THREE.Vector3(x * sc, y * sc, kitZ + z * sc);
  const ghostMat = ghost ? hologram() : null;
  const M = (m) => ghostMat ?? m;
  const scheme = SCHEMES[number] ?? SCHEMES[1];
  const paint = ghostMat ?? new THREE.MeshPhysicalMaterial({ map: livery(scheme, number), roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.14, envMapIntensity: 2.4, side: THREE.DoubleSide });
  const wheelMat = M(new THREE.MeshStandardMaterial({ color: wheelColor, roughness: 0.42 }));
  const grey = M(new THREE.MeshStandardMaterial({ color: 0x6c7077, roughness: 0.55, metalness: 0.15 }));   // composite towers, wing mount

  const shell = new THREE.Mesh(shellGeometry(), paint);
  shell.castShadow = !ghost;
  kit.add(shell);

  // chassis plate (long and narrow, exposed front and rear), front bumper and skid
  kit.add(plate(M(A.carbon), [[-0.245, -0.045], [0.17, -0.05], [0.235, -0.03], [0.235, 0.03], [0.17, 0.05], [-0.245, 0.045]], 0.004, 'xy', -0.055));
  const bump = plate(M(A.plastic), [[0.215, -0.058], [0.268, -0.046], [0.276, -0.03], [0.268, -0.026], [0.222, -0.046]], 0.12, 'xz');
  kit.add(bump);
  // shock towers: grey composite plates, wide at the top where the shocks mount
  const tower = (x) => plate(grey, [[-0.05, -0.053], [0.05, -0.053], [0.088, 0.024], [0.082, 0.036], [0.058, 0.036], [0.026, 0.004], [-0.026, 0.004], [-0.058, 0.036], [-0.082, 0.036], [-0.088, 0.024]], 0.006, 'yz', x);
  if (!ghost) {
    const ft = tower(0.122), rt = tower(-0.122);
    kit.add(ft, rt);
    // diff housings and a centre driveshaft, visible through the open chassis
    for (const x of [0.15, -0.15]) kit.add(box(A.gun, 0.05, 0.06, 0.036, x, 0, -0.034));
    const cs = new THREE.Mesh(A.unitRod, A.alu); setRod(cs, V(0.125, 0, -0.038), V(-0.125, 0, -0.038), 0.004); kit.add(cs);
  }

  // wing: a wide swept wing on a grey mount from the rear tower, white endplates
  const wing = new THREE.Group();
  const foil = new THREE.Shape([[0.06, 0], [0.02, 0.009], [-0.03, 0.01], [-0.06, 0.004], [-0.065, 0.016], [-0.069, 0.016], [-0.062, -0.001], [0.06, -0.003]].map(([x, z]) => new THREE.Vector2(x, z)));
  const fg = new THREE.ExtrudeGeometry(foil, { depth: 0.25, bevelEnabled: false });
  fg.rotateX(Math.PI / 2); fg.translate(0, 0.125, 0);
  const wingMat = M(new THREE.MeshPhysicalMaterial({ color: 0x15171a, roughness: 0.25, clearcoat: 0.6, clearcoatRoughness: 0.2, side: THREE.DoubleSide }));
  const wm = new THREE.Mesh(fg, wingMat); wm.castShadow = !ghost; wing.add(wm);
  for (const sy of [1, -1]) {
    const ep = plate(M(new THREE.MeshStandardMaterial({ color: 0xf3f2ee, roughness: 0.4 })), [[0.065, -0.02], [-0.07, -0.025], [-0.075, 0.03], [0.03, 0.022]], 0.0025, 'xz', sy * 0.126);
    wing.add(ep);
  }
  for (const y of [-0.06, 0.0, 0.06]) wing.add(box(wingMat, 0.1, 0.002, 0.008, -0.005, y, 0.013));
  wing.position.set(-0.225, 0, 0.058); wing.rotation.y = 0.05;
  kit.add(wing);
  if (!ghost) for (const sy of [1, -1]) {
    const strut = plate(grey, [[-0.122, 0.03], [-0.108, 0.03], [-0.21, 0.06], [-0.234, 0.06]], 0.005, 'xz', sy * 0.03);
    kit.add(strut);
  }

  // per corner: wheel (steered knuckle + spinning tire), lower and upper arms, driveshaft, shock
  const corners = WHEELS.map((w) => {
    const front = w[0] === 'f', sy = w[1] === 'l' ? 1 : -1, cx = (front ? 1 : -1) * geom.corner_x, cy = sy * geom.corner_y;
    const wheel = new THREE.Group(), spin = new THREE.Group();
    const tire = new THREE.Mesh(A.tireGeo, M(A.tireMat)); tire.castShadow = !ghost; spin.add(tire);
    if (!ghost) {
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(RIM_R, RIM_R, TIRE_W * 0.92, 32, 1, true), wheelMat); spin.add(rim);
      const dish = new THREE.Mesh(new THREE.CircleGeometry(RIM_R, 32), new THREE.MeshStandardMaterial({ map: dishTexture(wheelColor), roughness: 0.38 }));
      dish.position.y = sy * TIRE_W * 0.16; dish.rotation.x = sy * -Math.PI / 2; spin.add(dish);
      const hex = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.012, 6), A.gun);
      hex.position.y = sy * (TIRE_W * 0.16 + 0.004); spin.add(hex);
      const knuckle = box(A.gun, 0.024, 0.012, 0.034, 0, -sy * (0.14 - HUB_Y - 0.004), 0);
      wheel.add(knuckle);
    }
    wheel.add(spin);
    wheel.scale.setScalar(kr);
    body.add(wheel);
    const part = { w, cx, cy, sy, front, wheel, spin, dx: cx / sc };
    if (!ghost) {
      part.lowerF = new THREE.Mesh(A.unitRod, A.plastic); part.lowerR = new THREE.Mesh(A.unitRod, A.plastic);
      part.upper = new THREE.Mesh(A.unitRod, A.gun); part.shaft = new THREE.Mesh(A.unitRod, A.alu);
      for (const k of ['lowerF', 'lowerR', 'upper', 'shaft']) { part[k].castShadow = true; body.add(part[k]); }
      // the shock: a group aimed from the tower top to the arm, +y down the shock
      const sh = part.shock = new THREE.Group();
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.0125, 0.0125, 0.008, 18).translate(0, 0.004, 0), A.gun);
      const bodyC = new THREE.Mesh(new THREE.CylinderGeometry(0.0105, 0.0105, 0.04, 18).translate(0, 0.028, 0), A.shockBody);
      const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.0155, 0.0155, 0.004, 18).translate(0, 0.012, 0), M(new THREE.MeshStandardMaterial({ color: accent === 0x15161a ? 0xff7418 : 0xc81e2b, roughness: 0.35, metalness: 0.7 })));
      part.rod = new THREE.Mesh(A.unitRod, A.shaft);
      part.spring = new THREE.Mesh(A.spring, M(new THREE.MeshStandardMaterial({ color: 0x1b1c20, roughness: 0.35, metalness: 0.6 })));
      part.eye = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.01, 10), A.gun);
      for (const m of [cap, bodyC, collar, part.rod, part.spring, part.eye]) { m.castShadow = true; sh.add(m); }
      sh.scale.setScalar(sc);
      body.add(sh);
    }
    return part;
  });

  const car = { body, wheels: corners.map((c) => c.wheel), corners, paint, ghost, spinAngle: 0, geom, R: geom.wheel_r, sc, kr, K,
    hubY: geom.corner_y - (0.14 - HUB_Y) * kr };
  setSuspension(car, [-0.002, -0.002, -0.002, -0.002], 0, 0);
  return car;
}

const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Vector3();
/** Pose the running gear: susp = slide travel per wheel (m, + = compressed), steer (rad), spin (rad). */
export function setSuspension(car, susp, steer, spin) {
  const { K, kr, sc } = car;
  for (let k = 0; k < 4; k++) {
    const c = car.corners[k], zc = car.geom.mount_z + susp[k];
    c.wheel.position.set(c.cx, c.cy, zc);
    c.wheel.rotation.set(0, 0, c.front ? steer : 0);
    c.spin.rotation.y = spin;
    if (car.ghost) continue;
    const sy = c.sy, hubY = sy * car.hubY, dx = c.dx;
    // lower A-arm: two rods from the chassis pivots (front and rear of the arm) to the hub's lower ball
    const ball = tmpA.set(c.cx, hubY, zc - 0.015 * kr);
    setRod(c.lowerF, K(dx + 0.03, sy * 0.034, -0.05), ball, 0.005 * sc);
    setRod(c.lowerR, K(dx - 0.03, sy * 0.034, -0.05), ball, 0.005 * sc);
    // upper link (turnbuckle) to the top of the hub carrier
    setRod(c.upper, K(dx + (c.front ? -0.02 : 0.02), sy * 0.045, -0.008), tmpB.set(c.cx, hubY, zc + 0.024 * kr), 0.0028 * sc);
    // driveshaft from the diff to the hub
    setRod(c.shaft, K(dx, sy * 0.03, -0.036), tmpB.set(c.cx, hubY, zc), 0.0038 * sc);
    // coil-over: from the top of the tower, leaning in, to a mount near the outer end of the lower arm
    const top = K(dx + (c.front ? -0.028 : 0.028), sy * 0.074, 0.03);
    const bot = tmpC.copy(K(dx, sy * 0.034, -0.05)).lerp(ball, 0.86);
    const len = aim(c.shock, top, bot) / sc;                    // in the shock group's own (unscaled) units
    c.shock.scale.setScalar(sc);
    c.rod.position.set(0, 0.04, 0); c.rod.scale.set(0.0032, Math.max(0.001, len - 0.04), 0.0032);
    c.spring.position.set(0, 0.014, 0); c.spring.quaternion.identity(); c.spring.scale.set(1, Math.max(0.01, len - 0.02), 1);
    c.eye.position.set(0, len, 0); c.eye.rotation.set(0, 0, Math.PI / 2);
  }
}

export { WHEEL_R };
