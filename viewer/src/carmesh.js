// The truggy's looks, from rc-racer's buggy (tschomay/sandbox projects/rc-racer/src/carmesh.js), reshaped as a
// truggy: a tall cab-forward shell, wide knobby tires, nerf bars and a big wing. Built around the physics model's
// dimensions (chassis 46 x 18 x 7 cm, 5.5 cm wheels at ±15 cm, ±14 cm). Body frame: x forward, y left, z up.
import * as THREE from 'three';
import { canvasTexture } from './terrainart.js';
import { WHEEL_R, CORNER, MOUNT_Z } from './carmodel.js';

let shared = null;
function sharedAssets() {
  if (shared) return shared;
  const tread = canvasTexture(128, 64, (g, w, h) => {
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#fff';
    for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) g.fillRect(i * 32 + (j % 2) * 16 + 2, j * 13 + 1, 13, 9);   // big knobs
  }, false);
  tread.wrapS = tread.wrapT = THREE.RepeatWrapping;
  tread.repeat.set(9, 1);
  const tireProfile = [[0.033, -0.03], [0.047, -0.032], [0.0535, -0.026], [0.055, -0.012], [0.055, 0.012], [0.0535, 0.026], [0.047, 0.032], [0.033, 0.03]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const spokes = canvasTexture(128, 128, (g, w) => {
    g.fillStyle = '#e9e6dc'; g.fillRect(0, 0, w, w);
    g.translate(w / 2, w / 2);
    g.fillStyle = '#17181a';
    for (let k = 0; k < 6; k++) { g.rotate(Math.PI / 3); g.beginPath(); g.moveTo(10, -9); g.lineTo(52, -17); g.lineTo(52, 17); g.lineTo(10, 9); g.fill(); }
    g.fillStyle = '#8a8d92'; g.beginPath(); g.arc(0, 0, 9, 0, 7); g.fill();
  });
  shared = {
    tireGeo: new THREE.LatheGeometry(tireProfile, 28),
    tireMat: new THREE.MeshStandardMaterial({ color: 0x1b1b1d, roughness: 0.92, bumpMap: tread, bumpScale: 1.6 }),
    rimGeo: new THREE.CylinderGeometry(0.034, 0.034, 0.05, 24),
    rimMat: [new THREE.MeshStandardMaterial({ color: 0xdedbd0, roughness: 0.45 }),
      new THREE.MeshStandardMaterial({ map: spokes, roughness: 0.45 }), new THREE.MeshStandardMaterial({ map: spokes, roughness: 0.45 })],
    carbon: new THREE.MeshStandardMaterial({ color: 0x18191b, roughness: 0.55, metalness: 0.3 }),
    alloy: new THREE.MeshStandardMaterial({ color: 0x9aa0a8, roughness: 0.35, metalness: 0.9 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x0b0f14, roughness: 0.08, metalness: 0.2, clearcoat: 1 }),
    white: new THREE.MeshStandardMaterial({ color: 0xf2efe6, roughness: 0.5 }),
  };
  return shared;
}

function box(mat, sx, sy, sz, x, y, z) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  return m;
}
function rod(mat, a, b, r) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b), len = va.distanceTo(vb);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 8), mat);
  m.position.copy(va).add(vb).multiplyScalar(0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
  m.castShadow = true;
  return m;
}

export function makeCar(color, { ghost = false, number = 1 } = {}) {
  const A = sharedAssets();
  const body = new THREE.Group();
  const paint = ghost
    ? new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, depthWrite: false })
    : new THREE.MeshPhysicalMaterial({ color, roughness: 0.38, metalness: 0.0, clearcoat: 0.9, clearcoatRoughness: 0.18 });
  const M = (m) => (ghost ? paint : m);

  // shell: side profile extruded across the car, cab forward, low nose
  const sh = new THREE.Shape();
  [[-0.21, -0.012], [-0.21, 0.035], [-0.16, 0.05], [-0.1, 0.06], [-0.06, 0.112], [0.03, 0.118], [0.075, 0.072], [0.16, 0.05], [0.225, 0.02], [0.23, -0.012]]
    .forEach(([x, z], i) => (i ? sh.lineTo(x, z) : sh.moveTo(x, z)));
  sh.closePath();
  const shellGeo = new THREE.ExtrudeGeometry(sh, { depth: 0.13, bevelEnabled: true, bevelThickness: 0.016, bevelSize: 0.012, bevelSegments: 4, curveSegments: 4 });
  shellGeo.translate(0, 0, -0.065);
  shellGeo.rotateX(Math.PI / 2);
  shellGeo.computeVertexNormals();
  const shell = new THREE.Mesh(shellGeo, paint);
  shell.castShadow = !ghost;
  body.add(shell);

  if (!ghost) {
    const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12), A.glass);
    canopy.scale.set(0.05, 0.07, 0.035);
    canopy.position.set(0.035, 0, 0.1);
    body.add(canopy);
    body.add(box(A.white, 0.12, 0.022, 0.002, 0.11, 0, 0.0475).rotateY(0.345));          // nose stripe
    // number plates on the wing endplates and nose are too small to read at RC scale: a roof roundel instead
    const roundel = new THREE.Mesh(new THREE.CircleGeometry(0.024, 20), new THREE.MeshStandardMaterial({
      map: canvasTexture(64, 64, (g) => {
        g.fillStyle = '#f2efe6'; g.beginPath(); g.arc(32, 32, 31, 0, 7); g.fill();
        g.fillStyle = '#15161a'; g.font = '700 46px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(number), 32, 35);
      }), roughness: 0.5, transparent: true }));
    roundel.position.set(-0.02, 0, 0.1315); roundel.rotation.y = 0; roundel.rotation.z = -Math.PI / 2;
    body.add(roundel);
  }
  body.add(box(M(A.carbon), 0.41, 0.13, 0.006, 0, 0, -0.032));                               // chassis plate
  body.add(box(M(A.carbon), 0.02, 0.2, 0.03, 0.232, 0, -0.022));                             // front bumper
  body.add(box(M(A.carbon), 0.012, 0.15, 0.022, -0.226, 0, -0.022));                         // rear bumper
  for (const sx of [0.15, -0.15]) {
    body.add(box(M(A.carbon), 0.006, 0.17, 0.07, sx, 0, 0.022));                             // shock towers
    for (const sy of [1, -1]) {
      body.add(box(M(A.carbon), 0.05, 0.1, 0.006, sx, sy * 0.075, MOUNT_Z + 0.005));         // suspension arms
      if (!ghost) {
        body.add(rod(A.alloy, [sx + 0.012, sy * 0.07, 0.055], [sx + 0.012, sy * 0.115, MOUNT_Z + 0.012], 0.004));
        body.add(rod(new THREE.MeshStandardMaterial({ color: 0xe9c531, roughness: 0.5 }),
          [sx + 0.012, sy * 0.078, 0.042], [sx + 0.012, sy * 0.108, MOUNT_Z + 0.022], 0.0075));   // springs
      }
    }
  }
  // rear wing
  const wing = new THREE.Group();
  wing.add(box(paint, 0.1, 0.24, 0.004, 0, 0, 0));
  for (const sy of [1, -1]) wing.add(box(M(A.white), 0.1, 0.003, 0.06, 0, sy * 0.12, 0.008));
  wing.position.set(-0.215, 0, 0.14); wing.rotation.y = 0.15;
  body.add(wing);
  for (const sy of [1, -1]) body.add(rod(M(A.carbon), [-0.17, sy * 0.04, 0.045], [-0.21, sy * 0.05, 0.138], 0.003));
  for (const sy of [1, -1]) {                                                                // nerf bars
    body.add(rod(M(A.carbon), [0.08, sy * 0.1, -0.028], [-0.08, sy * 0.1, -0.028], 0.005));
    body.add(rod(M(A.carbon), [0.08, sy * 0.1, -0.028], [0.1, sy * 0.06, -0.03], 0.004));
    body.add(rod(M(A.carbon), [-0.08, sy * 0.1, -0.028], [-0.1, sy * 0.06, -0.03], 0.004));
  }
  if (!ghost) {
    body.add(rod(A.white, [-0.13, 0.05, 0.05], [-0.14, 0.05, 0.29], 0.0015));                // antenna
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.005, 8, 6), new THREE.MeshStandardMaterial({ color: 0xff6a1f, roughness: 0.4 }));
    tip.position.set(-0.14, 0.05, 0.29);
    body.add(tip);
  }

  const wheels = [];
  for (const w of ['fl', 'fr', 'rl', 'rr']) {
    const g = new THREE.Group();
    const tire = new THREE.Mesh(A.tireGeo, M(A.tireMat));
    tire.castShadow = !ghost;
    g.add(tire);
    if (!ghost) g.add(new THREE.Mesh(A.rimGeo, A.rimMat));
    g.userData.rest = new THREE.Vector3(CORNER[w][0], CORNER[w][1], MOUNT_Z - 0.006);
    const spin = new THREE.Group();          // the spinning part, inside the (steered) wheel group
    spin.add(...g.children);
    g.add(spin);
    g.userData.spin = spin;
    wheels.push(g);
  }
  return { body, wheels, paint };
}

export { WHEEL_R };
