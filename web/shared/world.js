// Set dressing around a loop: wind-blown grass tufts and rocks, instanced, kept off the lane and its banks.
import * as THREE from 'three';
import { canvasTexture, fbm } from './terrainart.js';
import { LANE_W, BANK_RUN, rng32 } from './track.js';

export const wind = { value: 0 };     // shared time uniform for the grass sway

let grassTex = null;
function bladesTexture() {
  if (grassTex) return grassTex;
  grassTex = canvasTexture(128, 128, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    const rng = rng32(7);
    for (let i = 0; i < 26; i++) {
      const x0 = 20 + rng() * 88, lean = (rng() - 0.5) * 50, top = 8 + rng() * 50, wd = 2 + rng() * 3.5;
      const grd = g.createLinearGradient(0, h, 0, top);
      const hue = 62 + rng() * 22, sat = 32 + rng() * 20;
      grd.addColorStop(0, `hsl(${hue}, ${sat}%, 30%)`); grd.addColorStop(0.5, `hsl(${hue}, ${sat}%, 44%)`); grd.addColorStop(1, `hsl(${hue - 18}, ${sat - 8}%, 66%)`);
      g.fillStyle = grd;
      g.beginPath(); g.moveTo(x0 - wd, h); g.quadraticCurveTo(x0 + lean * 0.3, (h + top) / 2, x0 + lean, top); g.quadraticCurveTo(x0 + lean * 0.3 + wd * 0.4, (h + top) / 2, x0 + wd, h); g.fill();
    }
  });
  return grassTex;
}

function grassMaterial() {
  const m = new THREE.MeshLambertMaterial({ map: bladesTexture(), alphaTest: 0.45, side: THREE.DoubleSide });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.wind = wind;
    sh.vertexShader = 'uniform float wind;\n' + sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
      vec3 ip = vec3(instanceMatrix[3][0], instanceMatrix[3][1], 0.0);
      float sway = sin(wind * 1.7 + ip.x * 0.6 + ip.y * 0.4) + 0.5 * sin(wind * 3.1 + ip.x * 1.3);
      transformed.x += sway * 0.12 * uv.y * uv.y;
      transformed.y += sway * 0.05 * uv.y * uv.y;`);
    // a little light through the blades
    sh.fragmentShader = sh.fragmentShader.replace('#include <opaque_fragment>', 'outgoingLight += diffuseColor.rgb * 0.18;\n#include <opaque_fragment>');
  };
  return m;
}

/** Grass and rocks for circuit c. n: grass tuft count. Returns the group (added to `parent`). */
export function dress(c, parent, n) {
  const group = new THREE.Group();
  const rng = rng32(c.seed * 31 + 5), keep = LANE_W / 2 + BANK_RUN + 0.3;
  const pad = 6;
  const pick = () => {
    for (let k = 0; k < 40; k++) {
      const x = c.x0 - pad + rng() * (c.sizeX + 2 * pad), y = c.y0 - pad + rng() * (c.sizeY + 2 * pad);
      const q = c.locate(x, y);
      if (q && Math.abs(q.d) < keep) continue;
      // denser near the track and in patches
      const near = q ? 1 : 0.45;
      if (rng() > near * (0.35 + 0.65 * fbm(x * 0.25, y * 0.25, 3))) continue;
      const inside = x > c.x0 && x < c.x0 + c.sizeX && y > c.y0 && y < c.y0 + c.sizeY;
      return [x, y, inside ? c.height(x, y) : c.plateau];
    }
    return null;
  };
  // grass: two crossed quads per tuft, rooted at z = 0
  const quad = new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0).rotateX(Math.PI / 2);
  const cross = quad.clone().rotateZ(Math.PI / 2);
  const geo = mergeTwo(quad, cross);
  const grass = new THREE.InstancedMesh(geo, grassMaterial(), n);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), col = new THREE.Color();
  let k = 0;
  for (let i = 0; i < n; i++) {
    const at = pick(); if (!at) continue;
    const h = 0.07 + rng() * 0.16, w = h * (1.2 + rng() * 0.8);
    q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), rng() * Math.PI);
    m4.compose(p.set(at[0], at[1], at[2] - 0.01), q, s.set(w, w, h));
    grass.setMatrixAt(k, m4);
    col.setRGB(1, 1, 1).multiplyScalar(0.85 + rng() * 0.3); grass.setColorAt(k, col);
    k++;
  }
  grass.count = k; grass.receiveShadow = true;
  group.add(grass);
  // rocks: low-poly, lumpy, on the bank tops and in the grass
  const rockGeo = new THREE.IcosahedronGeometry(1, 1);
  const rp = rockGeo.attributes.position;
  for (let i = 0; i < rp.count; i++) { const f = 0.75 + 0.5 * fbm(rp.getX(i) * 2 + 9, rp.getY(i) * 2 + rp.getZ(i) * 2, 2); rp.setXYZ(i, rp.getX(i) * f, rp.getY(i) * f, rp.getZ(i) * f * 0.7); }
  rockGeo.computeVertexNormals();
  const nr = Math.round(n / 12);
  const rocks = new THREE.InstancedMesh(rockGeo, new THREE.MeshStandardMaterial({ color: 0xb3a58e, roughness: 0.92, flatShading: true }), nr);
  k = 0;
  for (let i = 0; i < nr; i++) {
    const at = pick(); if (!at) continue;
    const r = 0.015 + Math.pow(rng(), 3) * 0.12;
    q.setFromEuler(new THREE.Euler(rng() * 6, rng() * 6, rng() * 6));
    m4.compose(p.set(at[0], at[1], at[2] + r * 0.2), q, s.set(r, r * (0.7 + rng() * 0.6), r));
    rocks.setMatrixAt(k, m4);
    col.setHSL(0.08 + rng() * 0.04, 0.12 + rng() * 0.1, 0.5 + rng() * 0.25); rocks.setColorAt(k, col);
    k++;
  }
  rocks.count = k; rocks.castShadow = true; rocks.receiveShadow = true;
  group.add(rocks);
  parent.add(group);
  return { group, grass };
}

function mergeTwo(a, b) {
  const g = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv']) {
    const A = a.attributes[name], B = b.attributes[name];
    const arr = new Float32Array(A.array.length + B.array.length); arr.set(A.array); arr.set(B.array, A.array.length);
    g.setAttribute(name, new THREE.BufferAttribute(arr, A.itemSize));
  }
  const ia = a.index.array, ib = b.index.array, off = a.attributes.position.count, idx = new Uint16Array(ia.length + ib.length);
  idx.set(ia); for (let i = 0; i < ib.length; i++) idx[ia.length + i] = ib[i] + off;
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}
