// Procedural textures for the track: a baked colour map of the whole lot (clay lane with a worn groove, loose
// banks, dry grass infield, start line), and small tiling detail maps.
import * as THREE from 'three';
import { LANE_W, BANK_RUN } from './track.js';

function hash(ix, iy) {
  let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
export function vnoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}
export function fbm(x, y, oct = 4) {
  let s = 0, a = 0.5, f = 1;
  for (let k = 0; k < oct; k++) { s += a * vnoise(x * f, y * f); a *= 0.5; f *= 2.03; }
  return s / (1 - Math.pow(0.5, oct));
}
const mix = (a, b, t) => a + (b - a) * t;
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const sstep = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };

export const PALETTE = {
  clay: [150, 106, 66], groove: [104, 72, 46], damp: [122, 84, 52], loose: [181, 142, 96],
  grass: [104, 122, 58], grassDry: [146, 140, 78], grassDark: [70, 92, 44],
};

export function bakeGround(c, maxSize = 2048) {
  const aspect = c.sizeX / c.sizeY;
  const W = aspect >= 1 ? maxSize : Math.round(maxSize * aspect), H = aspect >= 1 ? Math.round(maxSize / aspect) : maxSize;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(W, H), px = img.data;
  const half = LANE_W / 2, P = PALETTE;
  for (let py = 0; py < H; py++) {
    const y = c.y0 + (1 - (py + 0.5) / H) * c.sizeY;
    const gy = Math.min(c.ny - 1, Math.max(0, Math.round((y - c.y0) / c.cell)));
    for (let pxi = 0; pxi < W; pxi++) {
      const x = c.x0 + (pxi + 0.5) / W * c.sizeX;
      const gx = Math.min(c.nx - 1, Math.max(0, Math.round((x - c.x0) / c.cell))), q = gy * c.nx + gx;
      const s = c.sMap[q], d = c.dMap[q], ad = Math.abs(d);
      const n1 = fbm(x * 0.35, y * 0.35, 3), n2 = fbm(x * 2.2 + 31, y * 2.2 + 7, 3), n3 = vnoise(x * 11, y * 11);
      // grass
      let r = mix(P.grass[0], P.grassDry[0], n1), g = mix(P.grass[1], P.grassDry[1], n1), b = mix(P.grass[2], P.grassDry[2], n1);
      const dk = sstep(0.45, 0.75, n2) * 0.5;
      r = mix(r, P.grassDark[0], dk); g = mix(g, P.grassDark[1], dk); b = mix(b, P.grassDark[2], dk);
      const worn = sstep(0.62, 0.8, fbm(x * 0.6 + 90, y * 0.6 + 40, 3));       // bare patches
      r = mix(r, P.loose[0] * 0.9, worn * 0.7); g = mix(g, P.loose[1] * 0.9, worn * 0.7); b = mix(b, P.loose[2] * 0.9, worn * 0.7);
      if (s >= 0) {
        // dirt: lane clay with a worn groove, then loose banks fading into the grass
        const streak = vnoise(s * 0.35, d * 7.0), line = Math.exp(-(d * d) / 0.42) * (0.45 + 0.55 * streak);
        let lr = mix(P.clay[0], P.groove[0], line), lg = mix(P.clay[1], P.groove[1], line), lb = mix(P.clay[2], P.groove[2], line);
        const prof = c.profile[Math.floor(s / c.ds) % c.n];
        const wet = clamp01(prof / 0.12) * 0.55;                                   // packed, watered jump faces
        lr = mix(lr, P.damp[0], wet); lg = mix(lg, P.damp[1], wet); lb = mix(lb, P.damp[2], wet);
        const edge = sstep(half - 0.35, half + BANK_RUN * 0.6, ad);                // loose dirt toward and on the banks
        lr = mix(lr, P.loose[0], edge); lg = mix(lg, P.loose[1], edge); lb = mix(lb, P.loose[2], edge);
        const mott = 0.86 + 0.28 * n2;
        lr *= mott; lg *= mott; lb *= mott;
        // start/finish: a chequered band
        let ss = s > c.length / 2 ? s - c.length : s;
        if (Math.abs(ss) < 0.2 && ad < half) {
          const chk = (Math.floor((ss + 0.2) / 0.1) + Math.floor((d + half) / 0.1)) % 2 === 0;
          const k = 0.85; lr = mix(lr, chk ? 236 : 34, k); lg = mix(lg, chk ? 232 : 32, k); lb = mix(lb, chk ? 222 : 30, k);
        }
        const out = sstep(half + BANK_RUN + 0.25, half + BANK_RUN + 1.3, ad + (n2 - 0.5) * 0.5);
        r = mix(lr, r, out); g = mix(lg, g, out); b = mix(lb, b, out);
      }
      // soft ambient occlusion from concavity (hollows between rollers, the foot of the banks)
      const h0 = c.hf[q], k = 3;
      const hn = (c.hf[Math.min(c.ny - 1, gy + k) * c.nx + gx] + c.hf[Math.max(0, gy - k) * c.nx + gx]
        + c.hf[gy * c.nx + Math.min(c.nx - 1, gx + k)] + c.hf[gy * c.nx + Math.max(0, gx - k)]) / 4;
      const ao = Math.min(1.12, Math.max(0.62, 1 - 2.2 * (hn - h0))) * (0.93 + 0.14 * n3);
      const o = 4 * (py * W + pxi);
      px[o] = Math.min(255, r * ao); px[o + 1] = Math.min(255, g * ao); px[o + 2] = Math.min(255, b * ao); px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.generateMipmaps = true;
  return tex;
}

/** A tiling normal map from fractal noise (grit and clods). */
export function detailNormal(size = 256, strength = 2.2) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(size, size), px = img.data;
  const per = 8;                                         // noise period (tiles seamlessly)
  const tile = (x, y) => {
    let s = 0, a = 0.5, f = 1;
    for (let k = 0; k < 4; k++) {
      const p = per * f, xx = x * f, yy = y * f;
      const ix = Math.floor(xx), iy = Math.floor(yy), fx = xx - ix, fy = yy - iy;
      const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
      const h = (i, j) => hash(((i % p) + p) % p + 17 * k, ((j % p) + p) % p);
      s += a * ((h(ix, iy) * (1 - u) + h(ix + 1, iy) * u) * (1 - v) + (h(ix, iy + 1) * (1 - u) + h(ix + 1, iy + 1) * u) * v);
      a *= 0.55; f *= 2;
    }
    return s;
  };
  const hgt = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) hgt[y * size + x] = tile(x / size * per, y / size * per);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const l = hgt[y * size + (x + size - 1) % size], r = hgt[y * size + (x + 1) % size];
      const u = hgt[((y + size - 1) % size) * size + x], d = hgt[((y + 1) % size) * size + x];
      let nx = (l - r) * strength * 8, ny = (u - d) * strength * 8, nz = 1;
      const n = Math.hypot(nx, ny, nz);
      const o = 4 * (y * size + x);
      px[o] = (nx / n * 0.5 + 0.5) * 255; px[o + 1] = (ny / n * 0.5 + 0.5) * 255; px[o + 2] = (nz / n * 0.5 + 0.5) * 255; px[o + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

export function canvasTexture(w, h, draw, srgb = true) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  draw(cv.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(cv);
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}
