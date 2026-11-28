// A recorded loop (sdc/race/record.py track()) as the circuit object the renderer draws: the sim's own terrain
// heights, the centreline, and per-cell track coordinates (s along the lap, d left of the centreline).
export const LANE_W = 3.0;          // loop.DRIVE_W
export const BANK_RUN = 0.4;        // loop.BANK_W
export const BANK_H = 0.35;
const CORRIDOR = LANE_W / 2 + 1.0 + 0.6;   // loop.STRIP, plus a little for the texture's fade

export function rng32(seed) {     // mulberry32 (rc-racer's circuit.js)
  let a = (seed >>> 0) || 1;
  const f = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.uniform = (lo, hi) => lo + f() * (hi - lo);
  return f;
}

export function makeTrack(T) {
  const { nx, ny, x0, y0, cell } = T.hf;
  const hf = new Float32Array(nx * ny);
  for (let q = 0; q < hf.length; q++) hf[q] = T.hf.mm[q] / 1000;
  const pts = T.center, n = pts.length, ds = T.ds, L = T.length;
  const heading = T.heading, tx = heading.map(Math.cos), ty = heading.map(Math.sin);
  // stamp each centreline point's neighbourhood; keep the nearest point per cell, then project on its tangent
  const sMap = new Float32Array(nx * ny).fill(-1), dMap = new Float32Array(nx * ny), best = new Float32Array(nx * ny).fill(1e9);
  const r = Math.ceil(CORRIDOR / cell) + 1;
  for (let i = 0; i < n; i++) {
    const cx = Math.round((pts[i][0] - x0) / cell), cy = Math.round((pts[i][1] - y0) / cell);
    for (let gy = Math.max(0, cy - r); gy <= Math.min(ny - 1, cy + r); gy++)
      for (let gx = Math.max(0, cx - r); gx <= Math.min(nx - 1, cx + r); gx++) {
        const x = x0 + gx * cell, y = y0 + gy * cell, d2 = (x - pts[i][0]) ** 2 + (y - pts[i][1]) ** 2;
        const q = gy * nx + gx;
        if (d2 < best[q]) { best[q] = d2; sMap[q] = i; }
      }
  }
  for (let q = 0; q < sMap.length; q++) {
    if (sMap[q] < 0) continue;
    const i = sMap[q], gx = q % nx, gy = (q - gx) / nx;
    const dx = x0 + gx * cell - pts[i][0], dy = y0 + gy * cell - pts[i][1];
    const lat = -dx * ty[i] + dy * tx[i];
    if (Math.abs(lat) > CORRIDOR) { sMap[q] = -1; continue; }
    sMap[q] = (((i * ds + dx * tx[i] + dy * ty[i]) % L) + L) % L;
    dMap[q] = lat;
  }
  const c = {
    seed: T.seed, length: L, ds, n, pts, tx, ty, heading, kappa: T.kappa, profile: T.profile, bank: T.bank,
    jumps: T.jumps, whoops: T.whoops, nx, ny, x0, y0, cell, hf, sMap, dMap,
    plateau: T.base + BANK_H,
    sizeX: (nx - 1) * cell, sizeY: (ny - 1) * cell,
    centerX: x0 + (nx - 1) * cell / 2, centerY: y0 + (ny - 1) * cell / 2,
    height(x, y) {
      const fx = Math.min(nx - 1.001, Math.max(0, (x - x0) / cell)), fy = Math.min(ny - 1.001, Math.max(0, (y - y0) / cell));
      const ix = Math.floor(fx), iy = Math.floor(fy), u = fx - ix, v = fy - iy, q = iy * nx + ix;
      return (hf[q] * (1 - u) + hf[q + 1] * u) * (1 - v) + (hf[q + nx] * (1 - u) + hf[q + nx + 1] * u) * v;
    },
    locate(x, y) {
      const gx = Math.round((x - x0) / cell), gy = Math.round((y - y0) / cell);
      if (gx < 0 || gy < 0 || gx >= nx || gy >= ny) return null;
      const q = gy * nx + gx;
      return sMap[q] < 0 ? null : { s: sMap[q], d: dMap[q] };
    },
    pointAt(s, d = 0) {
      const f = ((s / ds) % n + n) % n, i = Math.floor(f), u = f - i, j = (i + 1) % n;
      const x = pts[i][0] * (1 - u) + pts[j][0] * u, y = pts[i][1] * (1 - u) + pts[j][1] * u;
      return { x: x - d * ty[i], y: y + d * tx[i], heading: heading[i], kappa: T.kappa[i] };
    },
    kappaAt(s) { return T.kappa[Math.floor((((s / ds) % n) + n) % n)]; },
  };
  return c;
}

/** Decode a race recording's fixed-point columns into Float32Arrays. */
export function decodeRace(R) {
  const out = { ...R, f: {} };
  for (const [k, sc] of Object.entries(R.scales)) out.f[k] = Float32Array.from(R.cols[k], (v) => v / sc);
  delete out.cols;
  return out;
}
