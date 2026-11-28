// Phase 4.6 loop courses (sdc/jumps/loop.py), rebuilt from the arrays tools/export_loops.py writes.
// The heightfield is built exactly as LoopCourse.hfield() builds it, so the physics runs on the same ground as the
// training sim. Track coordinates: s = metres along the centreline from the start/finish line, d = metres left of it.

export const BOX = 64.0;          // m, side of the square heightfield, centred on the origin
export const D = 0.05;            // m, grid spacing
export const N = Math.round(BOX / D) + 1;
export const ZMAX = 1.5;
export const DRIVE_W = 3.0;
export const BANK_W = 0.4, BANK_H = 0.35;
export const STRIP = DRIVE_W / 2 + 1.0;
export const BASE = 0.45;
export const START_X = 1.5;       // course.START_X: cars start this far past the line

const wrap = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// numpy.rint: round half to even
const rint = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

export class LoopCourse {
  constructor(J) {
    const k = 1 / J.scale, n = this.n = J.n;
    const f = (a) => Float64Array.from(a, (v) => v * k);
    this.seed = J.seed; this.length = J.length; this.ds = J.length / n;
    this.cx = f(J.cx); this.cy = f(J.cy); this.heading = f(J.heading); this.kappa = f(J.kappa);
    this.profile = f(J.profile); this.rough = f(J.rough); this.bank = f(J.bank);
    this.jumps = J.jumps.map((j) => ({ kind: j.kind, start: j.start, lip: j.lip, height: j.height, angleDeg: j.angle_deg,
      landStart: j.land_start, end: j.end, designSpeed: j.design_speed }));
    this.whoops = J.whoops;
    this.check = J.check;
  }

  _at(arr, s) {
    const n = this.n, q = (((s % this.length) + this.length) % this.length) / this.ds;
    const i = Math.floor(q) % n, f = q - Math.floor(q);
    return arr[i] * (1 - f) + arr[(i + 1) % n] * f;
  }
  /** Ground height on the centreline at s (s may run past one lap). */
  height(s) { return BASE + this._at(this.profile, s) + this._at(this.rough, s); }
  slope(s, h = 0.1) { return (this.height(s + h) - this.height(s - h)) / (2 * h); }
  curvature(s) { return this._at(this.kappa, s); }

  /** World [X, Y, heading] at track coordinates (s, d). */
  pose(s, d = 0) {
    const i = Math.round((((s % this.length) + this.length) % this.length) / this.ds) % this.n, th = this.heading[i];
    return [this.cx[i] - d * Math.sin(th), this.cy[i] + d * Math.cos(th), th];
  }

  /** Track coordinates of a world point: { s in [0, length), d, th, i }. hint: a centreline index near the answer; the
   *  search stays within `window` m of it, so a nearby part of the loop can't be mistaken for this one. */
  project(X, Y, hint = null, window = 6.0) {
    const n = this.n;
    let best = -1, bd = Infinity;
    const test = (i) => { const dx = this.cx[i] - X, dy = this.cy[i] - Y, d2 = dx * dx + dy * dy; if (d2 < bd) { bd = d2; best = i; } };
    if (hint === null) for (let i = 0; i < n; i++) test(i);
    else { const w = Math.floor(window / this.ds); for (let k = hint - w; k <= hint + w; k++) test(((k % n) + n) % n); }
    const i = best, th = this.heading[i], tx = Math.cos(th), ty = Math.sin(th), rx = X - this.cx[i], ry = Y - this.cy[i];
    const along = rx * tx + ry * ty, d = -rx * ty + ry * tx;
    return { s: (((i * this.ds + along) % this.length) + this.length) % this.length, d, th, i };
  }

  /** Index of the jump whose stretch (1 m before its kicker to 1 m past its landing) contains s, or null. */
  jumpAt(s) {
    s = ((s % this.length) + this.length) % this.length;
    for (let i = 0; i < this.jumps.length; i++) { const j = this.jumps[i]; if (j.start - 1.0 <= s && s <= j.end + 1.0) return i; }
    return null;
  }

  surface(i, d) {
    const dc = clip(d, -DRIVE_W / 2, DRIVE_W / 2), bank = clip((Math.abs(d) - DRIVE_W / 2) / BANK_W, 0, 1) * BANK_H;
    return BASE + this.profile[i] + this.rough[i] + bank - dc * this.bank[i];
  }

  /** (N*N) elevations in [0, 1] (times ZMAX), rows along Y: LoopCourse.hfield(). */
  hfield() {
    if (this._hf) return this._hf;
    const n = this.n, z = new Float32Array(N * N).fill(NaN), sub = 4, nq = n * sub;
    const nd = Math.floor((2 * STRIP + 1e-9) / 0.025) + 1;
    // per fine sample along s: position, normal, and the two neighbouring points' indices and weight
    const I0 = new Int32Array(nq), F = new Float64Array(nq), PX = new Float64Array(nq), PY = new Float64Array(nq), NX = new Float64Array(nq), NY = new Float64Array(nq);
    for (let q = 0; q < nq; q++) {
      const qq = q / sub, i0 = Math.floor(qq) % n, i1 = (i0 + 1) % n, f = qq - Math.floor(qq);
      I0[q] = i0; F[q] = f;
      PX[q] = this.cx[i0] * (1 - f) + this.cx[i1] * f; PY[q] = this.cy[i0] * (1 - f) + this.cy[i1] * f;
      const th = this.heading[i0] + wrap(this.heading[i1] - this.heading[i0]) * f;
      NX[q] = -Math.sin(th); NY[q] = Math.cos(th);
    }
    for (let k = 0; k < nd; k++) {
      const d = -STRIP + k * 0.025;
      for (let q = 0; q < nq; q++) {
        const ix = rint((PX[q] + d * NX[q] + BOX / 2) / D), iy = rint((PY[q] + d * NY[q] + BOX / 2) / D);
        const i0 = I0[q], f = F[q];
        z[iy * N + ix] = this.surface(i0, d) * (1 - f) + this.surface((i0 + 1) % n, d) * f;
      }
    }
    // the odd cell the splat missed inside the strip: the mean of its filled neighbours (twice)
    for (let it = 0; it < 2; it++) {
      const fills = [];
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const q = y * N + x;
        if (!Number.isNaN(z[q])) continue;
        let cnt = 0, sum = 0;
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
          if (!a && !b) continue;
          const yy = y + a, xx = x + b;
          if (yy < 0 || xx < 0 || yy >= N || xx >= N) continue;
          const v = z[yy * N + xx];
          if (!Number.isNaN(v)) { cnt++; sum += v; }
        }
        if (cnt >= 5) fills.push(q, sum / cnt);
      }
      for (let k = 0; k < fills.length; k += 2) z[fills[k]] = fills[k + 1];
    }
    const out = new Float32Array(N * N);
    for (let q = 0; q < N * N; q++) out[q] = clip(Number.isNaN(z[q]) ? BASE + BANK_H : z[q], 0, ZMAX) / ZMAX;
    return (this._hf = out);
  }

  /** Terrain height at world (X, Y): bilinear on the heightfield, as MuJoCo's surface between grid points. */
  groundAt(X, Y) {
    const hf = this.hfield();
    const gx = clip((X + BOX / 2) / D, 0, N - 1.001), gy = clip((Y + BOX / 2) / D, 0, N - 1.001);
    const ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy, q = iy * N + ix;
    return ((hf[q] * (1 - fx) + hf[q + 1] * fx) * (1 - fy) + (hf[q + N] * (1 - fx) + hf[q + N + 1] * fx) * fy) * ZMAX;
  }

  /** The loop in the form the viewer's track.js makeTrack() reads (sdc/race/record.py track()): centreline every
   *  0.2 m, terrain cropped to the track and every other cell. */
  renderTrack() {
    const hf = this.hfield(), step = 2, pad = STRIP + 4.0, stride = 4;
    let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
    for (let i = 0; i < this.n; i++) { lx = Math.min(lx, this.cx[i]); hx = Math.max(hx, this.cx[i]); ly = Math.min(ly, this.cy[i]); hy = Math.max(hy, this.cy[i]); }
    const lo = (v) => Math.max(0, Math.floor((v - pad + BOX / 2) / D / step) * step);
    const hi = (v) => Math.min(Math.ceil((v + pad + BOX / 2) / D / step) * step, N - 1);
    const ix0 = lo(lx), iy0 = lo(ly), ix1 = hi(hx), iy1 = hi(hy);
    const nx = Math.floor((ix1 - ix0) / step) + 1, ny = Math.floor((iy1 - iy0) / step) + 1, mm = new Int32Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) mm[j * nx + i] = Math.round(hf[(iy0 + j * step) * N + ix0 + i * step] * ZMAX * 1000);
    const every = (a) => Array.from({ length: Math.ceil(this.n / stride) }, (_, k) => a[k * stride]);
    return {
      seed: this.seed, length: this.length, ds: D * stride, drive_w: DRIVE_W, bank_w: BANK_W, bank_h: BANK_H, base: BASE, start_x: START_X,
      center: every(this.cx).map((x, k) => [x, this.cy[k * stride]]), heading: every(this.heading), kappa: every(this.kappa),
      profile: every(this.profile), bank: every(this.bank),
      jumps: this.jumps.map((j) => ({ ...j, land_start: j.landStart, design_speed: j.designSpeed })),
      whoops: this.whoops,
      hf: { nx, ny, x0: -BOX / 2 + ix0 * D, y0: -BOX / 2 + iy0 * D, cell: D * step, mm },
    };
  }
}

/** Fetch a loop by seed (relative to the page). */
export async function loadLoop(seed, base = 'loops/') {
  const r = await fetch(`${base}${seed}.json`);
  if (!r.ok) throw new Error(`loop ${seed}: ${r.status}`);
  return new LoopCourse(await r.json());
}
