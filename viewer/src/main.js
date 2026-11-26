// Race replay: plays back a recorded 20-lap race (sdc/race/record.py) in 3D, with a ghost of the best lap so far,
// an optional rival car (another recording on the same loop), timing, slow-motion replays and an end card.
import { View, CAMERAS } from './render.js';
import { makeTrack, decodeRace } from './track.js';

const $ = (id) => document.getElementById(id);
const CAR_COLOR = 0xe8431a, RIVAL_COLOR = 0x1f6fd6;
const SPEEDS = [1, 2, 4, 8];

// recordings ship as gzipped JSON in base64 text (artifacts serve text, not binary archives)
async function gz(url) {
  const r = await fetch(`${url}.txt`);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const bin = atob((await r.text()).trim()), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(ds).text());
}

const fmt = (t) => { const m = Math.floor(t / 60), s = t - 60 * m; return `${m}:${s.toFixed(1).padStart(4, '0')}`; };
const fmtLap = (t) => (t >= 60 ? fmt(t) : t.toFixed(2));
const sgn = (x, d = 2) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}`;

// ---- a recorded race, prepared for playback -----------------------------------------------------------------
function prepRace(raw, c) {
  const R = decodeRace(raw), f = R.f, n = R.n, fps = R.fps, L = c.length;
  const laps = [0];
  for (const e of R.events) if (e.type === 'lap') laps.push(e.t);
  R.lapStarts = laps;                         // race time at the start of each lap (and the finish)
  R.lapOf = (t) => { let k = 0; while (k < laps.length - 2 && t >= laps[k + 1]) k++; return k; };
  // per frame: lap position u (metres from the line within its lap), running max for time lookups
  const u = new Float32Array(n), um = new Float32Array(n), prog = new Float32Array(n);
  let sPrev = 0;
  for (let i = 0; i < n; i++) {
    const t = i / fps, k = R.lapOf(t), q = c.locate(f.x[i], f.y[i]);
    let s = q ? q.s : sPrev; sPrev = s;
    if (s > L - 4 && t - laps[k] < 1.5) s -= L;
    if (s < 4 && k + 1 < laps.length && laps[k + 1] - t < 1.5) s += L;
    u[i] = s;
    um[i] = i > 0 && R.lapOf((i - 1) / fps) === k ? Math.max(um[i - 1], s) : s;
    prog[i] = k * L + um[i];
  }
  for (let i = 1; i < n; i++) prog[i] = Math.max(prog[i], prog[i - 1]);
  R.u = u; R.um = um; R.prog = prog;
  // the best completed lap before each lap
  R.bestBefore = [];
  let best = -1;
  for (let k = 0; k < R.lap_times.length; k++) {
    R.bestBefore.push(best);
    if (best < 0 || R.lap_times[k] < R.lap_times[best]) best = k;
  }
  R.bestLap = best;
  R.crashes_ = R.events.filter((e) => e.type === 'crash');
  R.jumps_ = R.events.filter((e) => e.type === 'jump');
  R.crashLaps = new Set(R.crashes_.map((e) => e.lap - 1));
  return R;
}

/** Interpolated sample of race R at time t (seconds of race time). */
function sample(R, t, out = {}) {
  const f = R.f, x = Math.min(Math.max(t * R.fps, 0), R.n - 1.0001), i = Math.floor(x), a = x - i, j = i + 1;
  for (const k of ['x', 'y', 'z', 's1', 's2', 's3', 's4', 'st', 'v', 'cs', 'ct']) out[k] = f[k][i] * (1 - a) + f[k][j] * a;
  let qw = f.qw[j], qx = f.qx[j], qy = f.qy[j], qz = f.qz[j];
  if (qw * f.qw[i] + qx * f.qx[i] + qy * f.qy[i] + qz * f.qz[i] < 0) { qw = -qw; qx = -qx; qy = -qy; qz = -qz; }
  out.qw = f.qw[i] * (1 - a) + qw * a; out.qx = f.qx[i] * (1 - a) + qx * a; out.qy = f.qy[i] * (1 - a) + qy * a; out.qz = f.qz[i] * (1 - a) + qz * a;
  const nq = Math.hypot(out.qw, out.qx, out.qy, out.qz); out.qw /= nq; out.qx /= nq; out.qy /= nq; out.qz /= nq;
  out.air = f.air[a < 0.5 ? i : j]; out.gnd = f.gnd[a < 0.5 ? i : j];
  out.yaw = Math.atan2(2 * (out.qw * out.qz + out.qx * out.qy), 1 - 2 * (out.qy * out.qy + out.qz * out.qz));
  const i0 = Math.max(0, i - 1), i1 = Math.min(R.n - 1, i + 2), dt = (i1 - i0) / R.fps;
  out.vx = (f.x[i1] - f.x[i0]) / dt; out.vy = (f.y[i1] - f.y[i0]) / dt;
  out.u = Math.abs(R.u[j] - R.u[i]) > 5 ? (a < 0.5 ? R.u[i] : R.u[j]) : R.u[i] * (1 - a) + R.u[j] * a;   // no blend across the line
  out.um = R.um[i]; out.prog = R.prog[i];
  out.s = ((out.u % R.L) + R.L) % R.L;
  return out;
}

/** Race time at which race R first reached progress P (metres since the gun), or null. */
function timeAtProg(R, P) {
  const p = R.prog;
  if (P > p[R.n - 1]) return null;
  let lo = 0, hi = R.n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (p[m] >= P) hi = m; else lo = m; }
  const a = p[hi] > p[lo] ? (P - p[lo]) / (p[hi] - p[lo]) : 0;
  return (lo + a) / R.fps;
}

/** Time into lap k at which the car first reached lap position u. */
function lapTimeAt(R, k, u) {
  const i0 = Math.ceil(R.lapStarts[k] * R.fps + 1e-6), i1 = Math.min(R.n - 1, Math.floor(R.lapStarts[k + 1] * R.fps - 1e-6));
  let lo = i0, hi = i1;
  if (u <= R.um[lo]) return 0;
  if (u > R.um[hi]) return null;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (R.um[m] >= u) hi = m; else lo = m; }
  const a = R.um[hi] > R.um[lo] ? (u - R.um[lo]) / (R.um[hi] - R.um[lo]) : 0;
  return (lo + a) / R.fps - R.lapStarts[k];
}

// ---- app ----------------------------------------------------------------------------------------------------
const canvas = $('view');
const coarse = matchMedia('(pointer: coarse)').matches;
let gfx = (() => { try { return +(localStorage.getItem('race-gfx') ?? (coarse ? 1 : 2)); } catch { return coarse ? 1 : 2; } })();
const quality = { pixelRatio: Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2), shadows: true, shadowSize: coarse ? 2048 : 4096, groundSize: coarse ? 2048 : 4096, post: gfx, grass: coarse ? 4000 : 9000 };
const view = new View(canvas, quality);
view.camMode = 0;

const S = {
  man: null, loopIdx: 0, c: null, R: null, rival: null, rivalName: null,
  t: 0, playing: true, speed: 1, ghostOn: true, replaysOn: true,
  car: null, ghost: null, rivalCar: null, replay: null, recordAir: 0, replayedLaps: new Set(), replayedCrashes: new Set(),
  lastAir: null, flashUntil: 0, ended: false, wasAir: false,
};
const P = {}, G = {}, Q = {};

function resize() { const r = canvas.getBoundingClientRect(); view.resize(Math.max(1, r.width), Math.max(1, r.height)); }
addEventListener('resize', resize);

async function loadLoop(idx) {
  S.loopIdx = idx;
  const lp = S.man.loops[idx];
  $('loading').hidden = false; $('loading-msg').textContent = `Building loop ${lp.seed}…`;
  $('card').hidden = true;
  const [T, raw, rawRival] = await Promise.all([gz(`data/${lp.track}`), gz(`data/${lp.races[S.man.main]}`),
    S.rivalName && lp.races[S.rivalName] ? gz(`data/${lp.races[S.rivalName]}`) : null]);
  await new Promise((r) => setTimeout(r, 30));
  const c = makeTrack(T);
  view.setCircuit(c);
  S.c = c;
  S.R = prepRace(raw, c); S.R.L = c.length;
  S.rival = rawRival ? prepRace(rawRival, c) : null; if (S.rival) S.rival.L = c.length;
  for (const k of ['car', 'ghost', 'rivalCar']) if (S[k]) view.removeCar(S[k]);
  S.car = view.addCar(CAR_COLOR, { number: 1, marks: true, wheelColor: 0xf3f2ee, accent: 0x15161a });
  S.ghost = view.addCar(CAR_COLOR, { ghost: true });
  S.rivalCar = S.rival ? view.addCar(RIVAL_COLOR, { number: 2, marks: true, wheelColor: 0xe9d23a, accent: 0xc81e2b }) : null;
  S.t = 0; S.replay = null; S.recordAir = 0; S.replayedLaps.clear(); S.replayedCrashes.clear(); S.ended = false;
  view.camInit = false;
  buildTimeline();
  buildSheet();
  $('car-tag').textContent = S.man.cars[S.man.main].label;
  $('rival-box').hidden = !S.rival;
  if (S.rival) $('rival-name').textContent = S.man.cars[S.rivalName].short;
  $('loading').hidden = true; $('hud').hidden = false; $('bar').hidden = false;
  resize();
}

function buildTimeline() {
  const R = S.R, T = R.race_time, tr = document.querySelector('#tl .track');
  tr.querySelectorAll('.tick,.crash,.head,.lbl').forEach((e) => e.remove());
  R.lapStarts.slice(1, -1).forEach((t, k) => {
    const d = document.createElement('div'); d.className = 'tick' + (k + 1 === R.bestLap ? ' best' : ''); d.style.left = `${(100 * t) / T}%`; tr.append(d);
    if ((k + 1) % 5 === 0) { const l = document.createElement('div'); l.className = 'lbl'; l.textContent = `L${k + 2}`; l.style.left = `${(100 * t) / T}%`; tr.append(l); }
  });
  for (const e of R.crashes_) { const d = document.createElement('div'); d.className = 'crash'; d.style.left = `${(100 * e.t) / T}%`; tr.append(d); }
  const h = document.createElement('div'); h.className = 'head'; tr.append(h);
  $('scrub').max = T;
}

function seg(el, items, cur, onPick) {
  el.innerHTML = '';
  for (const it of items) {
    const b = document.createElement('button');
    b.innerHTML = it.html; b.setAttribute('aria-pressed', String(it.v === cur));
    b.addEventListener('click', () => { el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); onPick(it.v); });
    el.append(b);
  }
}

function buildSheet() {
  const m = S.man, car = m.cars[m.main];
  $('sheet-sub').textContent = `${car.label}: ${car.what} ${m.laps} laps per loop; each crash costs ${m.crash_s} s.`;
  seg($('loops'), m.loops.map((lp, i) => ({ v: i, html: `Loop ${'ABCDEFG'[i]}<small>${lp.length.toFixed(0)} m · ${lp.jumps} jumps</small>` })), S.loopIdx,
    (i) => { $('sheet').hidden = true; loadLoop(i); });
  const rivals = m.rivals.filter((r) => r !== m.main);
  seg($('rival-seg'), [...rivals.map((r) => ({ v: r, html: m.cars[r].short })), { v: null, html: 'Off' }], S.rivalName,
    (v) => { const keep = S.t; S.rivalName = v; $('sheet').hidden = true; loadLoop(S.loopIdx).then(() => { S.t = keep; }); });
  $('how').innerHTML = m.how;
}

// ---- controls -------------------------------------------------------------------------------------------------
$('play').addEventListener('click', () => setPlaying(!S.playing));
function setPlaying(p) {
  S.playing = p;
  $('play').textContent = p ? '❚❚' : '▶';
  $('play').setAttribute('aria-label', p ? 'Pause' : 'Play');
}
$('speedbtn').addEventListener('click', () => { S.speed = SPEEDS[(SPEEDS.indexOf(S.speed) + 1) % SPEEDS.length]; $('speedbtn').textContent = `${S.speed}×`; });
$('cambtn').addEventListener('click', () => { view.camMode = (view.camMode + 1) % CAMERAS.length; $('cambtn').textContent = CAMERAS[view.camMode]; view.camInit = false; });
$('scrub').addEventListener('input', (e) => { S.t = +e.target.value; view.clearMarks(); S.replay = null; S.ended = false; $('card').hidden = true; view.camInit = false; });
// skip: ends a replay (and a crash's countdown with it), or jumps to the end of the marshal's countdown
$('skip').addEventListener('click', () => {
  const R = S.R;
  if (S.replay) { S.t = S.replay.resume; S.replay = null; }
  const cr = R.crashes_.find((e) => S.t >= e.t && S.t < e.until);
  if (cr) { S.t = cr.until; S.replayedCrashes.add(cr.t); }
  view.camInit = false;
  view.clearMarks();
});
$('menubtn').addEventListener('click', () => { $('sheet').hidden = !$('sheet').hidden; });
$('close-sheet').addEventListener('click', () => { $('sheet').hidden = true; });
seg($('ghost-seg'), [{ v: true, html: 'Best lap' }, { v: false, html: 'Off' }], true, (v) => { S.ghostOn = v; });
seg($('gfx-seg'), [{ v: 2, html: 'High' }, { v: 1, html: 'Medium' }, { v: 0, html: 'Fast' }], gfx, (v) => {
  gfx = v; view.setQuality(v); perf.locked = true; try { localStorage.setItem('race-gfx', String(v)); } catch { /* private mode */ }
});
$('gfx-seg').querySelectorAll('button').forEach((b, i) => (b.dataset.v = String(2 - i)));
seg($('replay-seg'), [{ v: true, html: 'Jumps &amp; crashes' }, { v: false, html: 'Off' }], true, (v) => { S.replaysOn = v; if (!v) S.replay = null; });
$('again').addEventListener('click', () => { S.t = 0; view.clearMarks(); S.ended = false; S.replayedLaps.clear(); S.replayedCrashes.clear(); S.recordAir = 0; $('card').hidden = true; setPlaying(true); view.camInit = false; });
$('next').addEventListener('click', () => { loadLoop((S.loopIdx + 1) % S.man.loops.length).then(() => setPlaying(true)); });
$('card-close').addEventListener('click', () => { $('card').hidden = true; });
addEventListener('keydown', (e) => {
  if (e.key === ' ') { e.preventDefault(); setPlaying(!S.playing); }
  if (e.key === 'c') $('cambtn').click();
  if (e.key === 'ArrowRight') { S.t += 5; view.clearMarks(); }
  if (e.key === 'ArrowLeft') { S.t = Math.max(0, S.t - 5); view.clearMarks(); }
});
// orbit: drag to look around, pinch or wheel to zoom
const ptrs = new Map();
canvas.addEventListener('pointerdown', (e) => {
  if (CAMERAS[view.camMode] !== 'Orbit') { view.camMode = CAMERAS.indexOf('Orbit'); $('cambtn').textContent = 'Orbit'; }
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); canvas.setPointerCapture(e.pointerId); view.dragging = true;
});
canvas.addEventListener('pointermove', (e) => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const o = view.orbit;
  if (ptrs.size === 1) { o.yaw -= (e.clientX - p.x) * 0.008; o.pitch = Math.min(1.45, Math.max(0.03, o.pitch + (e.clientY - p.y) * 0.006)); }
  else {
    const [a, b] = [...ptrs.values()], d0 = Math.hypot(a.x - b.x, a.y - b.y);
    p.x = e.clientX; p.y = e.clientY;
    const [a2, b2] = [...ptrs.values()], d1 = Math.hypot(a2.x - b2.x, a2.y - b2.y);
    o.dist = Math.min(12, Math.max(0.7, o.dist * d0 / Math.max(d1, 1)));
  }
  p.x = e.clientX; p.y = e.clientY;
});
const up = (e) => { ptrs.delete(e.pointerId); if (!ptrs.size) view.dragging = false; };
canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
canvas.addEventListener('wheel', (e) => { e.preventDefault(); view.orbit.dist = Math.min(12, Math.max(0.7, view.orbit.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });

// ---- per frame ------------------------------------------------------------------------------------------------
function triggerReplays(t0, t1) {
  if (!S.replaysOn || S.speed > 2 || S.replay) return;
  const R = S.R;
  for (const e of R.crashes_) {
    const at = e.t + 2.6;
    if (at > t0 && at <= t1 && !S.replayedCrashes.has(e.t)) {
      S.replayedCrashes.add(e.t);
      S.replay = { t: e.t - 2.4, from: e.t - 2.4, to: e.t + 1.6, rate: 0.35, resume: at, label: 'Crash replay' };
      return;
    }
  }
  for (const e of R.jumps_) {
    const at = e.t + e.air + 0.8;
    if (at > t0 && at <= t1 && e.air > S.recordAir && e.air >= 0.5 && !S.replayedLaps.has(e.lap)) {
      S.recordAir = e.air; S.replayedLaps.add(e.lap);
      S.replay = { t: e.t - 1.1, from: e.t - 1.1, to: e.t + e.air + 0.7, rate: 0.3, resume: at, label: 'Longest air so far' };
      return;
    }
    if (at > t0 && at <= t1) S.recordAir = Math.max(S.recordAir, e.air);
  }
}

function hud(p, t) {
  const R = S.R, k = R.lapOf(t), nl = R.lap_times.length;
  const finished = t >= R.race_time;
  $('lap').innerHTML = `${Math.min(k + 1, nl)}<b>/${nl}</b>`;
  const lapT = finished ? R.lap_times[nl - 1] : t - R.lapStarts[k];
  $('laptime').textContent = fmtLap(lapT);
  // gap to the best lap so far, at the same place on the lap
  const bb = R.bestBefore[k], dEl = $('delta');
  if (bb >= 0 && !finished) {
    const gt = lapTimeAt(R, bb, p.um);
    if (gt != null && p.um > 3) { const d = lapT - gt; dEl.textContent = `${sgn(d)} best`; dEl.className = 'num ' + (d <= 0 ? 'ahead' : 'behind'); }
    else { dEl.textContent = `best ${R.lap_times[bb].toFixed(2)}`; dEl.className = 'num'; }
  } else dEl.textContent = '';
  $('racetime').textContent = fmt(Math.min(t, R.race_time));
  const crashed = R.crashes_.filter((e) => e.t <= t).length;
  $('crashes').textContent = crashed ? `${crashed}  +${crashed * R.crash_s}s` : '0';
  $('speed').innerHTML = `${Math.abs(p.v).toFixed(1)}<small>m/s</small>`;
  const thr = $('thr').firstElementChild, st = $('steer').firstElementChild;
  thr.className = p.ct < 0 ? 'brake' : '';
  thr.style.left = `${50 + Math.min(0, p.ct) * 50}%`; thr.style.width = `${Math.abs(p.ct) * 50}%`;
  st.style.left = `${50 - Math.max(0, p.cs) * 50}%`; st.style.width = `${Math.abs(p.cs) * 50}%`;   // left steer shows on the left
  // airtime: counts up in the air, holds after landing
  const air = $('air');
  const jumpNow = R.jumps_.find((e) => t >= e.t && t <= e.t + e.air + 1.4);
  if (jumpNow) {
    const a = Math.min(t - e_t(jumpNow), jumpNow.air);
    air.innerHTML = `${a.toFixed(2)} s<small>air${jumpNow.air >= S.recordAir && t > jumpNow.t + jumpNow.air ? ' · longest yet' : ''}</small>`;
    air.classList.add('show');
  } else air.classList.remove('show');
  // crash flag and the marshal's countdown
  const cr = R.crashes_.find((e) => t >= e.t && t < e.until);
  $('flag').hidden = !cr || !!S.replay;
  const sk = $('skip');
  sk.hidden = !(S.replay || cr);
  if (!sk.hidden) sk.textContent = S.replay ? 'Skip replay ›' : 'Skip countdown ›';
  if (cr) {
    $('flag-h').textContent = { 'rolled over': 'Rolled over', 'off the lane': 'Off the lane', stuck: 'Stuck', 'spun round': 'Spun round' }[cr.reason] ?? cr.reason;
    $('flag-why').textContent = t - cr.t < 2 ? `Crash ${R.crashes_.indexOf(cr) + 1} · +${R.crash_s} s` : 'Marshal on the way';
    $('flag-n').textContent = Math.ceil(cr.until - t);
  }
  // lap flash for a moment after each line crossing
  const lf = $('lapflash');
  const justDone = k > 0 && t - R.lapStarts[k] < 2.5 && !finished && !S.replay;
  if (justDone) {
    const lt = R.lap_times[k - 1], isBest = R.bestBefore[k] === k - 1;
    lf.innerHTML = `<small>Lap ${k}${isBest ? ' · best lap' : ''}</small>${lt.toFixed(2)} s`;
    lf.className = 'panel' + (isBest ? ' best' : '');
  }
  lf.hidden = !justDone;
  // rival gap: who reached this point on the race first, and by how long
  if (S.rival) {
    const g = $('rival-gap'), tr = timeAtProg(S.rival, p.prog);
    let gap;
    if (tr != null) gap = t - tr;
    else { const rp = sample(S.rival, t, Q).prog, tm = timeAtProg(R, rp); gap = tm != null ? -(t - tm) : 0; }
    g.textContent = Math.abs(gap) < 0.05 ? 'level' : `${Math.abs(gap).toFixed(1)} s ${gap > 0 ? 'behind' : 'ahead'}`;
    g.style.color = gap > 0.05 ? 'var(--warn)' : 'var(--good)';
  }
  // timeline
  const T = R.race_time, pct = `${(100 * Math.min(t, T)) / T}%`;
  document.querySelector('#tl .fill').style.width = pct;
  document.querySelector('#tl .head').style.left = pct;
  if (document.activeElement !== $('scrub')) $('scrub').value = t;
}
const e_t = (e) => e.t;

function endCard() {
  const R = S.R, m = S.man, lp = m.loops[S.loopIdx], car = m.cars[m.main];
  $('card-eyebrow').textContent = `Loop ${'ABCDEFG'[S.loopIdx]} · ${lp.length.toFixed(0)} m · ${R.lap_times.length} laps`;
  const flying = R.lap_times.slice(1).filter((_, i) => !R.crashLaps.has(i + 1));
  const stats = [
    ['Race time', fmt(R.race_time), `${R.crashes} crash${R.crashes === 1 ? '' : 'es'} × ${R.crash_s} s included`, ''],
    ['Best lap', R.lap_times[R.bestLap].toFixed(2), `lap ${R.bestLap + 1}`, 'good'],
    ['Crashes', String(R.crashes), R.crashes ? R.crashes_.map((e) => e.reason).join(', ') : 'a clean race', R.crashes ? 'warn' : 'good'],
    ['Clean laps', flying.length ? (flying.reduce((a, b) => a + b, 0) / flying.length).toFixed(2) : '–', 'mean, laps 2+ without a crash', ''],
  ];
  const longest = R.jumps_.reduce((a, e) => (e.air > a.air ? e : a), { air: 0 });
  stats.push(['Longest air', `${longest.air.toFixed(2)} s`, longest.lap ? `lap ${longest.lap}, jump ${longest.jump + 1}` : '', '']);
  if (S.rival) {
    const d = R.race_time - S.rival.race_time;
    stats.push([`vs ${m.cars[S.rivalName].short}`, `${sgn(d, 1)} s`, `${m.cars[S.rivalName].short}: ${fmt(S.rival.race_time)}, ${S.rival.crashes} crash${S.rival.crashes === 1 ? '' : 'es'}`, d <= 0 ? 'good' : 'warn']);
  }
  if (!S.rival) stats.push(['Top speed', `${Math.max(...R.f.v).toFixed(1)} m/s`, `${(Math.max(...R.f.v) * 3.6).toFixed(0)} km/h`, '']);
  $('stats').innerHTML = stats.map(([l, v, s, c]) => `<div class="stat"><span class="label">${l}</span><span class="num ${c}">${v}</span><small>${s}</small></div>`).join('');
  $('card-title').innerHTML = `${car.label} <span>${fmt(R.race_time)}</span>`;
  drawChart();
  $('card').hidden = false;
}

function drawChart() {
  const R = S.R, rv = S.rival, svg = $('chart'), W = 640, H = 300, l = 58, r = 14, t0 = 30, b = 40;
  const clean = R.lap_times.filter((_, k) => !R.crashLaps.has(k));
  const rClean = rv ? rv.lap_times.filter((_, k) => !rv.crashLaps.has(k)) : [];
  const all = [...clean, ...rClean];
  const lo = Math.floor(Math.min(...all) - 1), hi = Math.ceil(Math.max(...all) + 1), n = R.lap_times.length;
  const X = (k) => l + (k + 0.5) * (W - l - r) / n, Y = (v) => t0 + (hi - Math.min(v, hi)) / (hi - lo) * (H - t0 - b);
  const step = hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1;
  let s = '';
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step)
    s += `<line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#3d4a40" stroke-width="1"/><text x="${l - 8}" y="${Y(v) + 6}" text-anchor="end">${v} s</text>`;
  for (let k = 0; k < n; k++) if (k % (n > 10 ? 2 : 1) === 0 || k === n - 1) s += `<text x="${X(k)}" y="${H - 12}" text-anchor="middle">${k + 1}</text>`;
  const bw = Math.max(5, (W - l - r) / n * 0.6);
  R.lap_times.forEach((v, k) => {
    const crash = R.crashLaps.has(k), col = k === R.bestLap ? '#86d993' : crash ? '#ff7a4d' : '#f2c230';
    s += `<rect x="${X(k) - bw / 2}" y="${Y(v)}" width="${bw}" height="${H - b - Y(v)}" fill="${col}"/>`;
    if (v > hi) s += `<text x="${X(k)}" y="${t0 - 8}" text-anchor="middle" style="fill:#ff7a4d">${Math.round(v)}</text>`;
  });
  if (rv) {
    const pts = rv.lap_times.map((v, k) => `${X(k)},${Y(v)}`).join(' ');
    s += `<polyline points="${pts}" fill="none" stroke="#e5484d" stroke-width="3"/>`;
    rv.lap_times.forEach((v, k) => { s += `<circle cx="${X(k)}" cy="${Y(v)}" r="4.5" fill="#e5484d"/>`; });
  }
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = s;
  $('legend').innerHTML = `<span><i style="background:#f2c230"></i>Lap time</span><span><i style="background:#86d993"></i>Best lap</span><span><i style="background:#ff7a4d"></i>Lap with a crash (+${R.crash_s} s; time on top if off the scale)</span>`
    + (rv ? `<span><i style="background:#e5484d"></i>${S.man.cars[S.rivalName].short}</span>` : '') + '<span>Lap 1 starts from standstill.</span>';
}

// auto quality: if frames are slow, drop ambient occlusion, then bloom and grading, then resolution
const perf = { n: 0, sum: 0, settle: 0 };
function autoQuality(dt) {
  if (perf.locked) return;
  if (++perf.settle < 40) return;                   // let the shaders compile and the first loop build
  perf.n++; perf.sum += dt;
  if (perf.n < 60) return;
  const avg = perf.sum / perf.n; perf.n = 0; perf.sum = 0;
  if (avg < 1 / 40) return;
  if (gfx > 0) { gfx--; view.setQuality(gfx); markGfx(); }
  else if (view.renderer.getPixelRatio() > 1) { view.renderer.setPixelRatio(Math.max(1, view.renderer.getPixelRatio() - 0.25)); resize(); }
  else perf.locked = true;
}
function markGfx() { $('gfx-seg')?.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.v === gfx))); }

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dtw = Math.min(0.1, (now - last) / 1000); last = now;
  if (!S.R) return;
  const R = S.R;
  let cut = false;
  if (S.replay) {
    S.replay.t += S.playing ? dtw * S.replay.rate : 0;
    if (S.replay.t >= S.replay.to) { S.t = S.replay.resume; S.replay = null; cut = true; view.camInit = false; }
  } else if (S.playing && !S.ended) {
    const t0 = S.t;
    S.t = Math.min(R.race_time, S.t + dtw * S.speed);
    triggerReplays(t0, S.t);
    if (S.replay) { cut = true; view.camInit = false; }
    if (S.t >= R.race_time && !S.ended) { S.ended = true; endCard(); }
  }
  const t = S.replay ? S.replay.t : S.t, dtv = S.replay ? dtw * S.replay.rate : dtw * S.speed;
  const p = sample(R, t, P);
  view.poseCar(S.car, p, S.playing ? dtv : 0);
  const air = p.air > 0.5;
  const landed = S.wasAir && !air && S.playing;
  view.dustFrom(p, landed, dtv);
  if (S.playing) view.roost(p, landed, dtv);
  S.wasAir = air;
  // ghost: the best completed lap, at the same time into its lap
  const k = R.lapOf(t), bb = R.bestBefore[k];
  if (S.ghostOn && bb >= 0 && !S.replay && t < R.race_time) {
    const gt = R.lapStarts[bb] + (t - R.lapStarts[k]);
    if (gt < R.lapStarts[bb + 1]) view.poseCar(S.ghost, sample(R, gt, G), S.playing ? dtv : 0); else view.hideCar(S.ghost);
  } else view.hideCar(S.ghost);
  if (S.rival) {
    if (t <= S.rival.race_time + 3) view.poseCar(S.rivalCar, sample(S.rival, t, Q), S.playing ? dtv : 0);
    else view.hideCar(S.rivalCar);
  }
  const saved = view.camMode;
  if (S.replay) view.camMode = CAMERAS.indexOf('TV');
  view.updateCamera(p, S.car, dtw, cut);
  view.camMode = saved;
  $('replay-tag').hidden = !S.replay;
  if (S.replay) $('replay-tag').textContent = S.replay.label;
  hud(p, t);
  const dtp = S.playing ? (S.replay ? dtw * S.replay.rate : dtw * Math.min(S.speed, 2)) : 0;
  view.stepDust(dtp);
  view.stepClods(dtp);
  view.render(dtw, !!S.replay, S.car.body.position);
  autoQuality(dtw);
}

async function main() {
  try {
    S.man = await (await fetch('data/manifest.json')).json();
    S.rivalName = S.man.rivals.find((r) => r !== S.man.main) ?? null;
    await loadLoop(0);
    requestAnimationFrame(frame);
  } catch (err) {
    $('loading-msg').textContent = `Could not load the race: ${err.message}`;
    console.error(err);
  }
}
window.__S = S; window.__view = view;
main();
