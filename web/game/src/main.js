// RC Dirt Circuit: glue between the menu, the race, the view and the HUD.
import loadMujoco from '@mujoco/mujoco';
import { loadLoop } from './loop.js';
import { Race, GRIPS, RIVALS, snapshot } from './race.js';
import { LoopExpert, RACER } from './driver.js';
import { View, CAMERAS } from '../../shared/render.js';
import { makeTrack } from '../../shared/track.js';
import { Input } from './input.js';
import { Sound } from './audio.js';
import * as cm from './carmodel.js';

// The shared renderer draws each car at the physics car's geometry (the full-size 1/8 truggy, size8).
const GEOM = { wheel_r: cm.WHEEL_R, corner_x: cm.CORNER.fl[0], corner_y: cm.CORNER.fl[1], mount_z: cm.MOUNT_Z, box: cm.CHASSIS_HALF };

const $ = (id) => document.getElementById(id);
const fmt = (t) => (t == null ? '–' : t >= 60 ? `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}` : t.toFixed(2));
const store = {
  get(k, d) { try { const v = localStorage.getItem('rcdc:' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('rcdc:' + k, JSON.stringify(v)); } catch { /* private window */ } },
};
const bestKey = () => `best46:${cfg.seed}:${cfg.grip}`;     // truggy physics on 4.6 loops: older bests don't carry over

// the cars' looks: yours, then the AI cars in RIVALS order
const LIVERY = [
  { color: 0xe8431a, wheelColor: 0xf3f2ee, accent: 0x15161a },
  { color: 0x1f6fd6, wheelColor: 0xe9d23a, accent: 0xc81e2b },
  { color: 0xf2b90f, wheelColor: 0x15161a, accent: 0xe8431a },
  { color: 0x7a3fd0, wheelColor: 0xf3f2ee, accent: 0x15161a },
];
const MAP_COLORS = ['#37d34b', '#e5484d', '#f2c230', '#a46cf0'];   // the cars' liveries (view/carmesh.js SCHEMES)
const GAME_CAMS = ['Chase', 'TV', 'On board', 'Heli'];      // the viewer's Orbit needs a drag, which steers here

const coarse = matchMedia('(pointer: coarse)').matches;
let gfx = store.get('gfx', coarse ? 1 : 2);
const quality = {
  pixelRatio: Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2), shadows: true,
  shadowSize: coarse ? 2048 : 4096, groundSize: coarse ? 2048 : 4096, post: gfx, grass: coarse ? 4000 : 9000,
};
const canvas = $('view');
const view = new View(canvas, quality);
view.geom = GEOM;
const input = new Input($('touch'));
input.expo = 0;                                             // the car itself has the sim's steering expo (0.4)
const sound = new Sound();
sound.setOn(store.get('sound', true));

const cfg = { seed: store.get('loop', 10001), mode: store.get('mode', 'trial'), laps: store.get('laps', 3), grip: store.get('grip', 0) };
let mj = null, index = [], course = null, circuit = null, race = null, vis = [], ghostVis = null;
let screen = 'loading';           // loading | menu | race | paused | finished
let last = performance.now(), orbit = 0, msgTimer = 0, wakeLock = null, camName = store.get('cam2', 'Chase');
const samples = [{}, {}, {}, {}];

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w && h) view.resize(w, h);
}
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(canvas);

function show(name) {
  screen = name;
  $('menu').hidden = name !== 'menu';
  $('hud').hidden = !(name === 'race' || name === 'paused' || name === 'finished');
  $('pause').hidden = name !== 'paused';
  $('finish').hidden = name !== 'finished';
  $('loading').hidden = name !== 'loading';
  $('touch').hidden = name !== 'race';
  if (name !== 'race') sound.quiet();
  if (name !== 'menu' && name !== 'loading') view.camera.clearViewOffset();
}

async function busy(text, fn) {
  $('loading-text').textContent = text;
  $('loading').hidden = false;
  await new Promise((r) => setTimeout(r, 30));
  try { await fn(); } finally { $('loading').hidden = true; }
}

// ---- menu ---------------------------------------------------------------------------------------------
function drawMap(cv, c, cars) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
  for (const p of c.pts) { lx = Math.min(lx, p[0]); hx = Math.max(hx, p[0]); ly = Math.min(ly, p[1]); hy = Math.max(hy, p[1]); }
  const pad = 9, sc = Math.min((w - 2 * pad) / (hx - lx), (h - 2 * pad) / (hy - ly));
  const X = (x) => w / 2 + (x - (lx + hx) / 2) * sc, Y = (y) => h / 2 - (y - (ly + hy) / 2) * sc;
  const css = getComputedStyle(document.documentElement);
  const line = (a, b, col, wd, cap = 'round') => {
    g.beginPath();
    for (let s = a; s <= b + 1e-6; s += 0.5) { const p = c.pointAt(s); if (s === a) g.moveTo(X(p.x), Y(p.y)); else g.lineTo(X(p.x), Y(p.y)); }
    g.strokeStyle = col; g.lineWidth = wd; g.lineCap = cap; g.lineJoin = 'round'; g.stroke();
  };
  const wd = Math.max(3, 3.0 * sc);
  line(0, c.length, css.getPropertyValue('--map-track'), wd);
  for (const j of c.jumps) line(j.start, Math.max(j.end, j.lip + 1), css.getPropertyValue('--chalk'), wd, 'butt');
  if (c.whoops && c.whoops[1] > c.whoops[0]) line(c.whoops[0], c.whoops[1], css.getPropertyValue('--map-whoops'), wd, 'butt');
  const p0 = c.pointAt(0), nx = -Math.sin(p0.heading), ny = Math.cos(p0.heading);
  g.beginPath(); g.moveTo(X(p0.x + nx * 2.4), Y(p0.y + ny * 2.4)); g.lineTo(X(p0.x - nx * 2.4), Y(p0.y - ny * 2.4));
  g.strokeStyle = css.getPropertyValue('--ink'); g.lineWidth = 2; g.lineCap = 'butt'; g.stroke();
  if (cars) for (let i = cars.length - 1; i >= 0; i--) {
    const st = cars[i].sim.st;
    g.beginPath(); g.arc(X(st.X), Y(st.Y), i ? 3 : 4.2, 0, 7);
    g.fillStyle = MAP_COLORS[i]; g.fill();
    if (!i) { g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.stroke(); }
  }
}

function entry() { return index.find((e) => e.seed === cfg.seed) ?? index[0]; }

function refreshMenu() {
  const e = entry();
  $('track-no').textContent = String(cfg.seed);
  $('track-set').textContent = e ? (e.set === 'validation' ? 'validation loop' : 'practice loop') : '';
  if (circuit && e) {
    const parts = [`${Math.round(course.length)} m lap`, `${e.jumps.length} jumps (${e.jumps.join(', ')})`];
    if (e.whoops) parts.push('whoops');
    $('track-stats').textContent = parts.join(' · ');
    const best = store.get(bestKey(), null);
    $('track-best').textContent = best ? `Your best lap: ${fmt(best)} s` : 'No lap set yet';
    drawMap($('menu-map'), circuit);
  }
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', String(b.dataset.mode === cfg.mode));
  for (const b of document.querySelectorAll('[data-laps]')) b.setAttribute('aria-pressed', String(+b.dataset.laps === cfg.laps));
  for (const b of document.querySelectorAll('[data-grip]')) b.setAttribute('aria-pressed', String(+b.dataset.grip === cfg.grip));
  for (const b of document.querySelectorAll('[data-gfx]')) b.setAttribute('aria-pressed', String(+b.dataset.gfx === gfx));
  $('laps-row').hidden = cfg.mode !== 'race';
  $('grip-note').textContent = `Tire grip ${GRIPS[cfg.grip].grip.toFixed(2)}: ${GRIPS[cfg.grip].note}.`;
  $('sound-btn').textContent = sound.on ? 'Sound on' : 'Sound off';
  $('pad-status').textContent = input.pad ? `Controller connected: ${input.pad.id.replace(/\s*\(.*\)/, '').slice(0, 40)}` : 'No controller detected. Press any button on it to connect.';
}

async function loadTrack(seed) {
  cfg.seed = seed;
  store.set('loop', seed);
  await busy(`Building loop ${seed}…`, async () => {
    if (race) { race.dispose(); race = null; }
    for (const v of [...vis, ghostVis]) if (v) view.removeCar(v);
    vis = []; ghostVis = null;
    course = await loadLoop(seed);
    course.hfield();
    circuit = makeTrack(course.renderTrack());
    view.setCircuit(circuit);
    refreshMenu();
  });
}

function stepTrack(k) {
  const i = Math.max(0, index.findIndex((e) => e.seed === cfg.seed));
  loadTrack(index[(i + k + index.length) % index.length].seed);
}

async function startRace() {
  sound.start();
  if (navigator.wakeLock && !wakeLock) navigator.wakeLock.request('screen').then((l) => { wakeLock = l; l.addEventListener('release', () => (wakeLock = null)); }).catch(() => {});
  await busy('Setting the cars on the grid…', () => {
    if (race) race.dispose();
    for (const v of [...vis, ghostVis]) if (v) view.removeCar(v);
    view.clearMarks();
    race = new Race(mj, course, { mode: cfg.mode, laps: cfg.laps, grip: GRIPS[cfg.grip].grip });
    vis = race.cars.map((_, i) => view.addCar(LIVERY[i].color, { number: i + 1, marks: true, wheelColor: LIVERY[i].wheelColor, accent: LIVERY[i].accent, geom: GEOM }));
    ghostVis = cfg.mode === 'trial' ? view.addCar(LIVERY[0].color, { ghost: true, geom: GEOM }) : null;
    if (ghostVis) view.hideCar(ghostVis);
    setCamera(camName);
    $('hud-msg').textContent = ''; $('last').textContent = '';
    $('pos-box').hidden = cfg.mode !== 'race';
    show('race');
    syncAll(0, true);
    last = performance.now();
  });
}

function message(text, secs = 1.6, cls = '') {
  const el = $('hud-msg');
  el.textContent = text; el.className = 'msg ' + cls;
  msgTimer = secs;
}

function handleEvents() {
  for (const e of race.events) {
    if (e.type === 'count') { message(String(e.n), 0.9, 'big'); sound.beep(440); }
    else if (e.type === 'go') { message('GO', 0.9, 'big'); sound.beep(880, 0.35); }
    else if (e.type === 'land') { sound.thump(Math.min(3, Math.abs(e.vz || 0))); if (e.air > 0.35) message(`${e.air.toFixed(2)} s of air`, 1.2); }
    else if (e.type === 'crash') message(e.reason === 'Reset' ? 'Marshal!' : e.reason, 1.0, 'warn');
    else if (e.type === 'reset') message('Back on track', 0.8);
    else if (e.type === 'lap') {
      const prev = store.get(bestKey(), null), record = prev == null || e.time < prev;
      if (record) store.set(bestKey(), +e.time.toFixed(3));
      $('last').textContent = `Last ${fmt(e.time)}` + (prev != null ? ` (${e.time - prev >= 0 ? '+' : '−'}${Math.abs(e.time - prev).toFixed(2)})` : '');
      message(record ? `${fmt(e.time)} · new best` : fmt(e.time), 2.2, record ? 'good' : '');
      sound.beep(record ? 990 : 660, 0.2);
    } else if (e.type === 'finish') {
      const p = race.player, ord = ['1st', '2nd', '3rd', '4th'][e.place - 1];
      $('finish-title').textContent = e.place === 1 ? 'You won' : `You finished ${ord}`;
      $('finish-body').innerHTML = '';
      p.laps.forEach((t, i) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>Lap ${i + 1}</span><b>${fmt(t)}</b>`;
        if (t === p.best) li.className = 'best';
        $('finish-body').appendChild(li);
      });
      const crashes = p.crashes.filter((r) => r !== 'Reset').length;
      $('finish-total').textContent = `Race time ${fmt(p.finished)} · ${crashes} crash${crashes === 1 ? '' : 'es'}`;
      setTimeout(() => { if (screen === 'race') show('finished'); }, 1800);
    }
  }
  race.events.length = 0;
}

function updateHud(dt) {
  const p = race.player, st = p.sim.st;
  $('lap-time').textContent = race.state === 'countdown' ? '0.00' : fmt(race.state === 'finished' ? p.laps[p.laps.length - 1] : race.lapTime());
  $('lap-no').textContent = cfg.mode === 'race' ? `${Math.min(p.lap + 1, cfg.laps)}/${cfg.laps}` : String(p.lap + 1);
  if (cfg.mode === 'race') $('pos').textContent = `${race.placeOf(p)}/${race.cars.length}`;
  $('best').textContent = `Best ${fmt(store.get(bestKey(), null))}`;
  $('speed').textContent = (Math.abs(st.vFwd) * 3.6).toFixed(0);
  $('thr-bar').style.transform = `scaleY(${Math.abs(input.throttle)})`;
  $('thr-bar').dataset.neg = String(input.throttle < 0);
  $('steer-dot').style.transform = `translateX(${-input.steer * 26}px)`;
  if (msgTimer > 0) { msgTimer -= dt; if (msgTimer <= 0) $('hud-msg').textContent = ''; }
  else if (p.offTime > 0.3) { $('hud-msg').textContent = 'Off the lane'; $('hud-msg').className = 'msg warn'; }
  else if (p.spunTime > 0.2 || (Math.abs(st.yaw) > 2.2 && st.vFwd > 1)) { $('hud-msg').textContent = 'Wrong way'; $('hud-msg').className = 'msg warn'; }
  else if ($('hud-msg').className.includes('warn')) $('hud-msg').textContent = '';
  drawMap($('hud-map'), circuit, race.cars);
  for (const side of ['steer', 'throttle']) {
    const t = input.touch[side], ring = $('ring-' + side);
    ring.hidden = !t;
    if (t) {
      ring.style.left = t.x0 + 'px'; ring.style.top = t.y0 + 'px';
      const dx = side === 'steer' ? Math.max(-70, Math.min(70, t.x - t.x0)) : 0, dy = side === 'throttle' ? Math.max(-80, Math.min(80, t.y - t.y0)) : 0;
      ring.firstElementChild.style.transform = `translate(${dx}px, ${dy}px)`;
    }
  }
}

// ---- auto quality: drop ambient occlusion, then bloom and grading, then resolution, if frames are slow ------------
const perf = { n: 0, sum: 0, settle: 0, locked: false };
function autoQuality(dt) {
  if (perf.locked) return;
  if (++perf.settle < 60) return;
  perf.n++; perf.sum += dt;
  if (perf.n < 60) return;
  const avg = perf.sum / perf.n; perf.n = 0; perf.sum = 0;
  if (avg < 1 / 40) return;
  if (gfx > 0) { gfx--; view.setQuality(gfx); }
  else if (view.renderer.getPixelRatio() > 1) { view.renderer.setPixelRatio(Math.max(1, view.renderer.getPixelRatio() - 0.25)); resize(); }
  else perf.locked = true;
}

// ---- main loop ----------------------------------------------------------------------------------------
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.max(0, Math.min(0.05, (now - last) / 1000));   // rAF timestamps can predate a blocking setup
  last = now;
  input.poll(dt);
  if (screen === 'menu' || screen === 'loading') {
    if (input.take('confirm') && screen === 'menu') startRace();
    if (circuit) menuCamera(dt);
    if (screen === 'menu' && (frame.n = (frame.n || 0) + 1) % 30 === 0) refreshMenu();
    input.pressed.clear();
    return;
  }
  if (input.take('pause')) show(screen === 'paused' ? 'race' : screen === 'race' ? 'paused' : screen);
  if (input.take('camera')) cycleCamera();
  if (screen === 'paused' || screen === 'finished') {
    if (input.take('confirm')) { if (screen === 'paused') show('race'); else startRace(); }
    if (screen === 'finished') { race.step(dt, [0, 0]); syncAll(dt); }
    view.render(dt);
    input.pressed.clear();
    return;
  }
  if (input.take('reset')) race.rescue();
  race.step(dt, playerInput());
  handleEvents();
  syncAll(dt);
  updateHud(dt);
  const st = race.player.sim.st;
  sound.update(st.wheelSpeed, input.throttle, st.speed, Math.abs(st.wheelSpeed - st.vFwd) + Math.abs(st.vLat), !race.player.sim.airborne);
  view.render(dt);
  autoQuality(dt);
  input.pressed.clear();
}

function menuCamera(dt) {
  orbit += dt * 0.12;
  const W = canvas.clientWidth, H = canvas.clientHeight, m = $('menu').getBoundingClientRect();
  const side = m.height > H * 0.9, R = Math.max(circuit.sizeX, circuit.sizeY) * 0.55;
  const k = Math.max(1, 1.45 / ((side ? W - m.width : W) / H), side ? 0 : 1.1 / Math.max(0.2, 1 - m.height / H));
  const cam = view.camera;
  cam.up.set(0, 0, 1);
  cam.position.set(circuit.centerX + k * R * Math.cos(orbit), circuit.centerY + k * R * Math.sin(orbit), circuit.plateau + k * R * 0.55);
  cam.lookAt(circuit.centerX, circuit.centerY, circuit.plateau);
  if (cam.fov !== 50) { cam.fov = 50; cam.updateProjectionMatrix(); }
  if (screen === 'menu') cam.setViewOffset(W, H, side ? -m.width / 2 : 0, side ? 0 : m.height / 2, W, H);
  view.sun.target.position.set(circuit.centerX, circuit.centerY, 0);
  view.sun.position.copy(view.sun.target.position).addScaledVector(view.sunDir, 40);
  view.stepDust(dt);
  view.render(dt);
}

// Debug/demo hooks: window.__game.autopilot = true lets the faster AI drive your car.
let auto = null;
function playerInput() {
  if (!window.__game.autopilot) return [input.steer, input.throttle];
  if (!auto || auto.c !== course) auto = new LoopExpert(course, RACER);
  return auto.act(race.player.sim);
}

function syncAll(dt, cut = false) {
  race.cars.forEach((car, i) => {
    const p = snapshot(car.sim, samples[i]), v = vis[i];
    view.poseCar(v, p, dt);
    const landed = !!car.sim.justLanded;
    car.sim.justLanded = false;
    view.dustFrom(p, landed, dt);
    view.roost(p, landed && i === 0, dt);
  });
  if (ghostVis) { const f = race.ghostFrame(); if (f) view.poseCar(ghostVis, f, dt); else view.hideCar(ghostVis); }
  view.stepDust(dt);
  view.stepClods(dt);
  view.updateCamera(samples[0], vis[0], dt, cut);
}

function setCamera(name) {
  camName = GAME_CAMS.includes(name) ? name : 'Chase';
  view.camMode = CAMERAS.indexOf(camName);
  view.camInit = false;
  store.set('cam2', camName);
  $('cam-name').textContent = camName;
}
function cycleCamera() {
  setCamera(GAME_CAMS[(GAME_CAMS.indexOf(camName) + 1) % GAME_CAMS.length]);
  if (screen === 'race') message(camName + ' camera', 1.2);
}

// ---- wiring -------------------------------------------------------------------------------------------
const on = (id, fn) => $(id).addEventListener('click', fn);
on('prev', () => stepTrack(-1));
on('next', () => stepTrack(1));
on('shuffle', () => loadTrack(index[Math.floor(Math.random() * index.length)].seed));
on('start', startRace);
on('btn-cam', cycleCamera);
on('btn-reset', () => race && race.rescue());
on('btn-pause', () => show('paused'));
on('resume', () => { show('race'); last = performance.now(); });
on('restart', startRace);
on('again', startRace);
on('pause-cam', cycleCamera);
for (const id of ['to-menu', 'to-menu2']) on(id, () => { show('menu'); refreshMenu(); });
for (const id of ['sound-btn', 'pause-sound']) on(id, () => {
  sound.setOn(!sound.on); store.set('sound', sound.on);
  $('sound-btn').textContent = $('pause-sound').textContent = sound.on ? 'Sound on' : 'Sound off';
});
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-mode],[data-laps],[data-grip],[data-gfx]');
  if (!b) return;
  if (b.dataset.mode) { cfg.mode = b.dataset.mode; store.set('mode', cfg.mode); }
  if (b.dataset.laps) { cfg.laps = +b.dataset.laps; store.set('laps', cfg.laps); }
  if (b.dataset.grip) { cfg.grip = +b.dataset.grip; store.set('grip', cfg.grip); }
  if (b.dataset.gfx) { gfx = +b.dataset.gfx; view.setQuality(gfx); perf.locked = true; store.set('gfx', gfx); }
  refreshMenu();
});
document.addEventListener('visibilitychange', () => { if (document.hidden && screen === 'race') show('paused'); });
window.addEventListener('gamepadconnected', refreshMenu);

$('rivals').innerHTML = RIVALS.map((r, i) => `<li><i style="background:${MAP_COLORS[i + 1]}"></i><b>${r.name}</b>: ${r.what}</li>`).join('');
$('pause-sound').textContent = sound.on ? 'Sound on' : 'Sound off';
setCamera(camName);
resize();
show('loading');
$('loading-text').textContent = 'Loading the physics engine…';
window.__game = {
  autopilot: false,
  /** Run the sim ahead by `secs` without rendering (testing). */
  fast(secs) { for (let t = 0; t < secs; t += 0.02) race.step(0.02, playerInput()); view.camInit = false; },
  get race() { return race; }, get circuit() { return circuit; }, get course() { return course; },
  view, input, cfg, startRace, loadTrack,
};
Promise.all([loadMujoco({ locateFile: (f) => f }), fetch('loops/index.json').then((r) => r.json())]).then(async ([m, idx]) => {
  mj = m; index = idx;
  if (!index.some((e) => e.seed === cfg.seed)) cfg.seed = index[0].seed;
  await loadTrack(cfg.seed);
  show('menu');
  refreshMenu();
  requestAnimationFrame(frame);
}).catch((err) => {
  $('loading').hidden = false;
  $('loading-text').textContent = 'Could not load: ' + (err && err.message ? err.message : err);
});
