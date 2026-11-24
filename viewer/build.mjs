// Build the race replay into dist/: race.html (the page, with the bundled viewer inline; three.js loads from a
// CDN through an import map) and data/ (manifest.json plus the gzipped recordings from sdc/race/record.py).
//
//   node build.mjs [--data ../runs/race] [--main e5] [--serve]
import { build } from 'esbuild';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import http from 'http';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const DATA = arg('--data', '../runs/race');
const THREE_VER = '0.180.0';

// what each recorded car is (keys: the recording's name)
const CARS = {
  teacher: { label: 'Teacher', short: 'Teacher', what: 'the rule-based driver the sensor cars learn from. It reads the sim\'s true state (its place on the track, speed, attitude), which a real car can\'t.' },
  base: { label: 'Base sensor car', short: 'Base', what: 'Phase 4.6\'s sensor-only students (two GRU networks, averaged), driving from the car\'s own sensors.' },
  e4: { label: 'E4 sensor car', short: 'E4', what: 'Phase 4.7 E4\'s sensor-only students (two GRU networks, averaged), copying the re-tuned teacher.' },
  e5: { label: 'E5 sensor car', short: 'E5', what: 'Phase 4.7 E5\'s sensor-only students (two GRU networks, averaged), copying the privileged RL racer.' },
};

const files = fs.readdirSync(DATA).filter((f) => f.endsWith('.json.gz'));
const read = (f) => JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(DATA, f))));
const loops = {};
for (const f of files) {
  const m = f.match(/^(.+)-(\d+)\.json\.gz$/);
  if (!m) continue;
  const [, name, seed] = m;
  loops[seed] ??= { seed: +seed, races: {} };
  if (name === 'track') { const T = read(f); loops[seed].track = f; loops[seed].length = T.length; loops[seed].jumps = T.jumps.length; }
  else loops[seed].races[name] = f;
}
const list = Object.values(loops).filter((l) => l.track && Object.keys(l.races).length).sort((a, b) => a.seed - b.seed);
const names = [...new Set(list.flatMap((l) => Object.keys(l.races)))];
const main = arg('--main', names.find((n) => n !== 'teacher') ?? names[0]);
const sample = read(list[0].races[main]);
const manifest = {
  main, rivals: names.filter((n) => n !== main && list.every((l) => l.races[n])), laps: sample.laps, crash_s: sample.crash_s,
  cars: Object.fromEntries(names.map((n) => [n, CARS[n] ?? { label: n, short: n, what: 'a recorded car.' }])),
  loops: list,
  how: `<b>Recorded, not re-simulated.</b> Each race ran in the project's MuJoCo sim (sdc/race/record.py) with the current car: truggy physics, steering expo 0.4, and one physics draw from the gate range kept for the whole race. Poses, suspension, steering and controls were saved 25 times a second, and this page plays them back. <b>Scoring:</b> total time for ${sample.laps} laps, plus ${sample.crash_s} s for each crash: rolled over, off the lane, stuck, or spun round (facing backwards for half a second). After a crash the car sits out the penalty, then a marshal puts it back on the centre line at the furthest point it reached (or ${9} m before the kicker if it crashed on a jump), and its memory restarts. <b>The ghost</b> replays the fastest lap so far. <b>The rival</b> is a separate recording on the same loop. The two cars never touch. The loops are validation loops ${list.map((l) => l.seed).join(', ')}. The track, terrain and car look come from the rc-racer game.`,
};

fs.mkdirSync('dist/data', { recursive: true });
fs.rmSync('dist/data', { recursive: true, force: true }); fs.mkdirSync('dist/data', { recursive: true });
for (const l of list) for (const f of [l.track, ...Object.values(l.races)])   // base64 text: artifacts serve text, not .gz
  fs.writeFileSync(path.join('dist/data', `${f}.txt`), fs.readFileSync(path.join(DATA, f)).toString('base64'));
fs.writeFileSync('dist/data/manifest.json', JSON.stringify(manifest, null, 1));

const out = await build({ entryPoints: ['src/main.js'], bundle: true, format: 'esm', write: false, minify: true, target: 'es2020', external: ['three'], logLevel: 'warning' });
const js = out.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const page = fs.readFileSync('src/page.html', 'utf8');
fs.writeFileSync('dist/race.html', `${page}
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@${THREE_VER}/build/three.module.js"}}</script>
<script type="module">${js}</script>
`);
console.log(`dist/race.html (${(js.length / 1024).toFixed(0)} KB js), ${list.length} loops, cars ${names.join(', ')}, main ${main}`);

if (process.argv.includes('--serve')) {   // a static server that wraps race.html in a document, as the artifact host does
  const types = { '.json': 'application/json', '.txt': 'text/plain', '.html': 'text/html' };
  http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (p === '/') { res.setHeader('content-type', 'text/html'); return res.end(`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>*,*::before,*::after{box-sizing:border-box}body{margin:0}[hidden]{display:none!important}</style></head><body>${fs.readFileSync('dist/race.html', 'utf8')}</body></html>`); }
    const f = path.join('dist', p);
    if (!fs.existsSync(f)) { res.statusCode = 404; return res.end(); }
    res.setHeader('content-type', types[path.extname(f)] ?? 'application/octet-stream');
    fs.createReadStream(f).pipe(res);
  }).listen(8124, () => console.log('http://localhost:8124'));
}
