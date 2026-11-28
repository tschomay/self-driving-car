// Build the game into dist/: game.js (bundle), mujoco.wasm, loops/ (Phase 4.6 loop data), index.html (standalone page) and
// artifact.html (the same page without the document shell, for publishing as a Claude artifact).
import { build, context } from 'esbuild';
import fs from 'fs';

const serve = process.argv.includes('--serve');
fs.mkdirSync('dist', { recursive: true });
fs.copyFileSync('../node_modules/@mujoco/mujoco/mujoco.wasm', 'dist/mujoco.wasm');   // web/node_modules: one install for both apps
fs.rmSync('dist/loops', { recursive: true, force: true });
fs.cpSync('loops', 'dist/loops', { recursive: true });   // tools/export_loops.py output
const page = fs.readFileSync('src/page.html', 'utf8');
fs.writeFileSync('dist/artifact.html', page);
fs.writeFileSync('dist/index.html', `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-capable" content="yes">
<style>*,*::before,*::after{box-sizing:border-box}body{margin:0}[hidden]{display:none!important}</style>
</head><body>
${page}
</body></html>
`);
const opts = {
  entryPoints: ['src/main.js'], bundle: true, format: 'esm', outfile: 'dist/game.js', minify: !serve, sourcemap: serve,
  target: 'es2020', platform: 'browser', external: ['module'], logLevel: 'info',
};
if (serve) {
  const ctx = await context(opts);
  await ctx.watch();
  const { port } = await ctx.serve({ servedir: 'dist', port: 8123 });
  console.log(`http://localhost:${port}`);
} else await build(opts);
