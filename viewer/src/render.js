// three.js view of the world: z is up, metres, the same frame as the physics. From rc-racer's render.js
// (tschomay/sandbox projects/rc-racer); the circuit is a recorded sim loop and the cars replay recorded frames.
import * as THREE from 'three';
import { bakeGround, detailNormal, canvasTexture, fbm, PALETTE } from './terrainart.js';
import { makeCar, setSuspension, holoTime } from './carmesh.js';
import { Post } from './post.js';
import { dress, wind } from './world.js';
import { LANE_W, BANK_RUN, rng32 } from './track.js';
import { WHEEL_R, CORNER, MOUNT_Z } from './carmodel.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export const CAMERAS = ['TV', 'Chase', 'Orbit', 'On board', 'Heli'];
const SUN_ELEV = 30, SUN_AZ = 205;     // degrees: a late-afternoon sun, low enough for the jumps to throw shadows

export class View {
  constructor(canvas, quality) {
    this.quality = quality;             // { pixelRatio, shadows, shadowSize, groundSize }
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    r.setPixelRatio(quality.pixelRatio);
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.9;
    r.shadowMap.enabled = quality.shadows;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.04, 900);
    this.camMode = 0;
    this.camPos = new THREE.Vector3(); this.camLook = new THREE.Vector3(); this.camYaw = 0; this.camInit = false;
    this.world = new THREE.Group();     // everything that belongs to one circuit
    this.scene.add(this.world);
    this._sky();
    this._dust();
    this.cars = [];
    this.orbit = { yaw: 0.8, pitch: 0.35, dist: 2.2 };
    this.tmpQ = new THREE.Quaternion(); this.tmpV = new THREE.Vector3();
    this.noAO = [this.dust];            // drawn, but left out of the ambient-occlusion and depth-of-field depth passes
    this._clods();
    this.post = new Post(r, this.scene, this.camera, quality.post);
    for (const pass of [this.post.ao, this.post.dof]) {
      const orig = pass.render.bind(pass);
      pass.render = (...a) => {
        const hidden = this.noAO.filter((o) => o.visible); hidden.forEach((o) => (o.visible = false));
        const sky = this.sky.visible; this.sky.visible = pass !== this.post.ao && sky;
        orig(...a);
        hidden.forEach((o) => (o.visible = true)); this.sky.visible = sky;
      };
    }
    this.shake = 0; this.clock = 0;
  }

  setQuality(level) { this.post.setLevel(level); }

  resize(w, h) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.portrait = h > w;
    this.camera.updateProjectionMatrix();
    this.post.setSize(w, h);
  }

  _sky() {
    const el = THREE.MathUtils.degToRad(SUN_ELEV), az = THREE.MathUtils.degToRad(SUN_AZ);
    this.sunDir = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    // sky dome: blue overhead fading to a warm haze at the horizon, a sun glow, and drifting fair-weather clouds
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { sunDir: { value: this.sunDir }, time: { value: 0 } },
      vertexShader: 'varying vec3 vDir; void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }',
      fragmentShader: `
        uniform vec3 sunDir; uniform float time; varying vec3 vDir;
        float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
          return mix(mix(hash(i), hash(i+vec2(1,0)), f.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y); }
        float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int k = 0; k < 5; k++){ s += a * noise(p); p = p * 2.03 + 11.0; a *= 0.5; } return s; }
        void main(){
          vec3 d = normalize(vDir);
          float h = max(d.z, 0.0);
          vec3 zenith = vec3(0.13, 0.33, 0.70), horizon = vec3(0.66, 0.76, 0.86), haze = vec3(0.93, 0.84, 0.70);
          vec3 col = mix(horizon, zenith, pow(h, 0.5));
          float sd = max(dot(d, sunDir), 0.0);
          col = mix(col, haze, 0.55 * pow(1.0 - h, 6.0) * (0.45 + 0.55 * sd));
          col += vec3(1.0, 0.82, 0.55) * (0.35 * pow(sd, 8.0) + 0.5 * pow(sd, 90.0));
          col += vec3(1.0, 0.95, 0.85) * 6.0 * smoothstep(0.9994, 0.9998, sd);
          // clouds on a plane overhead
          vec2 uv = d.xy / (d.z + 0.12) * 0.55 + vec2(time * 0.004, time * 0.0015);
          float c = fbm(uv * 1.6), c2 = fbm(uv * 1.6 + sunDir.xy * 0.12);
          float cover = smoothstep(0.50, 0.74, c) * smoothstep(0.02, 0.22, d.z);
          float lit = clamp(0.5 + 2.2 * (c - c2), 0.0, 1.0);
          vec3 cloud = mix(vec3(0.60, 0.65, 0.74), vec3(1.0, 0.97, 0.92), lit);
          col = mix(col, cloud, cover * 0.9);
          if (d.z < 0.0) col = mix(col, vec3(0.42, 0.47, 0.36), smoothstep(0.0, -0.08, d.z));
          gl_FragColor = vec4(col * 1.25, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    const sky = this.sky = new THREE.Mesh(new THREE.SphereGeometry(800, 32, 20), skyMat);
    sky.frustumCulled = false; sky.renderOrder = -1;
    this.scene.add(sky);
    // image-based light from the sky itself, so paint and tires pick up the sky and the horizon glow
    const pm = new THREE.PMREMGenerator(this.renderer), tmp = new THREE.Scene();
    tmp.add(new THREE.Mesh(sky.geometry, skyMat));
    this.scene.environment = pm.fromScene(tmp, 0, 1, 2000).texture;
    this.scene.environmentIntensity = 0.4;
    pm.dispose();
    this.scene.fog = new THREE.Fog(0xb9c9d6, 70, 560);
    const sun = this.sun = new THREE.DirectionalLight(0xffe9c8, 3.6);
    sun.castShadow = this.quality.shadows;
    const s = this.quality.shadowSize;
    sun.shadow.mapSize.set(s, s);
    const cam = sun.shadow.camera;
    cam.left = -4.5; cam.right = 4.5; cam.top = 4.5; cam.bottom = -4.5; cam.near = 1; cam.far = 80;
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.01;
    this.scene.add(sun, sun.target);
    this.scene.add(new THREE.HemisphereLight(0xa9c6ee, 0x6d5a3c, 0.5));
  }

  _dust() {
    const N = this.dustN = 420;
    const geo = new THREE.BufferGeometry();
    this.dustPos = new Float32Array(N * 3); this.dustAux = new Float32Array(N * 2);   // size, alpha
    this.dustVel = new Float32Array(N * 3); this.dustLife = new Float32Array(N); this.dustAge = new Float32Array(N).fill(1e9);
    geo.setAttribute('position', new THREE.BufferAttribute(this.dustPos, 3));
    geo.setAttribute('aux', new THREE.BufferAttribute(this.dustAux, 2));
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { color: { value: new THREE.Color(0xc9a878) }, scale: { value: 600 } },
      vertexShader: `attribute vec2 aux; varying float vA; uniform float scale;
        void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); gl_Position = projectionMatrix * mv;
          vA = max(aux.y, 0.0) * smoothstep(0.35, 1.6, -mv.z);                       // fade dust right at the lens
          gl_PointSize = min(aux.x * scale / -mv.z, 90.0); }`,
      fragmentShader: `uniform vec3 color; varying float vA;
        void main(){ float d = length(gl_PointCoord - 0.5) * 2.0; float a = smoothstep(1.0, 0.25, d) * vA; if (a < 0.004) discard; gl_FragColor = vec4(color, a); }`,
    });
    const pts = this.dust = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    this.scene.add(pts);
    this.dustNext = 0;
  }

  puff(x, y, z, vx, vy, vz, size, life) {
    const i = this.dustNext; this.dustNext = (i + 1) % this.dustN;
    this.dustPos.set([x, y, z], i * 3); this.dustVel.set([vx, vy, vz], i * 3);
    this.dustAux[i * 2] = size; this.dustLife[i] = life; this.dustAge[i] = 0;
  }

  // ---- circuit --------------------------------------------------------------------------------------
  setCircuit(c) {
    this.clearCircuit();
    this.c = c;
    const W = this.world, q = this.quality;
    const ground = bakeGround(c, q.groundSize);
    const detail = detailNormal(256);
    detail.repeat.set(c.sizeX / 1.3, c.sizeY / 1.3);
    const mat = new THREE.MeshStandardMaterial({ map: ground, normalMap: detail, normalScale: new THREE.Vector2(0.5, 0.5), roughness: 0.96, metalness: 0 });
    // terrain in chunks, so off-screen parts are culled
    const CH = 48;
    for (let cy = 0; cy < c.ny - 1; cy += CH)
      for (let cx = 0; cx < c.nx - 1; cx += CH) {
        const w = Math.min(CH, c.nx - 1 - cx) + 1, h = Math.min(CH, c.ny - 1 - cy) + 1;
        const pos = new Float32Array(w * h * 3), nor = new Float32Array(w * h * 3), uv = new Float32Array(w * h * 2);
        for (let j = 0; j < h; j++)
          for (let i = 0; i < w; i++) {
            const gx = cx + i, gy = cy + j, k = j * w + i, qi = gy * c.nx + gx;
            pos[3 * k] = c.x0 + gx * c.cell; pos[3 * k + 1] = c.y0 + gy * c.cell; pos[3 * k + 2] = c.hf[qi];
            const hx = (c.hf[gy * c.nx + Math.min(c.nx - 1, gx + 1)] - c.hf[gy * c.nx + Math.max(0, gx - 1)]) / (2 * c.cell);
            const hy = (c.hf[Math.min(c.ny - 1, gy + 1) * c.nx + gx] - c.hf[Math.max(0, gy - 1) * c.nx + gx]) / (2 * c.cell);
            const n = Math.hypot(hx, hy, 1);
            nor[3 * k] = -hx / n; nor[3 * k + 1] = -hy / n; nor[3 * k + 2] = 1 / n;
            uv[2 * k] = gx / (c.nx - 1); uv[2 * k + 1] = gy / (c.ny - 1);
          }
        const idx = new Uint16Array((w - 1) * (h - 1) * 6);
        let t = 0;
        for (let j = 0; j < h - 1; j++)
          for (let i = 0; i < w - 1; i++) {
            const a = j * w + i, b = a + 1, d2 = a + w, e = d2 + 1;
            idx[t++] = a; idx[t++] = b; idx[t++] = e; idx[t++] = a; idx[t++] = e; idx[t++] = d2;
          }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
        geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        const mesh = new THREE.Mesh(geo, mat);
        mesh.receiveShadow = true; mesh.castShadow = q.shadows;
        W.add(mesh);
      }
    this._surroundings(c);
    this._trackside(c);
    this._tvCams(c);
    const d = dress(c, this.world, this.quality.grass);
    this.noAO = this.noAO.filter((o) => o !== this.grassObj).concat([d.grass]);
    this.grassObj = d.grass;
  }

  clearCircuit() {
    const W = this.world;
    W.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) { for (const k of ['map', 'normalMap', 'bumpMap']) if (m[k]) m[k].dispose(); m.dispose(); }
    });
    W.clear();
    this.clearMarks?.();
        this.camInit = false;
  }

  _surroundings(c) {
    const W0 = this.world, rng = rng32(c.seed + 4242), P = PALETTE;
    const W = new THREE.Group(); W.position.z = c.plateau; W0.add(W);
    // the field beyond the lot
    const fieldTex = canvasTexture(256, 256, (g, w, h) => {
      const img = g.createImageData(w, h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const n = fbm(x / 26, y / 26, 3), o = 4 * (y * w + x);
        img.data[o] = P.grass[0] + (P.grassDry[0] - P.grass[0]) * n; img.data[o + 1] = P.grass[1] + (P.grassDry[1] - P.grass[1]) * n;
        img.data[o + 2] = P.grass[2] + (P.grassDry[2] - P.grass[2]) * n; img.data[o + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    });
    fieldTex.wrapS = fieldTex.wrapT = THREE.RepeatWrapping; fieldTex.repeat.set(60, 60);
    // the field beyond the terrain mesh, level with the sim's plateau, with a hole where the terrain is
    const outer = new THREE.Shape([[-900, -900], [900, -900], [900, 900], [-900, 900]].map(([x, y]) => new THREE.Vector2(x, y)));
    const hx0 = c.sizeX / 2 - 0.05, hy0 = c.sizeY / 2 - 0.05;
    outer.holes.push(new THREE.Path([[-hx0, -hy0], [-hx0, hy0], [hx0, hy0], [hx0, -hy0]].map(([x, y]) => new THREE.Vector2(x, y))));
    const fieldGeo = new THREE.ShapeGeometry(outer);
    const fuv = fieldGeo.attributes.uv; for (let i = 0; i < fuv.count; i++) fuv.setXY(i, (fuv.getX(i) + 900) / 1800, (fuv.getY(i) + 900) / 1800);
    const field = new THREE.Mesh(fieldGeo, new THREE.MeshStandardMaterial({ map: fieldTex, roughness: 1 }));
    field.position.set(c.centerX, c.centerY, -0.01); field.receiveShadow = true;
    W.add(field);
    // far hills
    const hillGeo = new THREE.CylinderGeometry(520, 520, 1, 96, 1, true);
    hillGeo.rotateX(Math.PI / 2);
    const hp = hillGeo.attributes.position;
    for (let i = 0; i < hp.count; i++) {
      const a = Math.atan2(hp.getY(i), hp.getX(i));
      if (hp.getZ(i) > 0) hp.setZ(i, 14 + 46 * fbm(a * 2.2 + 5 + c.seed, 1.7, 4)); else hp.setZ(i, -2);
    }
    hillGeo.computeVertexNormals();
    const hills = new THREE.Mesh(hillGeo, new THREE.MeshStandardMaterial({ color: 0x5c7350, roughness: 1, side: THREE.BackSide }));
    hills.position.set(c.centerX, c.centerY, 0);
    W.add(hills);
    // trees around the lot
    const nT = 150, trunkGeo = new THREE.CylinderGeometry(0.16, 0.24, 1, 9), leafGeo = new THREE.IcosahedronGeometry(1, 3);
    trunkGeo.rotateX(Math.PI / 2);
    const lp = leafGeo.attributes.position;       // lumpy crowns
    for (let i = 0; i < lp.count; i++) {
      const k = 0.78 + 0.3 * fbm(lp.getX(i) * 2 + 3, lp.getY(i) * 2 + lp.getZ(i) * 3, 3) + 0.08 * fbm(lp.getX(i) * 9, lp.getY(i) * 9 + lp.getZ(i) * 7, 2);
      lp.setXYZ(i, lp.getX(i) * k, lp.getY(i) * k, lp.getZ(i) * k);
    }
    leafGeo.computeVertexNormals();
    const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x5a4632, roughness: 1 }), nT);
    const leaves = new THREE.InstancedMesh(leafGeo, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85 }), nT * 3);
    leaves.castShadow = true;
    const m4 = new THREE.Matrix4(), col = new THREE.Color(), qI = new THREE.Quaternion();
    const hx = c.sizeX / 2, hy = c.sizeY / 2;
    for (let i = 0; i < nT; i++) {
      let x, y;
      do { x = rng.uniform(-hx - 70, hx + 70); y = rng.uniform(-hy - 70, hy + 70); } while (Math.abs(x) < hx + 3.5 && Math.abs(y) < hy + 3.5);
      x += c.centerX; y += c.centerY;
      const th = rng.uniform(2.2, 4.0), cr = rng.uniform(1.7, 3.1);
      m4.compose(new THREE.Vector3(x, y, th / 2), qI, new THREE.Vector3(1, 1, th)); trunks.setMatrixAt(i, m4);
      for (let k = 0; k < 3; k++) {
        const s = cr * (k ? 0.68 : 1);
        qI.setFromEuler(new THREE.Euler(rng() * 6, rng() * 6, rng() * 6));
        m4.compose(new THREE.Vector3(x + (k ? rng.uniform(-1.1, 1.1) : 0), y + (k ? rng.uniform(-1.1, 1.1) : 0), th + cr * (k === 1 ? 1.25 : k === 2 ? 0.8 : 0.55)), qI, new THREE.Vector3(s, s, s * rng.uniform(0.85, 1.15)));
        leaves.setMatrixAt(i * 3 + k, m4);
        col.setHSL(rng.uniform(0.22, 0.31), rng.uniform(0.3, 0.5), rng.uniform(0.2, 0.31)); leaves.setColorAt(i * 3 + k, col);
      }
      qI.identity();
    }
    W.add(trunks, leaves);
    // a rail fence around the lot
    const fenceMat = new THREE.MeshStandardMaterial({ color: 0xe8e2d2, roughness: 0.8 });
    const inset = 1.2, fx0 = c.x0 + inset, fx1 = c.x0 + c.sizeX - inset, fy0 = c.y0 + inset, fy1 = c.y0 + c.sizeY - inset;
    const posts = [];
    for (let x = fx0; x <= fx1 + 0.01; x += (fx1 - fx0) / Math.round((fx1 - fx0) / 2.4)) posts.push([x, fy0], [x, fy1]);
    for (let y = fy0; y <= fy1 + 0.01; y += (fy1 - fy0) / Math.round((fy1 - fy0) / 2.4)) posts.push([fx0, y], [fx1, y]);
    const postMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.09, 0.09, 0.95), fenceMat, posts.length);
    posts.forEach(([x, y], i) => { m4.makeTranslation(x, y, 0.47); postMesh.setMatrixAt(i, m4); });
    W.add(postMesh);
    for (const z of [0.42, 0.8]) {
      W.add(new THREE.Mesh(new THREE.BoxGeometry(fx1 - fx0, 0.04, 0.11), fenceMat).translateX((fx0 + fx1) / 2).translateY(fy0).translateZ(z));
      W.add(new THREE.Mesh(new THREE.BoxGeometry(fx1 - fx0, 0.04, 0.11), fenceMat).translateX((fx0 + fx1) / 2).translateY(fy1).translateZ(z));
      W.add(new THREE.Mesh(new THREE.BoxGeometry(0.04, fy1 - fy0, 0.11), fenceMat).translateX(fx0).translateY((fy0 + fy1) / 2).translateZ(z));
      W.add(new THREE.Mesh(new THREE.BoxGeometry(0.04, fy1 - fy0, 0.11), fenceMat).translateX(fx1).translateY((fy0 + fy1) / 2).translateZ(z));
    }
  }

  _trackside(c) {
    const W = this.world, half = LANE_W / 2;
    // drain-pipe lane markers along the foot of both banks
    const pipeTex = canvasTexture(64, 8, (g) => {
      g.fillStyle = '#ece8dc'; g.fillRect(0, 0, 64, 8);
      g.fillStyle = '#c9c4b6'; for (let i = 0; i < 64; i += 4) g.fillRect(i, 0, 1, 8);    // corrugation
    });
    pipeTex.wrapS = pipeTex.wrapT = THREE.RepeatWrapping;
    for (const side of [1, -1]) {
      const pts = [];
      for (let s = 0; s < c.length; s += 0.4) {
        const p = c.pointAt(s, side * (half + 0.1));
        pts.push(new THREE.Vector3(p.x, p.y, c.height(p.x, p.y) + 0.035));
      }
      const curve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.3);
      const tex = pipeTex.clone(); tex.repeat.set(c.length / 0.25, 1); tex.needsUpdate = true;
      const pipe = new THREE.Mesh(new THREE.TubeGeometry(curve, pts.length * 2, 0.055, 7, true),
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.7, color: 0xd9d3c4 }));
      pipe.castShadow = true; pipe.receiveShadow = true;
      W.add(pipe);
    }
    // start gantry
    const p0 = c.pointAt(0), g = new THREE.Group(), dark = new THREE.MeshStandardMaterial({ color: 0x24262b, roughness: 0.6, metalness: 0.5 });
    const span = half + BANK_RUN + 0.25, hgt = 1.25;
    for (const sy of [1, -1]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, hgt + 0.4), dark);
      post.position.set(0, sy * span, hgt / 2 + 0.1); post.castShadow = true; g.add(post);
    }
    const chk = canvasTexture(512, 64, (x2, w, h) => {
      x2.fillStyle = '#16171a'; x2.fillRect(0, 0, w, h);
      x2.fillStyle = '#f1eee4';
      for (let i = 0; i < 32; i++) for (let j = 0; j < 4; j++) if ((i + j) % 2 === 0 && (i < 9 || i > 22)) x2.fillRect(i * 16, j * 16, 16, 16);
      x2.font = '700 44px "Barlow Condensed", sans-serif'; x2.textAlign = 'center'; x2.textBaseline = 'middle';
      x2.fillText('START · FINISH', w / 2, h / 2 + 3);
    });
    const flag = new THREE.MeshStandardMaterial({ map: chk, roughness: 0.7 });
    const banner = new THREE.Mesh(new THREE.BoxGeometry(0.05, span * 2 + 0.1, 0.3), [flag, flag, dark, dark, dark, dark]);
    banner.position.set(0, 0, hgt + 0.2); banner.castShadow = true; g.add(banner);
    g.position.set(p0.x, p0.y, c.height(p0.x, p0.y)); g.rotation.z = p0.heading;
    W.add(g);
    // the drivers' stand: a raised deck beside the start straight, outside the track (where you would stand)
    let best = null;
    for (const side of [1, -1]) for (const s of [3, 8, -4, 14]) for (const off of [6.5, 5.5]) {
      const p = c.pointAt(s, side * off);
      const inLot = p.x > c.x0 + 2.5 && p.x < c.x0 + c.sizeX - 2.5 && p.y > c.y0 + 2.5 && p.y < c.y0 + c.sizeY - 2.5;
      let clear = inLot;
      for (let a = 0; a < 6.3 && clear; a += 0.5) if (c.locate(p.x + 1.9 * Math.cos(a), p.y + 1.9 * Math.sin(a))) clear = false;
      if (clear && !best) best = { p, side };
    }
    if (!best) { const p = c.pointAt(3, 5.5); best = { p, side: 1 }; }
    const st = new THREE.Group(), wood = new THREE.MeshStandardMaterial({ color: 0x9a7b54, roughness: 0.85 }), deckH = 1.1;
    st.add(new THREE.Mesh(new THREE.BoxGeometry(3.6, 1.3, 0.06), wood).translateZ(deckH));
    for (const x of [-1.7, 0, 1.7]) for (const y of [-0.58, 0.58]) st.add(new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, deckH + (y > 0 ? 0 : 1.05)), dark).translateX(x).translateY(y).translateZ((deckH + (y > 0 ? 0 : 1.05)) / 2));
    st.add(new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.05, 0.06), dark).translateY(0.6).translateZ(deckH + 0.55));
    for (const x of [-1.7, 0, 1.7]) st.add(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.55), dark).translateX(x).translateY(0.6).translateZ(deckH + 0.28));
    const roof = new THREE.Mesh(new THREE.BoxGeometry(3.9, 1.7, 0.04), new THREE.MeshStandardMaterial({ color: 0xb8321c, roughness: 0.7 }));
    roof.position.set(0, -0.1, deckH + 2.12); roof.rotation.x = 0.1; st.add(roof);
    st.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    // +y of the stand faces the track
    st.position.set(best.p.x, best.p.y, c.plateau + 0.02);
    st.rotation.z = best.p.heading + (best.side > 0 ? Math.PI : 0);
    W.add(st);
    this.standPos = new THREE.Vector3(best.p.x, best.p.y, c.plateau + deckH);
  }

  stepDust(dt) {
    const N = this.dustN, p = this.dustPos, v = this.dustVel, aux = this.dustAux;
    for (let i = 0; i < N; i++) {
      if (this.dustAge[i] > this.dustLife[i]) { aux[2 * i + 1] = 0; continue; }
      this.dustAge[i] += dt;
      const t = this.dustAge[i] / this.dustLife[i];
      p[3 * i] += v[3 * i] * dt; p[3 * i + 1] += v[3 * i + 1] * dt; p[3 * i + 2] += v[3 * i + 2] * dt;
      v[3 * i] *= 1 - 2.5 * dt; v[3 * i + 1] *= 1 - 2.5 * dt; v[3 * i + 2] = v[3 * i + 2] * (1 - 2 * dt) + 0.05 * dt;
      aux[2 * i] += 0.35 * dt;
      aux[2 * i + 1] = 0.34 * Math.max(0, 1 - t) * Math.min(1, t * 8);   // never negative: a negative alpha paints black
    }
    this.dust.geometry.attributes.position.needsUpdate = true;
    this.dust.geometry.attributes.aux.needsUpdate = true;
    this.dust.material.uniforms.scale.value = this.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
  }

  // ---- trackside TV cameras ---------------------------------------------------------------------------
  /** Poles outside the track every ~16 m, on the outside of the bend, clear of other parts of the loop. */
  _tvCams(c) {
    const n = Math.max(6, Math.round(c.length / 16)), cams = [], half = LANE_W / 2 + BANK_RUN;
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x2a2c30, roughness: 0.6, metalness: 0.4 });
    for (let k = 0; k < n; k++) {
      const s = (k + 0.5) * c.length / n;
      let kap = 0;
      for (let ds = -4; ds <= 4; ds += 1) kap += c.kappaAt(s + ds);
      const order = kap > 0 ? [-1, 1] : [1, -1];             // outside of the bend first
      let spot = null;
      for (const side of order) for (const off of [2.4, 3.2, 1.9]) {
        if (spot) break;
        const p = c.pointAt(s, side * (half + off));
        let clear = true;
        for (let a = 0; a < 6.3 && clear; a += 0.7) {
          const q = c.locate(p.x + 0.6 * Math.cos(a), p.y + 0.6 * Math.sin(a));
          if (q && Math.abs(q.d) < half + 0.6) clear = false;
        }
        const inLot = p.x > c.x0 + 1 && p.x < c.x0 + c.sizeX - 1 && p.y > c.y0 + 1 && p.y < c.y0 + c.sizeY - 1;
        if (clear && inLot) spot = p;
      }
      if (!spot) continue;
      const z = c.height(spot.x, spot.y);
      cams.push({ s, pos: new THREE.Vector3(spot.x, spot.y, z + 1.5) });
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 1.4, 8).rotateX(Math.PI / 2), poleMat);
      pole.position.set(spot.x, spot.y, z + 0.7); pole.castShadow = true;
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.09, 0.1), poleMat);
      head.position.set(spot.x, spot.y, z + 1.45); head.castShadow = true;
      this.world.add(pole, head);
    }
    this.tv = cams;
    this.tvIdx = -1;
  }

  /** The TV camera covering lap position s: the car comes towards a camera, passes it, and the next one cuts in. */
  tvFor(s) {
    const L = this.c.length, cams = this.tv;
    let best = 0, bd = 1e9;
    for (let i = 0; i < cams.length; i++) {
      const ahead = ((cams[i].s - s) % L + L) % L;           // metres until the car reaches this camera
      const score = ahead < L - 5 ? ahead : ahead - L;         // up to 5 m past a camera keeps it
      if (score > -5 && score < bd) { bd = score; best = i; }
    }
    return best;
  }

  // ---- cars ---------------------------------------------------------------------------------------------
  addCar(color, opts = {}) {
    const car = makeCar(color, opts);
    if (opts.ghost) this.noAO.push(car.body);
    this.scene.add(car.body);
    car.spin = 0;
    car.blobs = opts.ghost ? [] : car.wheels.map(() => {
      const m = new THREE.Mesh(this._blobGeo(), new THREE.MeshBasicMaterial({ map: this._blobTex(), transparent: true, depthWrite: false, color: 0x000000, opacity: 0, polygonOffset: true, polygonOffsetFactor: -2 }));
      m.renderOrder = 2; this.scene.add(m); this.noAO.push(m); return m;
    });
    car.marks = opts.marks ? car.wheels.map(() => null) : null;
    return car;
  }

  removeCar(car) { this.noAO = this.noAO.filter((o) => o !== car.body && !car.blobs.includes(o)); this.scene.remove(car.body); car.blobs.forEach((b) => { this.scene.remove(b); b.material.dispose(); }); }

  _blobTex() {
    if (!this.blobTex) this.blobTex = canvasTexture(64, 64, (g) => {
      const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.45, 'rgba(255,255,255,0.6)'); r.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = r; g.fillRect(0, 0, 64, 64);
    }, false);
    return this.blobTex;
  }
  _blobGeo() { return this.blobGeo ??= new THREE.PlaneGeometry(1, 1); }

  /** Pose a car from a sample p: position, quaternion, suspension per wheel, steer angle; spin from speed. */
  poseCar(vis, p, dt) {
    vis.body.visible = true;
    vis.body.position.set(p.x, p.y, p.z);
    vis.body.quaternion.set(p.qx, p.qy, p.qz, p.qw);
    const spin = (vis.spin + p.v / WHEEL_R * dt) % (2 * Math.PI);
    vis.spin = Number.isFinite(spin) ? spin : 0;            // never let one bad sample stick
    setSuspension(vis, [p.s1, p.s2, p.s3, p.s4], p.st, vis.spin);
    if (!vis.blobs.length) return;
    vis.body.updateMatrixWorld(true);
    const c = this.c, lat = -p.vx * Math.sin(p.yaw) + p.vy * Math.cos(p.yaw);
    vis.wheels.forEach((w, k) => {
      const wp = w.getWorldPosition(this.tmpV), h = c.height(wp.x, wp.y), gap = wp.z - WHEEL_R - h;
      // contact shadow: dark and tight when the tire is down, fading and spreading as it lifts
      const b = vis.blobs[k], near = THREE.MathUtils.clamp(1 - gap / 0.12, 0, 1);
      b.visible = near > 0.02;
      b.position.set(wp.x, wp.y, h + 0.002);
      const sz = 0.085 + Math.max(0, gap) * 0.7;
      b.scale.set(sz * 1.25, sz, 1);
      b.rotation.set(0, 0, p.yaw);
      b.material.opacity = 0.62 * near * near;
      // skid marks where a tire slides sideways, spins up or lands
      if (!vis.marks) return;
      const down = (p.gnd & (1 << k)) !== 0 && gap < 0.012;
      const sliding = down && dt > 0 && (Math.abs(lat) > 0.9 || (k >= 2 && p.ct > 0.6 && p.v < 3.5));
      if (!sliding) { vis.marks[k] = null; return; }
      const last = vis.marks[k];
      if (!last) { vis.marks[k] = { x: wp.x, y: wp.y }; return; }
      const dx = wp.x - last.x, dy = wp.y - last.y, d = Math.hypot(dx, dy);
      if (d < 0.05) return;
      if (d < 0.6) this.addMark((wp.x + last.x) / 2, (wp.y + last.y) / 2, Math.atan2(dy, dx), d, Math.min(1, Math.abs(lat) / 2.5 + 0.35));
      vis.marks[k] = { x: wp.x, y: wp.y };
    });
  }

  hideCar(vis) { vis.body.visible = false; vis.blobs.forEach((b) => (b.visible = false)); }

  // ---- skid marks: a ring buffer of flat quads on the ground ----------------------------------------------
  _marksInit() {
    const N = this.markN = 5000;
    const mat = new THREE.MeshBasicMaterial({ color: 0x3b2817, transparent: true, opacity: 0.38, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1 });
    const m = this.markMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), mat, N);
    m.count = 0; m.frustumCulled = false; m.renderOrder = 1; this.noAO.push(m);
    m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3).fill(1), 3);
    this.scene.add(m);
    this.markNext = 0;
  }
  addMark(x, y, ang, len, strength) {
    if (!this.markMesh) this._marksInit();
    const i = this.markNext, m4 = new THREE.Matrix4(), q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), ang);
    m4.compose(new THREE.Vector3(x, y, this.c.height(x, y) + 0.003), q, new THREE.Vector3(len + 0.01, 0.034, 1));
    this.markMesh.setMatrixAt(i, m4);
    this.markMesh.setColorAt(i, new THREE.Color(1, 1, 1).multiplyScalar(0.4 + 0.6 * strength));
    this.markNext = (i + 1) % this.markN;
    this.markMesh.count = Math.max(this.markMesh.count, i + 1);
    this.markMesh.instanceMatrix.needsUpdate = true; this.markMesh.instanceColor.needsUpdate = true;
  }
  clearMarks() { if (this.markMesh) { this.markMesh.count = 0; this.markNext = 0; } }

  /** Dust: rear wheels throw dirt with speed; a landing throws a ring. */
  dustFrom(p, landed, dt) {
    const c = this.c, yaw = Math.atan2(2 * (p.qw * p.qz + p.qx * p.qy), 1 - 2 * (p.qy * p.qy + p.qz * p.qz));
    const vx = p.v * Math.cos(yaw), vy = p.v * Math.sin(yaw);
    if (landed) {
      for (let k = 0; k < 26; k++) {
        const a = Math.random() * 6.283, v = 0.6 + Math.random() * 1.6;
        this.puff(p.x + 0.2 * Math.cos(a), p.y + 0.2 * Math.sin(a), c.height(p.x, p.y) + 0.03, v * Math.cos(a) + vx * 0.3, v * Math.sin(a) + vy * 0.3, 0.3 + Math.random() * 0.6, 0.16 + Math.random() * 0.2, 0.7 + Math.random() * 0.6);
      }
    }
    const rate = Math.max(0, p.v - 1.5) * 6 * dt;
    for (const k of [2, 3]) {
      if (!(p.gnd & (1 << k)) || Math.random() > rate) continue;
      const w = CORNER[k === 2 ? 'rl' : 'rr'];
      const cy = Math.cos(yaw), sy = Math.sin(yaw), x = p.x + w[0] * cy - w[1] * sy, y = p.y + w[0] * sy + w[1] * cy;
      this.puff(x, y, c.height(x, y) + 0.01, -vx * 0.15 + (Math.random() - 0.5) * 0.5, -vy * 0.15 + (Math.random() - 0.5) * 0.5,
        0.25 + Math.random() * 0.5, 0.09 + Math.random() * 0.1, 0.5 + Math.random() * 0.5);
    }
  }

  /** Camera for car sample p (with yaw and velocity), in mode CAMERAS[camMode]. cut: jump straight to the shot. */
  updateCamera(p, vis, dt, cut) {
    const cam = this.camera, c = this.c, mode = CAMERAS[this.camMode];
    const k = (rate) => 1 - Math.exp(-rate * dt);
    const moving = p.v > 1.2;
    const want = moving ? Math.atan2(p.vy, p.vx) : p.yaw;
    if (!this.camInit || cut) this.camYaw = want;
    let dyaw = want - this.camYaw; dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
    this.camYaw += dyaw * k(moving ? 5.5 : 2.5);
    let px, py, pz, lx = p.x, ly = p.y, lz = p.z + 0.05, fov = 58, snap = !!cut;
    cam.up.set(0, 0, 1);
    if (mode === 'Chase') {
      const back = this.portrait ? 1.9 : 1.4, up = this.portrait ? 0.75 : 0.55;
      px = p.x - back * Math.cos(this.camYaw); py = p.y - back * Math.sin(this.camYaw);
      pz = Math.max(p.z + up, c.height(px, py) + 0.25);
      lx += 0.6 * Math.cos(this.camYaw); ly += 0.6 * Math.sin(this.camYaw);
      fov = (this.portrait ? 72 : 60) + THREE.MathUtils.clamp(p.v - 4, 0, 8) * 1.3;     // a speed kick
    } else if (mode === 'TV') {
      const i = this.tvFor(p.s);
      if (i !== this.tvIdx) { this.tvIdx = i; snap = true; }
      const cp = this.tv[i].pos;
      px = cp.x; py = cp.y; pz = cp.z;
      const dist = Math.hypot(p.x - px, p.y - py, p.z - pz);
      fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(1.3 / Math.max(dist, 1.5))), 7, 55) * (this.portrait ? 1.45 : 1);
      // lead the car a little, as a camera operator does
      lx += p.vx * 0.12; ly += p.vy * 0.12;
      const tt = this.clock;                                    // a hand on the camera: slow drift
      lx += 0.05 * Math.sin(tt * 0.9) + 0.02 * Math.sin(tt * 2.3); lz += 0.04 * Math.sin(tt * 1.3 + 1);
    } else if (mode === 'Orbit') {
      const o = this.orbit;
      if (!this.dragging) o.yaw += 0.12 * dt;               // a slow drift when left alone
      px = p.x + o.dist * Math.cos(o.pitch) * Math.cos(o.yaw); py = p.y + o.dist * Math.cos(o.pitch) * Math.sin(o.yaw);
      pz = Math.max(p.z + o.dist * Math.sin(o.pitch), c.height(px, py) + 0.1);
      fov = this.portrait ? 70 : 55; snap = true;
    } else if (mode === 'On board') {
      const f = new THREE.Vector3(0.02, 0, 0.075).applyQuaternion(vis.body.quaternion).add(vis.body.position);
      const l = new THREE.Vector3(2, 0, 0.1).applyQuaternion(vis.body.quaternion).add(vis.body.position);
      const u = new THREE.Vector3(0, 0, 1).applyQuaternion(vis.body.quaternion);
      cam.up.copy(u);
      px = f.x; py = f.y; pz = f.z; lx = l.x; ly = l.y; lz = l.z; fov = this.portrait ? 95 : 80; snap = true;
    } else {
      px = p.x - 4.5 * Math.cos(this.camYaw); py = p.y - 4.5 * Math.sin(this.camYaw); pz = p.z + 7.5; fov = this.portrait ? 62 : 48;
    }
    if (!this.camInit || snap) { this.camPos.set(px, py, pz); this.camLook.set(lx, ly, lz); this.camInit = true; }
    else {
      this.camPos.x += (px - this.camPos.x) * k(9); this.camPos.y += (py - this.camPos.y) * k(9); this.camPos.z += (pz - this.camPos.z) * k(5);
      this.camLook.x += (lx - this.camLook.x) * k(14); this.camLook.y += (ly - this.camLook.y) * k(14); this.camLook.z += (lz - this.camLook.z) * k(9);
    }
    if (mode === 'TV') this.camLook.lerp(this.tmpV.set(lx, ly, lz), snap ? 1 : k(10));
    cam.position.copy(this.camPos);
    if (this.shake > 0.001 && mode !== 'Orbit') {               // landing shake
      const a = this.shake, t = this.clock * 40;
      cam.position.x += a * 0.03 * Math.sin(t * 1.1); cam.position.y += a * 0.03 * Math.sin(t * 1.7 + 2); cam.position.z += a * 0.04 * Math.sin(t * 2.3 + 1);
    }
    this.shake *= Math.exp(-dt * 7);
    cam.lookAt(this.camLook);
    if (snap && mode === 'TV') { cam.fov = fov; cam.updateProjectionMatrix(); }
    else if (Math.abs(cam.fov - fov) > 0.01) { cam.fov += (fov - cam.fov) * k(4); cam.updateProjectionMatrix(); }
    this.sun.target.position.set(p.x, p.y, p.z);
    this.sun.position.copy(this.sun.target.position).addScaledVector(this.sunDir, 40);
  }

  /** replay: letterbox + depth of field on the car. */
  render(dt = 0.016, replay = false, focusOn = null) {
    this.clock += dt;
    wind.value += dt; holoTime.value = performance.now() / 1000;
    this.sky.position.copy(this.camera.position); this.sky.material.uniforms.time.value = performance.now() / 1000;
    // the sun on screen, for the lens flare
    const sp = this.tmpV.copy(this.camera.position).addScaledVector(this.sunDir, 500).project(this.camera);
    const L = this.post.lens.uniforms;
    L.sunUv.value.set(sp.x * 0.5 + 0.5, sp.y * 0.5 + 0.5);
    L.sunOn.value = sp.z < 1 && Math.abs(sp.x) < 1.3 && Math.abs(sp.y) < 1.3 ? 1 : 0;
    const focus = focusOn ? this.camera.position.distanceTo(focusOn) : 6;
    this.post.render(dt, replay, focus);
  }

  // ---- dirt clods thrown by the tires --------------------------------------------------------------------
  _clods() {
    const N = this.clodN = 420;
    const geo = new THREE.DodecahedronGeometry(1, 0);
    const m = this.clodMesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color: 0x6e4a2c, roughness: 1, flatShading: true }), N);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); m.frustumCulled = false; m.castShadow = false; m.count = N;
    this.clodP = new Float32Array(N * 3); this.clodV = new Float32Array(N * 3); this.clodL = new Float32Array(N); this.clodS = new Float32Array(N); this.clodNext = 0;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < N; i++) m.setMatrixAt(i, zero);
    this.scene.add(m);
  }
  clod(x, y, z, vx, vy, vz, size) {
    const i = this.clodNext; this.clodNext = (i + 1) % this.clodN;
    this.clodP.set([x, y, z], 3 * i); this.clodV.set([vx, vy, vz], 3 * i); this.clodL[i] = 1.2; this.clodS[i] = size;
  }
  stepClods(dt) {
    const m = this.clodMesh, P = this.clodP, Vv = this.clodV, m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
    for (let i = 0; i < this.clodN; i++) {
      if (this.clodL[i] <= 0) continue;
      this.clodL[i] -= dt;
      Vv[3 * i + 2] -= 9.81 * dt;
      for (let k = 0; k < 3; k++) P[3 * i + k] += Vv[3 * i + k] * dt;
      const g = this.c ? this.c.height(P[3 * i], P[3 * i + 1]) : 0;
      if (P[3 * i + 2] < g || this.clodL[i] <= 0) { this.clodL[i] = 0; m.setMatrixAt(i, m4.makeScale(0, 0, 0)); continue; }
      const s = this.clodS[i];
      e.set(this.clodL[i] * 13 + i, this.clodL[i] * 9, i); q.setFromEuler(e);
      m.setMatrixAt(i, m4.compose(pos.set(P[3 * i], P[3 * i + 1], P[3 * i + 2]), q, sc.set(s, s, s * 0.8)));
    }
    m.instanceMatrix.needsUpdate = true;
  }
  /** Roost: rear tires on the power throw clods back and up; a landing kicks a ring of them. */
  roost(p, landed, dt) {
    const yaw = p.yaw, cy = Math.cos(yaw), sy = Math.sin(yaw);
    if (landed) {
      this.shake = Math.min(1.5, this.shake + 0.8);
      for (let k = 0; k < 24; k++) { const a = Math.random() * 6.283, v = 0.8 + Math.random() * 1.8; this.clod(p.x + 0.2 * Math.cos(a), p.y + 0.2 * Math.sin(a), p.z - 0.08, v * Math.cos(a), v * Math.sin(a), 0.8 + Math.random() * 1.6, 0.004 + Math.random() * 0.008); }
    }
    const rate = (p.ct > 0.3 && p.v > 1.5 ? (p.ct - 0.3) * 60 : 0) * dt;
    for (const k of [2, 3]) {
      if (!(p.gnd & (1 << k))) continue;
      let n = rate; while (n > 0) { if (Math.random() < n) {
        const w = CORNER[k === 2 ? 'rl' : 'rr'], x = p.x + (w[0] - 0.04) * cy - w[1] * sy, y = p.y + (w[0] - 0.04) * sy + w[1] * cy;
        const back = 1.2 + Math.random() * 2.2, side = (Math.random() - 0.5) * 1.2;
        this.clod(x, y, this.c.height(x, y) + 0.02, p.vx * 0.4 - back * cy - side * sy, p.vy * 0.4 - back * sy + side * cy, 0.6 + Math.random() * 1.5, 0.003 + Math.random() * 0.007);
      } n -= 1; }
    }
  }
}
