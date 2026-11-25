// Post-processing: MSAA scene render -> ambient occlusion (high quality only) -> depth of field (replays) -> bloom ->
// tone map -> a "lens" pass (grade, vignette, chromatic fringe, grain, letterbox).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const LensShader = {
  uniforms: { tDiffuse: { value: null }, time: { value: 0 }, letterbox: { value: 0 }, aspect: { value: 1 }, grain: { value: 0.035 }, fringe: { value: 0.0016 }, sunUv: { value: new THREE.Vector2(-9, -9) }, sunOn: { value: 0 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float time, letterbox, aspect, grain, fringe, sunOn; uniform vec2 sunUv; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + time * 7.13) * 43758.5453); }
    void main(){
      vec2 c = vUv - 0.5;
      float r2 = dot(c * vec2(aspect, 1.0), c * vec2(aspect, 1.0));
      // chromatic fringe toward the edges, as a real lens
      vec2 off = c * fringe * (1.0 + 6.0 * r2);
      vec3 col = vec3(texture2D(tDiffuse, vUv + off).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - off).b);
      // lens flare from the sun, if the sun itself is visible (bright where it should be)
      if (sunOn > 0.5) {
        vec3 s = texture2D(tDiffuse, clamp(sunUv, 0.001, 0.999)).rgb;
        float vis = smoothstep(0.82, 0.97, dot(s, vec3(0.333))) * step(0.0, sunUv.x) * step(sunUv.x, 1.0) * step(0.0, sunUv.y) * step(sunUv.y, 1.0);
        vec2 d = (vUv - sunUv) * vec2(aspect, 1.0);
        col += vis * vec3(1.0, 0.85, 0.6) * 0.35 * exp(-length(d) * 7.0);                       // glow
        col += vis * vec3(1.0, 0.9, 0.75) * 0.22 * exp(-abs(d.y) * 120.0) * exp(-abs(d.x) * 2.5);  // streak
        vec2 axis = vec2(0.5) - sunUv;
        for (int i = 1; i <= 4; i++) {                                                          // ghosts along the axis
          float k = float(i) * 0.48;
          vec2 g = (vUv - (sunUv + axis * k)) * vec2(aspect, 1.0);
          float r = 0.025 + 0.022 * float(i);
          vec3 tint = i == 1 ? vec3(0.4, 0.7, 1.0) : i == 2 ? vec3(1.0, 0.6, 0.3) : i == 3 ? vec3(0.5, 1.0, 0.6) : vec3(0.8, 0.5, 1.0);
          col += vis * tint * 0.07 * smoothstep(r, r * 0.7, length(g));
        }
      }
      // grade: a touch of contrast, warm highlights, cool shadows
      float l = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(col, col * vec3(1.04, 1.0, 0.94), smoothstep(0.4, 1.0, l));
      col = mix(col, col * vec3(0.94, 0.98, 1.05), 1.0 - smoothstep(0.0, 0.35, l));
      col = (col - 0.5) * 1.06 + 0.5;
      col = mix(vec3(l), col, 1.08);
      // vignette and grain
      col *= 1.0 - 0.32 * smoothstep(0.15, 0.95, r2);
      col += (hash(vUv * 900.0) - 0.5) * grain;
      // letterbox bars (replays)
      float bar = letterbox * 0.11;
      if (vUv.y < bar || vUv.y > 1.0 - bar) col = vec3(0.0);
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }`,
};

export class Post {
  constructor(renderer, scene, camera, level) {
    this.renderer = renderer; this.scene = scene; this.camera = camera;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
    const comp = this.composer = new EffectComposer(renderer, rt);
    comp.addPass(new RenderPass(scene, camera));
    this.ao = new GTAOPass(scene, camera, size.x, size.y);
    this.ao.updateGtaoMaterial({ radius: 0.35, distanceExponent: 1.4, thickness: 1.2, scale: 1.0, samples: 12 });
    this.ao.blendIntensity = 0.85;
    comp.addPass(this.ao);
    this.dof = new BokehPass(scene, camera, { focus: 6, aperture: 0.004, maxblur: 0.006 });
    this.dof.enabled = false;
    comp.addPass(this.dof);
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.32, 0.55, 0.88);
    comp.addPass(this.bloom);
    comp.addPass(new OutputPass());
    this.lens = new ShaderPass(LensShader);
    comp.addPass(this.lens);
    this.letter = 0;
    this.setLevel(level);
  }

  /** 2: everything; 1: no AO; 0: no post at all (plain render). */
  setLevel(level) {
    this.level = level;
    this.ao.enabled = level >= 2;
    this.bloom.enabled = level >= 1;
  }

  setSize(w, h) {
    this.composer.setPixelRatio(this.renderer.getPixelRatio());
    this.composer.setSize(w, h);
    this.lens.uniforms.aspect.value = w / h;
  }

  /** replay: letterbox and depth of field focused on the car at distance focus (m). */
  render(dt, replay, focus) {
    if (this.level <= 0) { this.renderer.render(this.scene, this.camera); return; }
    this.letter += ((replay ? 1 : 0) - this.letter) * Math.min(1, dt * 5);
    const L = this.lens.uniforms;
    L.letterbox.value = this.letter; L.time.value = (L.time.value + dt) % 100;
    this.dof.enabled = replay && this.level >= 1;
    if (this.dof.enabled) { this.dof.uniforms.focus.value = focus; this.dof.uniforms.aperture.value = 0.0035 / Math.max(1, focus * 0.4); }
    this.composer.render(dt);
  }
}
