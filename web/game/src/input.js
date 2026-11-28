// Driver input: Bluetooth/USB gamepad, touch (two thumbs, like a stick transmitter), keyboard.
// Output: steer in [-1, 1] (+ = left, as the sim) and throttle in [-1, 1] (- = brake, then reverse).

const dead = (v, dz) => (Math.abs(v) < dz ? 0 : Math.sign(v) * (Math.abs(v) - dz) / (1 - dz));
const expo = (v, e) => v * (1 - e + e * v * v);          // transmitter-style expo: softer around centre

export class Input {
  constructor(el) {
    this.steer = 0; this.throttle = 0; this.source = 'none';
    this.keys = new Set(); this.pressed = new Set();     // one-shot actions: 'reset', 'camera', 'pause', 'confirm'
    this.expo = 0.35;
    this.touch = { steer: null, throttle: null };        // { id, x0, y0, x, y }
    this.pad = null; this.padPrev = [];
    this.keySteer = 0;
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      this.keys.add(e.code);
      const act = { KeyR: 'reset', KeyC: 'camera', Escape: 'pause', KeyP: 'pause', Enter: 'confirm' }[e.code];
      if (act) this.pressed.add(act);
      if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
    const opts = { passive: false };
    el.addEventListener('touchstart', (e) => this._touch(e, 'start'), opts);
    el.addEventListener('touchmove', (e) => this._touch(e, 'move'), opts);
    el.addEventListener('touchend', (e) => this._touch(e, 'end'), opts);
    el.addEventListener('touchcancel', (e) => this._touch(e, 'end'), opts);
    this.el = el;
  }

  _touch(e, kind) {
    e.preventDefault();
    const r = this.el.getBoundingClientRect();
    for (const t of e.changedTouches) {
      const x = t.clientX - r.left, y = t.clientY - r.top;
      if (kind === 'start') {
        const side = x < r.width / 2 ? 'steer' : 'throttle';
        if (!this.touch[side]) this.touch[side] = { id: t.identifier, x0: x, y0: y, x, y };
      } else for (const side of ['steer', 'throttle']) {
        const s = this.touch[side];
        if (!s || s.id !== t.identifier) continue;
        if (kind === 'move') { s.x = x; s.y = y; } else this.touch[side] = null;
      }
    }
  }

  /** Call once per frame. */
  poll(dt) {
    let steer = 0, thr = 0, source = null;
    // gamepad
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    let pad = null;
    for (const p of pads) if (p && p.connected && p.axes.length >= 2) { pad = p; break; }
    this.pad = pad;
    if (pad) {
      const b = (i) => (pad.buttons[i] ? pad.buttons[i].value : 0);
      const sx = dead(pad.axes[0] || 0, 0.08);
      const trig = b(7) - b(6);                                        // right trigger drives, left brakes / reverses
      const ry = pad.axes.length > 3 ? -dead(pad.axes[3] || 0, 0.12) : 0;   // or the right stick, as on a stick radio
      const t = Math.abs(trig) > 0.02 ? trig : ry;
      if (sx !== 0 || t !== 0 || this.source === 'gamepad') { steer = -sx; thr = t; source = 'gamepad'; }
      const map = { 0: 'confirm', 2: 'reset', 3: 'camera', 9: 'pause', 1: 'reset' };
      for (const i in map) {
        const down = b(+i) > 0.5;
        if (down && !this.padPrev[i]) { this.pressed.add(map[i]); source = 'gamepad'; }
        this.padPrev[i] = down;
      }
    }
    // touch
    const ts = this.touch.steer, tt = this.touch.throttle;
    if (ts || tt) {
      source = 'touch';
      steer = ts ? -Math.max(-1, Math.min(1, (ts.x - ts.x0) / 70)) : 0;
      thr = tt ? Math.max(-1, Math.min(1, -(tt.y - tt.y0) / 80)) : 0;
    }
    // keyboard (digital: ramp the steering so taps are usable)
    const K = this.keys;
    const kl = K.has('ArrowLeft') || K.has('KeyA'), kr = K.has('ArrowRight') || K.has('KeyD');
    const ku = K.has('ArrowUp') || K.has('KeyW'), kd = K.has('ArrowDown') || K.has('KeyS');
    const tgt = (kl ? 1 : 0) - (kr ? 1 : 0);
    this.keySteer += Math.max(-dt * 4, Math.min(dt * 4, tgt * 0.75 - this.keySteer));
    if (!tgt && Math.abs(this.keySteer) < 0.02) this.keySteer = 0;
    if (kl || kr || ku || kd || (source === null && this.keySteer !== 0)) {
      source = 'keys'; steer = this.keySteer; thr = (ku ? 0.85 : 0) - (kd ? 1 : 0);
    }
    if (source) this.source = source;
    this.steer = source === 'keys' ? steer : expo(steer, this.expo);
    this.throttle = thr;
  }

  take(action) { const had = this.pressed.has(action); this.pressed.delete(action); return had; }
}
