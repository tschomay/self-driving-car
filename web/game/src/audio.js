// Sound: the whine of a brushless motor and gears (pitch follows wheel speed), tire noise on dirt, landing thumps.
export class Sound {
  constructor() { this.ctx = null; this.on = true; }

  start() {
    if (this.ctx) { this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = this.ctx = new AC();
    this.master = ctx.createGain(); this.master.gain.value = this.on ? 0.5 : 0; this.master.connect(ctx.destination);
    const mk = (type) => { const o = ctx.createOscillator(); o.type = type; o.start(); return o; };
    this.o1 = mk('sawtooth'); this.o2 = mk('square');
    this.lp = ctx.createBiquadFilter(); this.lp.type = 'lowpass'; this.lp.Q.value = 2;
    this.motorGain = ctx.createGain(); this.motorGain.gain.value = 0;
    const g2 = ctx.createGain(); g2.gain.value = 0.25;
    this.o1.connect(this.lp); this.o2.connect(g2); g2.connect(this.lp); this.lp.connect(this.motorGain); this.motorGain.connect(this.master);
    const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate), ch = buf.getChannelData(0);
    for (let i = 0; i < len; i++) ch[i] = Math.random() * 2 - 1;
    this.noiseBuf = buf;
    const n = ctx.createBufferSource(); n.buffer = buf; n.loop = true; n.start();
    this.dirtF = ctx.createBiquadFilter(); this.dirtF.type = 'bandpass'; this.dirtF.frequency.value = 900; this.dirtF.Q.value = 0.6;
    this.dirtGain = ctx.createGain(); this.dirtGain.gain.value = 0;
    n.connect(this.dirtF); this.dirtF.connect(this.dirtGain); this.dirtGain.connect(this.master);
  }

  setOn(on) { this.on = on; if (this.master) this.master.gain.value = on ? 0.5 : 0; }

  update(wheelSpeed, throttle, speed, slip, grounded) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime, w = Math.abs(wheelSpeed);
    const f = 70 + w * 62;
    this.o1.frequency.setTargetAtTime(f, t, 0.03); this.o2.frequency.setTargetAtTime(f * 2.01, t, 0.03);
    this.lp.frequency.setTargetAtTime(500 + w * 260 + Math.abs(throttle) * 900, t, 0.05);
    this.motorGain.gain.setTargetAtTime(Math.min(0.2, 0.012 + 0.006 * w + 0.07 * Math.abs(throttle)), t, 0.05);
    this.dirtGain.gain.setTargetAtTime(grounded ? Math.min(0.5, 0.012 * speed + 0.09 * Math.min(slip, 4)) : 0, t, 0.06);
    this.dirtF.frequency.setTargetAtTime(500 + speed * 110, t, 0.1);
  }

  thump(strength) {
    if (!this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime, n = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    n.buffer = this.noiseBuf; f.type = 'lowpass'; f.frequency.value = 260;
    g.gain.setValueAtTime(Math.min(1, 0.25 + strength * 0.25), t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.22);
    n.connect(f); f.connect(g); g.connect(this.master); n.start(t, Math.random()); n.stop(t + 0.25);
  }

  beep(freq, dur = 0.12) {
    if (!this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime, o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = freq; g.gain.setValueAtTime(0.25, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(this.master); o.start(t); o.stop(t + dur);
  }

  quiet() { if (this.ctx) { const t = this.ctx.currentTime; this.motorGain.gain.setTargetAtTime(0, t, 0.05); this.dirtGain.gain.setTargetAtTime(0, t, 0.05); } }
}
