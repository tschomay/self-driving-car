"""Phase 4 after E1: sim review + steering debug report -> reports/phase4-debug.html"""

from pathlib import Path

import torch

from sdc.jumps import render
from sdc.phase4 import report as rp
from sdc.phase4 import train as tr

OUT = Path("reports/phase4-debug.html")
ROWS = [  # measured 2026-11-13 (steer probe, 10 validation courses, flat stretches; landing: 180 validation jumps)
    ("E1 seed 0 (no heading rays, std ≈ 1)", "1.37", "98%", "36%"),
    ("heading rays only", "1.88", "98%", "55%"),
    ("noise schedule only (std 0.5 → 0.1)", "0.32", "54%", "46%"),
    ("heading rays + noise schedule, seed 0", "0.25", "42%", "57%"),
    ("heading rays + noise schedule, seed 1", "0.10", "27%", "48%"),
    ("hand-coded expert", "0.00", "–", "96%"),
]


def main():
    torch.set_num_threads(1)
    figs = []
    for run, who in [("runs/p4-d-fan-s0", "heading rays, std ≈ 1"), ("runs/p4-d-fannoise-s1", "heading rays + noise schedule")]:
        pol, _ = tr.load(run)
        p = Path(run) / "debug-10000.gif"
        ep = render.gif(pol, 10000, p, who)
        figs.append(rp.figure(p, f"{who}: course 10000, one run, real speed, fixed renderer. Watch the steering gauge. "
                                 f"{ep['reason']}, landed {ep['landed']}/3."))
    sections = [
        """<h2>1. The sim review</h2><div class="note">
<p><b>"Jumping on the flats" was mostly the GIF.</b> It drew the ground 2–8 cm too low (it used the smooth course plan,
not the real bumpy surface) and stretched the height 1.75×, so there was always a gap under the wheels.
Measured in the sim, the wheels never leave the ground on the flats at any speed, even with full-lock sawing.
Real flat-ground air is limited to bounces right after a landing and hitting the side banks. Fixed: the GIFs now draw the real
surface at one scale for both axes.</p>
<p><b>Other fixes from the review:</b> the tires floated 1 cm above the ground (a MuJoCo setting that doesn't work in
this version); the expert's landing guess used the same wrong ground height; the accelerometer was a single noisy 2 ms
sample (now averaged over each 20 ms step, like a real IMU's filter); "off the lane" now ends the run when the car's
centre crosses the lane edge, not after it has climbed the bank. The expert lands 96% after the fixes.</p>
<p><b>Checked and fine:</b> steering servo (about 0.1 s per 60°, like a standard RC servo), rays, gyro, motor, top speed, braking,
rollover threshold (it slides before it tips).</p></div>""",
        """<h2>2. Why it saws</h2><div class="note">
<p><b>The exploration noise.</b> Training adds random noise to every action, 50 times a second, at a spread of about 1
(the whole steering range). The servo can't follow a 50 Hz signal, so the sawing costs nothing, and the policy learns
a bang-bang style that fights the noise. Same lesson as Phase 3: a noise schedule (spread 0.5 shrinking to 0.1) cuts the
step-to-step steering change by 5–14×.</p>
<p><b>The car couldn't see which way it was pointing.</b> Its rays looked straight ahead and straight sideways; those read
the same whether the nose points 9° left or 9° right. With only the sensors, even the expert's own steering law weaves off
the lane. A fan of 6 level rays (20°, 35°, 60° left and right) shows the lane edges ahead. That added about 10–15 points of landings
on its own, but didn't stop the sawing.</p>
<p>Your guess about chasing an exact centre line: there is no centre-line target at all (the reward is metres of progress
plus jump bonuses). The weaving came from not knowing the heading, the sawing from the noise. The policy already sees its
last steering and throttle command. It doesn't see the servo's actual angle; that could be added.</p></div>""",
        "<h2>3. Measured</h2>" + rp.table(["driver", "steering change per step", "sign flips", "landed"], [list(r) for r in ROWS])
        + "<p class='caption'>Steering change: mean |steer now − steer last step| on the flats (range −1…1). "
          "Landed: 180 validation jumps; runs trained before the sim fixes.</p>",
        "<h2>4. See it</h2><div class='gifs'>" + "".join(figs) + "</div>",
    ]
    rp.page("Phase 4 steering debug", "Phase 4 · after E1 · sim review and debugging", "Why it saws, and the floating car",
            "Is the sim at fault, and why does the steering saw?", sections, OUT,
            footer="Code: sdc/jumps/ (fixes), sdc/phase4/debug_report.py")
    print(OUT)


if __name__ == "__main__":
    main()
