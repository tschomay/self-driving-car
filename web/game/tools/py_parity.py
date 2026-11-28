"""The training sim's side of test/parity.mjs: the same scripted commands on the same loop, recorded per control step.

    python web/game/tools/py_parity.py [seed] > /tmp/py_traj.json      (from the repo root)
    cd web/game && node test/parity.mjs /tmp/py_traj.json [seed]

The car is the one in web/shared/generated/sim.json (the champion's sim), at nominal physics, with no command delay and
no steering expo, started as env.reset starts it. Rows: X, Y, z, forward speed, mean wheel speed (m/s).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sdc.jumps import env as je  # noqa: E402

SIM = json.loads((ROOT / "web" / "shared" / "generated" / "sim.json").read_text())
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10001
SECS = 6.0


def cmd(t):   # test/parity.mjs
    return [0, 1] if t < 1.5 else [0, -1] if t < 2.0 else [0.6, 0.7] if t < 3.5 else [0.6, 0] if t < 4.0 else [-0.3, 1]


b = je.JumpBatch(1, courses=[SEED], real=SIM["real"], sensors="parts", loop=True, dr=None, expo=0.0)
c = b.cars[0]
c.reset(SEED)
c.phys.delay = 0.0
rows = []
for k in range(int(SECS / 0.02)):
    c.step(np.array(cmd(k * 0.02), np.float32), max_steps=10**9)
    st = c.state()
    rows.append([round(v, 6) for v in (st["X"], st["Y"], st["z"], st["v_fwd"], c._mean_wheel_speed())])
print(json.dumps(rows))
