"""The browser apps follow the sim: web/shared/generated/sim.json is current, and the game's MuJoCo port drives like it.

If test_sim_json_is_current fails: run `python -m sdc.export` and commit web/shared/generated/sim.json (and port any
new physics switch to web/game/src: see AGENTS.md).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sdc import export

ROOT = Path(__file__).resolve().parents[1]


def test_sim_json_is_current():
    assert export.OUT.exists(), "run python -m sdc.export"
    assert export.OUT.read_text() == export.text(), "web/shared/generated/sim.json is stale: run python -m sdc.export"


@pytest.mark.skipif(shutil.which("node") is None or not (ROOT / "web" / "node_modules" / "@mujoco").exists(),
                    reason="needs node and `npm install` in web/")
def test_game_matches_sim_for_the_first_second(tmp_path):
    """Same scripted commands through the sim and the game's MuJoCo (WebAssembly). They agree to millimetres until
    the car reaches the first jump (about 1.2 s), after which tiny differences grow by chaos."""
    traj = tmp_path / "py.json"
    traj.write_text(subprocess.run([sys.executable, "web/game/tools/py_parity.py"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout)
    out = subprocess.run(["node", "test/parity.mjs", str(traj)], cwd=ROOT / "web" / "game", check=True,
                         capture_output=True, text=True, timeout=600).stdout
    gap = {float(t): float(g) for t, g in re.findall(r"t ([\d.]+) s .* position gap ([\d.]+) mm", out)}
    assert gap[0.5] < 1.0 and gap[1.0] < 10.0, out
