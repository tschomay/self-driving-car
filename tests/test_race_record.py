"""Race replay recorder (sdc/race/record.py): a short teacher race records frames, laps and a viewer track."""

import numpy as np

from sdc.race import record as rc


def test_two_lap_teacher_race():
    r = rc.race(["teacher"], 10001, laps=2, log=lambda *a: None)
    assert len(r["lap_times"]) == 2 and not r["dnf"]
    assert abs(r["race_time"] - (sum(r["lap_times"]))) < 0.05
    assert r["race_time"] >= r["crashes"] * rc.CRASH_S
    n = r["n"]
    assert all(len(v) == n for v in r["cols"].values())
    assert abs(n / rc.FPS - r["race_time"]) < 0.2               # frames are on race time
    q = np.array([r["cols"][k] for k in ("qw", "qx", "qy", "qz")]) / r["scales"]["qw"]
    assert np.allclose(np.linalg.norm(q, axis=0), 1, atol=2e-3)
    laps = [e for e in r["events"] if e["type"] == "lap"]
    assert [e["lap"] for e in laps] == [1, 2]


def test_track_export():
    t = rc.track(10001)
    hf = t["hf"]
    assert len(hf["mm"]) == hf["nx"] * hf["ny"]
    assert len(t["center"]) == len(t["profile"]) == len(t["bank"])
    assert abs(len(t["center"]) * t["ds"] - t["length"]) < 0.5
