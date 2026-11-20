"""Phase 4.7 E3: out-lap maps of the turns on the truggy physics (Phase 4.6's maps were made on the old physics).

    python -m sdc.phase47.e3_maps    -> runs/p47-maps-{train,val,select}.json

Same recipe as sdc/phase46/maps.py (the out-lap stand-in drives one slow lap; the owner's dead reckoning makes the
map), with the truggy stand-in settings (teacher.OUTLAP_TRUGGY):
  train    practice loops, start 0, training range (bc_loop's copying)
  val      validation loops, 3 starts, gate range (bc_loop's per-round scores)
  select   E2's 50 selection loops, 3 starts, gate range (E3's scoring)
"""

from __future__ import annotations

import json
from pathlib import Path

from sdc.jumps import driver as dr
from sdc.phase46 import common as cm
from sdc.phase46.maps import collect
from sdc.phase47 import e2_prep

OUT = {k: Path(f"runs/p47-maps-{k}.json") for k in ("train", "val", "select")}


def main():
    select = json.loads(e2_prep.OUT.read_text())["split"]["select"]
    for name, courses, starts, d in (("select", select, dr.STARTS, "gate"), ("val", cm.VAL_SEEDS, dr.STARTS, "gate"),
                                     ("train", cm.PRACTICE, (0.0,), "train")):
        if OUT[name].exists():
            continue
        maps = collect(courses, starts, d, procs=4, real="truggy")
        OUT[name].write_text(json.dumps(maps))
        print(name, len(maps), "maps;", sum(m["reason"] == "finished" for m in maps.values()), "out-laps finished",
              flush=True)


if __name__ == "__main__":
    main()
