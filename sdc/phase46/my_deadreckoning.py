"""The owner's dead-reckoning estimator (learner-owned, Phase 4.6): where is the car on the loop, from its own sensors?

Inputs each control step (50 Hz), all from the parts-list sensors:
  gyro (rad/s, body frame; z = turning left), motor speed (m/s), and the owner's attitude filter's pitch and roll.
Output: position (x, y) in metres and heading in radians, in a frame that starts at the start line: x along the
track at the line, y to the left. It starts at (0, 0, 0) and the finish-line sensor says when a lap is complete.

The map of the turns comes from this: the heading's change per metre travelled is the track's curvature.

OWNER TODO: four choices, marked below. Until they are made, ready() is False and nothing uses this file.
"""

from __future__ import annotations

import numpy as np

DT = 0.02

# OWNER TODO 1 - heading: where does the turn rate come from?
#   "gyro"      integrate the gyro's z rate (minus its bias measured while still). Drifts slowly with the bias.
#   "steering"  from the steering command and speed (bicycle model). No drift, but wrong when the tires slide.
#   "blend"     gyro for the short term, pulled toward the steering model over a few seconds.
HEADING = None

# OWNER TODO 2 - tilt: on a banked turn or a ramp the gyro's z axis leans with the car, so it reads less than the
# true turn rate. True = use the attitude filter's pitch and roll to project the gyro onto the vertical;
# False = use the z reading as it is.
TILT_CORRECT = None

# OWNER TODO 3 - distance: motor speed x time overcounts when the wheels spin in the air or slip.
#   "raw"    motor speed x time, always.
#   "hold"   in free fall, keep the last speed measured on the ground (Phase 4.5's odometer).
DISTANCE = None

# OWNER TODO 4 - the finish line: after a lap the estimate should be back at (0, 0) facing 0, and it won't be.
#   "reset"   snap back to (0, 0, 0) at the line; the lap's map keeps its error.
#   "spread"  also go back over the lap just driven and spread the closing error evenly along it, so the mapped
#             loop closes.
AT_LINE = None


def ready():
    return None not in (HEADING, TILT_CORRECT, DISTANCE, AT_LINE)
