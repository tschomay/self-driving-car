"""Shared Phase 4.5 settings."""

from sdc.jumps import course as crs

# The Phase 4.5 teacher (tuned 2026-11-14 on validation courses, all physics switches, gate-range physics):
# each jump's design speed + 0.3 m/s (the damper soaks up some of the kicker), and twice the in-air pitch
# damping (kd 1.0: with up to 45 ms of control delay the Phase 4 gains overshoot to 20-33° nose down).
EXPERT = dict(plan=True, plan_offset=0.3, kd=1.0)
EXPERT_STEADY = dict(plan=False, v_target=7.4, kd=1.0)   # one speed for every jump (what Phase 4 copied)
EXPERT_P4 = dict(plan=False, v_target=7.1)               # Phase 4's teacher, unchanged

TEST_SEEDS = list(range(45000, 45020))   # gate only
PHYS_SEED_VAL = 0
PHYS_SEED_TEST = 45                      # physics/noise stream for the gate (never used by experiments)

assert not set(TEST_SEEDS) & (set(crs.VAL_SEEDS) | set(crs.TEST_SEEDS))
