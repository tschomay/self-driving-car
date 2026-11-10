# Phase 3 sim review: physics, controls, sensors (2026-11-10)

A background code review requested by the owner after the Phase 3 gate. It covers `sdc/racer/dynamics.py` at realism 2, as used in E4–E10 and the gate. The review edited no files. The agent spot-checked finding 3: on dirt at nominal grip, a 0.25 steer at full throttle gives a yaw rate of 0.06 rad/s (both axles "spinning"), vs. 0.23 at 85% throttle.

## Findings, most important first

1. **Bug: the low-speed blend wipes out sideways motion and spin.** `dynamics.py`, blend toward the kinematic model.
   - The blend factor uses `vx` alone, so a car sliding sideways or spinning at low forward speed is snapped back to the no-slip model.
   - From vx=0.5, vy=8, r=2: one 10 ms substep gives vy=0, r=0.
   - The blend runs on every substep, so it compounds: the 1.5–3 m/s ramp acts like a switch at about 2.9 m/s.
   - Fix: blend on |v|, blend forces rather than states, and make it independent of the substep length.
2. **Bug: a locked or saturated front tire pushes along the wheel's heading.**
   - Steering full left with full brake at 15 m/s turns the car right (yaw −0.26 rad/s).
   - A locked axle has zero sideways force, so a sliding car keeps sliding.
   - Fix: a sliding tire's force should oppose the contact patch's velocity, at SLIDE_FRAC × peak.
3. **Realism gap: a fixed drive and brake force creates a grip cliff.**
   - Drive and brake forces scale with nominal mass, but tire grip scales with grip × mass × dirt.
   - Full throttle (below 14 m/s) therefore spins both axles whenever grip × mass × dirt < 0.72. That includes all dirt at nominal physics and the low end of the training range.
   - Spinning axles give zero sideways force, so the car stops turning.
   - This likely explains much of the low-grip and light-car failures, and the RL cars' avoidance of full throttle.
   - Fix: model wheel slip ratio (or a smooth combined-slip curve), and keep partial sideways grip when a wheel spins.
4. **Realism gap: sawing the steering is nearly free.**
   - Three seconds at 15 m/s flipping ±1 every step: the servo and tire relaxation shrink it to ±7.5° at the wheel. The car covers 44.39 m vs. 45.00 m driving straight, and at full throttle the hard 25 m/s cap hides even that.
   - Missing: aero drag, rolling resistance, load transfer, tire load sensitivity, and any steering cost. PPO's step-to-step exploration noise is uncorrelated, so training practises sawing.
   - Suggested: drag and rolling resistance instead of the speed cap; load transfer; correlated exploration (gSDE) or a smaller initial noise; steering-rate actions.
5. **Partial observability: the actual wheel angle (servo position) is never observed.** The last command is the right one for delays up to 0.1 s, but above that a second command is also in flight. Suggested: add servo feedback.
6. **Cornering stiffness doesn't scale with load.** The tire's peak slip angle ranges from 4.7° (grip 0.5) to 12.1° (grip 1.3), so light cars feel twitchy and heavy ones soft. Suggested: C_ALPHA proportional to axle load.
7. **Relaxation length (0.5 m) makes low-speed steering bouncy.** Yaw overshoot is 36% at 4 m/s. 0.5 m is also about 10× too long for a 1/10 RC tyre.
8. **Nitpick:** at top speed under full throttle, the accelerometer reports acceleration while the speed clip discards it.
9. **Nitpick:** the speed sensor is ideal (a real wheel encoder misreads a locked or spinning wheel).
10. **Nitpick:** clipping `vx` at 0 deletes backward momentum during spins; the off-track check uses the centre of mass only.
11. **Stale docstring:** the peak is at 9.3°, not 6–7°. The force is still 92% of peak at 20° of slip and 83% at 40°, so sliding past the peak costs little.
12. **Real car:** 10 Hz control at 1/10 scale is like about 3 Hz at full size. Plan for 30–50 Hz on the RC car.

## Confirmed correct

- Body-frame equations and slip-angle signs.
- Static axle loads and yaw inertia.
- The magic-formula slope at small slip.
- The friction circle.
- The power limit.
- No reverse.
- Midpoint position integration (stable down to 0.65× mass).
- Delay timing.
- The last-command observation and the tanh squash, consistent between training and evaluation.
- History ordering.
- Rays, progress and lateral offset all from the same origin.
- Progress is measured along the centerline (weaving earns nothing; the inside line is rewarded).
- Gyro scale and clipping.
- The oracle grip sensor.
