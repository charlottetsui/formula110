# Competitor-proximity reward, a regression -- reverted (2026-09-07/09-08)

Proposed next step after the obstacle-lidar observation change
(`2026-09-07_obstacle-lidar-nstep3-seed110`) showed a real average
improvement in car-contact/marshal rate but left one severe outlier where
the car got stuck against a moving opponent. Added `WEIGHT_ROBOT_PROXIMITY`
to `src/training/reward.py`: the direct structural analog of
`WEIGHT_WALL_PROXIMITY`, penalizing proximity to the nearest reading from
`camera.competitors` (ramping from 0 at `ROBOT_WARNING_DISTANCE_M = 8.0`
to a max at 0m, scaled by the same speed-risk multiplier as wall
proximity), intended to teach proactive avoidance before contact rather
than only reacting to it via the existing `WEIGHT_CONTACT`. No angle
restriction on the nearest-competitor reading (flagged explicitly at
implementation time as a possible weak point). Same seed (110), races=40,
round_seconds=120, buffer_capacity=800000, n_step=3, and observation
(with obstacle lidar) as the reference -- the new reward term is the only
changed variable. Verified with 171 passing tests and a real smoke race
before running the full experiment.

## Result: a regression, and the specific problem it targeted got worse

| | + obstacle lidar (reference) | + robot-proximity reward |
| --- | --- | --- |
| avg damage/off-track/wall-contact | 0.0027 / 0.027s / 0.009s | 0.0000 / 0.000s / 0.000s |
| avg car-contact | 0.952s | 0.825s (mild improvement) |
| **max car-contact in any race** | 2.68s | **6.82s (worst seen in any run)** |
| avg laps | 4.00 | **2.90** |
| avg best lap time | 24.79s | **35.31s (+42%)** |
| avg marshal/race | 0.10 | **0.55 (5.5x worse)** |
| avg max speed | 23.24 m/s | 17.40 m/s |

The average car-contact number looks like a mild win, but per-race detail
shows it isn't: `seed=110 vs crash_fast` (both races, reproducibly, not a
fluke -- the exact scenario this term was meant to fix, a stationary
mid-track blocker) rose to 6.3-6.8s of car-contact with 2 marshal
recoveries and only 2 laps completed, worse than the outlier this term
was built to prevent. Laps dropped broadly across nearly every race
(2-3, down from 4-5), and max speed converged to a suspiciously uniform
~17.3-17.7 m/s everywhere (previously 14.7-23+ m/s, situation-dependent)
-- a broad, uniform slowdown rather than a targeted fix.

## Read

Likely mechanism: with no angle restriction, the penalty fires for *any*
nearby competitor regardless of whether it's actually in the way (beside,
behind, or on a different part of a switchback), teaching generalized
caution around any competitor rather than specifically avoiding
collisions. For a stationary blocker mid-track, that generalized caution
plausibly makes it *harder* to commit to a clean pass, not easier --
consistent with the seed=110/crash_fast result getting worse, not better.
This matches the established pattern on this track of a plausible-
sounding avoidance/caution term backfiring into broad overcaution
(`WEIGHT_STEERING_SMOOTHNESS`, the idle-penalty overcorrection, the
`WEIGHT_CENTER_OFFSET` halving attempt) rather than a targeted fix.

## Decision and rationale

Not adopted. Disabled `WEIGHT_ROBOT_PROXIMITY` (weight 0.0, mechanism
kept in code) -- same convention as the other two reverted terms in
`src/training/reward.py`. `2026-09-07_obstacle-lidar-nstep3-seed110`
remains the reference checkpoint.

## Next steps

1. An angle-restricted variant (mirroring `WALL_WARNING_BEAM_ANGLES_DEGREES`'s
   front-only beams) is untested and may not have this failure mode --
   only penalize a competitor reading within some forward cone
   (e.g. |angle_degrees| < 45-60), not any nearby competitor regardless
   of direction.
2. A closing-speed-scaled variant (using `closing_speed_mps` from
   `CameraCompetitorReading` instead of own absolute speed) is also
   untested -- this would only penalize proximity when actually gaining
   on the competitor, not merely being near one at any relative speed.
3. Given two consecutive proximity-based hesitation fixes (steering-
   reversal, then this) have not been the answer, consider whether the
   obstacle-lidar observation alone (already adopted, real improvement on
   19/20 races) is sufficient for now, and revisit competitor-avoidance
   reward shaping later with more diagnostic detail on the seed=110/
   crash_fast case specifically.
4. Still open: a genuine multi-training-seed sweep, and repackaging
   `controllers.race_faster`.
