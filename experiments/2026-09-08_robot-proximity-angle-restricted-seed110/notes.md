# Angle-restricted competitor-proximity penalty -- much worse, reverted (2026-09-08)

Follow-up to the reverted unrestricted `WEIGHT_ROBOT_PROXIMITY`
(`2026-09-07_robot-proximity-obstacle-lidar-nstep3-seed110`): that version
penalized proximity to *any* nearby competitor regardless of direction,
and regressed by teaching broad, situation-blind caution. This test added
an angle restriction (`ROBOT_WARNING_ANGLE_DEGREES = 45.0`) so only a
competitor within 45 degrees of straight ahead counts -- mirroring how
`WALL_WARNING_BEAM_ANGLES_DEGREES` is front-only. Same seed (110),
races=40, round_seconds=120, buffer_capacity=800000, n_step=3, and
observation (with obstacle lidar) as both prior comparisons -- the angle
restriction is the only new variable relative to the unrestricted attempt.
Verified with 172 passing tests and a real smoke race before running the
full experiment.

## Result: much worse than the unrestricted version, not better

| | obstacle lidar only (reference) | robot-proximity, unrestricted (reverted) | robot-proximity, angle-restricted |
| --- | --- | --- | --- |
| avg laps | 4.00 | 2.90 | **0.20** |
| avg low-progress | 3.61s (3.0%) | 3.99s (3.3%) | **46.22s (38.5%)** |
| avg marshal/race | 0.10 | 0.55 | **6.35** |
| avg max speed | 23.24 m/s | 17.40 m/s | **8.30 m/s** |
| avg best lap time (n races that finished a lap) | 24.79s (20/20) | 35.31s (20/20) | 82.46s (**4/20**) |
| avg car-contact | 0.952s | 0.825s | 0.662s (best of the three) |
| eliminated | 0/20 | 0/20 | 0/20 |

Per-race detail: **17 of 20 evaluation races completed exactly zero
laps**, uniformly across every seed and both baselines (not one outlier --
e.g. `seed=42 vs crash_fast race=2` hit 119.58s of low-progress, 59
marshal recoveries, and 0.7 m/s max speed, essentially motionless for the
entire round). Car-contact time was actually the lowest of the three
configurations tested -- the car genuinely avoided the opponent -- but
only by nearly never moving.

## Read

Avoiding a wall requires active steering (there's no way to "just not
approach" a wall you're driving toward on a fixed track), so
`WEIGHT_WALL_PROXIMITY` teaches real avoidance behavior. Avoiding an
*ahead* competitor is trivially satisfiable by never closing distance at
all -- crawling at 6-10 m/s comfortably clears `WEIGHT_IDLE`'s 0.5 m/s
threshold while permanently avoiding the front-cone proximity penalty,
with no need to ever commit to a pass. The unrestricted version (broad,
situation-blind caution, still moving) and this restricted version (a
clean escape hatch: just never approach) are different failure modes, not
two points on a spectrum from bad to good.

## Decision and rationale

Not adopted. Disabled `WEIGHT_ROBOT_PROXIMITY` again (weight 0.0, both
the base mechanism and the angle restriction kept in code, since the
angle-filtering logic itself is verified correct via a direct test --
it's the policy's response to the resulting incentive that failed, not a
bug). Not tuning the angle/distance further without a different
underlying idea, given two consecutive attempts on this specific
mechanism have each failed in a different, novel way.
`2026-09-07_obstacle-lidar-nstep3-seed110` (obstacle lidar in the
observation, no robot-proximity reward term at all) remains the reference
checkpoint and is the config locked in for the next step: a genuine
multi-training-seed sweep, since every result on this track to date comes
from a single training seed (110).

## Next steps

1. If competitor-proximity reward shaping is revisited later: a
   closing-speed-scaled variant (`closing_speed_mps` instead of own
   speed) is still untested and may not have an equivalent "just stay far
   away forever" escape hatch, since it only fires while actually gaining
   -- though a similar "never gain on them" strategy could plausibly
   still emerge.
2. Proceeding to the multi-training-seed sweep (locked-in config: n_step=3,
   obstacle lidar, no robot-proximity term) as the next step, per
   direction -- see `docs/lab_notebook.md`'s 2026-09-08 entry.
