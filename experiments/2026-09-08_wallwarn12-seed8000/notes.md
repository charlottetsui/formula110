# Doubling WALL_WARNING_DISTANCE_M -- fixes the near-miss, costs most of the pace gain (2026-09-08)

Directed to improve safety from the newly-adopted
`2026-09-08_seed-sweep-v2-8000` checkpoint (packaged in `race_faster.py`),
which has a documented near-miss: 1/20 evaluation races took 0.5942
damage from a hard wall impact while accelerating through a corner
(diagnosed in `2026-09-08_seed-sweep-v2-8000/notes.md`).

Traced the near-miss to a stale calibration rather than guessing at a
new mechanism: `WALL_WARNING_DISTANCE_M = 6.0` was explicitly tuned
(2026-09-01) for "~0.6s at 10 m/s" reaction time, back when checkpoints
cruised near that speed. This checkpoint reaches ~27 m/s, where 6.0m
gives only ~0.22s -- well under half the originally intended margin, and
consistent with the diagnosed event (wall clearance shrank from 3.90m to
1.30m in ~0.4s, faster than the mechanism had lead time to correct for).
Doubled `WALL_WARNING_DISTANCE_M` to 12.0 (matching the relative size of
the 3.0->6.0 step that worked the first time) and trained fresh on seed
8000 -- the exact seed that produced the near-miss -- otherwise identical
config (n_step=3, obstacle lidar, no robot-proximity, races=40,
round_seconds=120, buffer_capacity=800000).

## Result: the near-miss is gone, but so is most of the pace gain

| | seed 8000, warn=6.0 (packaged) | seed 8000, warn=12.0 |
| --- | --- | --- |
| avg damage | 0.0298 | **0.0000** |
| avg off-track | 0.152s | **0.003s** |
| avg wall-contact | 0.107s | **0.000s** |
| avg laps | 7.30 | **3.35** |
| avg best lap time | 14.96s | **31.46s** |
| avg max speed | 26.65 m/s | 18.74 m/s |
| avg scored distance | 1444.6m | 710.2m |
| eliminated | 0/20 | 0/20 |

Doubling the warning distance essentially eliminated the near-miss --
but the resulting checkpoint is a genuinely different, much more
conservative policy, not a marginally-safer version of the fast one.
Laps dropped by more than half, lap time more than doubled, top speed
dropped by a third.

## Read

This is the same "safety fix overcorrects into a much slower policy"
pattern seen twice already this session with `WEIGHT_ROBOT_PROXIMITY`
(both the unrestricted and angle-restricted attempts). Across three
consecutive safety-motivated reward changes now, the outcome has been
either a clear regression in pace or (for the angle-restricted
competitor case) a near-total collapse -- there is no evidence yet of a
reward-shape lever on this axis that trades a *small* amount of pace for
a *large* amount of safety; the trades found so far are closer to
all-or-nothing. Training appears to converge to qualitatively different
equilibria (aggressive-and-risky vs. cautious-and-slow) rather than
smoothly interpolating between them as constants are tuned.

## Decision and rationale

Reverted `WALL_WARNING_DISTANCE_M` to 6.0. Not adopting this checkpoint
-- it solves the stated problem but at a cost that erases most of why
`seed-sweep-v2-8000` was adopted in the first place.
`2026-09-08_seed-sweep-v2-8000` remains packaged in `race_faster.py`,
near-miss and all.

## Next steps

1. Given three consecutive reward-shape safety attempts have each
   produced an all-or-nothing trade rather than a middle ground, the
   evidence favors a different strategy: keep sampling genuine-init seeds
   (the approach that already found `seed-sweep-v2-8000` itself) looking
   specifically for one that matches or beats its pace *without* a
   near-miss in its own evaluation, rather than continuing to search the
   reward-magnitude axis for a middle ground that may not exist here.
2. A controller-level (not reward-level) hard safety backstop -- e.g.
   overriding throttle when a forward wall reading is both very close and
   speed is high, regardless of what the learned policy outputs -- was
   proposed early in this project's refinement plan and remains untested;
   it wouldn't have the "policy learns to avoid the situation entirely"
   side effect that reward shaping keeps producing, since it doesn't
   change what's being optimized, only adds a deterministic exception.
3. Still open: the `controllers.minimum_viable` module gap.
