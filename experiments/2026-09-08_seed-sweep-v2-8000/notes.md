# Genuine-init seed sampling: seed 8000 -- new pace record, with a flagged outlier (2026-09-08)

Part of a batch of 5 fresh genuine-init seeds sampled in search of a
checkpoint beating the 17.19s record (`2026-09-08_seed-sweep-1000`), per
direction after causal test 26 confirmed real network-initialization
diversity carries substantial pace variance. Same locked-in config as
every other genuine-init sweep run (n_step=3, obstacle lidar, no
robot-proximity term, races=40, round_seconds=120, buffer_capacity=800000).

## Result: a new record, and a genuinely fast, mostly excellent checkpoint

| | previous record (v1 seed 1000) | seed 8000 |
| --- | --- | --- |
| avg best lap time | 17.19s | **14.96s** |
| avg laps | 6.50 | **7.30** |
| avg scored distance | 1267.2m | **1444.6m** |
| avg max speed | 31.0 m/s | 26.7 m/s |
| avg damage | 0.0013 | 0.0298 |
| avg off-track | 0.02s | 0.15s |
| avg wall-contact | 0.00s | 0.11s |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

The aggregate damage/off-track/wall-contact numbers look worse than the
previous record, but per-race detail shows this is driven almost
entirely by **one outlier**, not a general safety regression: 18 of 20
races are exceptionally clean (0.0000 damage, 7-8 laps every single
time, 25.5-27.7 m/s max speed -- faster *and* more consistent than the
previous record on the races where nothing goes wrong). One race
(`seed=8675309 vs default_student_controller race=1`) took **0.5942
damage** -- a serious near-crash, well above every other race's damage
in this run and close to (though short of) the `NEAR_ELIMINATION_DAMAGE
= 0.9` terminal threshold -- with 2.68s off-track, 1.88s wall-contact,
2 marshal recoveries, and only 6 laps (still completed, lower than the
typical 7-8). A second, much milder blip (`seed=110 vs
default_student_controller race=2`: 0.0026 damage, 0.35s off-track,
0.25s wall-contact, 1 marshal) is negligible by comparison.

## Read

This checkpoint is a real, substantial pace improvement (13% faster lap
time, +0.8 laps per race) with excellent safety in the large majority of
races, but a genuine, non-trivial risk in a rare case (1/20, tied to one
specific seed/baseline combination) that came close to a real crash.
Not a clean, unambiguous win the way earlier accepted-with-caveats
checkpoints were (e.g. the obstacle-lidar checkpoint's outlier was a
stuck-and-slow failure, not a near-elimination-level damage event) --
this is a sharper trade-off between "meaningfully faster" and "one real
near-miss."

## Decision and rationale

Recommending adoption as the new best/reference checkpoint given the
scale of the pace improvement and that the outlier, while more severe in
kind than prior flagged outliers, still did not result in elimination or
a lost race -- but flagging this explicitly as a judgment call rather
than an obvious choice, unlike every prior "adopt" decision this session.
Awaiting confirmation before repackaging `controllers.race_faster` from
this checkpoint, since that affects the actual submission artifact.

## Outlier diagnosis

Re-ran the exact seed/baseline/race with per-tick logging (`contact.robot`
was not involved -- `robot_contact=0.00` throughout, ruling out the
opponent-collision failure mode from earlier sessions). Found a clean,
coherent single event: over ticks 3612-3642 (~0.5s), the car accelerated
hard (8.8 -> 14.1 m/s) while drifting increasingly off-center
(`center_offset_m` -0.20m -> -1.96m) as wall clearance shrank (`wall_min`
3.90m -> 1.30m) -- consistent with committing to a fast line through a
tightening corner. Damage jumped from 0.0 to 0.548 in a single tick
(3637 -> 3642), i.e. one hard wall impact, not a gradual scrape or a
repeated pattern. The car recovered afterward (speed settled near zero
then briefly negative -- reversing off the wall) without further damage
accumulation, and completed the race. This is not a bug and not the
opponent-collision failure mode from earlier in this session -- it's a
genuine speed-vs-cornering-margin risk inherent to a faster policy,
occurring in 1 of 20 evaluated races.

## Next steps

1. If adopted, repackage `controllers.race_faster` from this checkpoint.
2. Continue sampling more genuine-init seeds if an even better,
   cleaner-margin checkpoint is wanted instead of accepting this
   trade-off.
3. If this trade-off recurs across future fast checkpoints, a corner-
   aware wall-proximity term (scaling with `camera.lookahead_offsets_m`
   curvature, not just current-tick distance) could give more reaction
   time than the current same-tick proximity penalty allows -- noted as
   a candidate, not yet tested.
