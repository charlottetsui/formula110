# Multi-training-seed sweep: seed 909 (2026-09-08)

Part of the first genuine multi-training-seed sweep on this track (see
`docs/lab_notebook.md`'s 2026-09-08 entry for the full 5-seed comparison
and analysis). Locked-in config: n_step=3, obstacle lidar in the
observation, no robot-proximity reward term
(`2026-09-07_obstacle-lidar-nstep3-seed110`'s reward/architecture),
races=40, round_seconds=120, buffer_capacity=800000. Seed 909 was chosen
specifically because it's the seed that produced a catastrophic "do
nothing" freeze back on 2026-09-01 under a much cruder reward -- this
re-checks whether that instability persists under the current stack.

**Result:** no longer unstable. 0/20 eliminated, 20/20 wins against both
baselines, 0.000 damage, 0.00s off-track/wall-contact, 5.00 avg laps
(actually the highest lap count of any seed in this sweep), 22.50s avg
best lap time. The 2026-09-01 instability does not recur -- see the lab
notebook entry for the full cross-seed table and discussion.
