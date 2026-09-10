# Multi-training-seed sweep, v2: seed 110, genuine network initialization (2026-09-08)

Re-run of the 5-seed sweep from `experiments/2026-09-08_seed-sweep-909/`
(and siblings) with the `seed=args.seed` fix in `scripts/train_sac.py`
applied -- see `docs/lab_notebook.md`'s 2026-09-08 entries for why the
original sweep never actually varied network initialization (a bug, not
a design choice). Same config otherwise: n_step=3, obstacle lidar in the
observation, no robot-proximity term, races=40, round_seconds=120,
buffer_capacity=800000.

**Result:** the best of the 5 genuinely-initialized seeds. 0/20
eliminated, 20/20 wins, 6.05 avg laps, 17.96s avg best lap time, 30.6 m/s
avg max speed, 0.0045 avg damage / 0.13s off-track / 0.08s wall-contact
(small but nonzero -- still very good). Close to but not quite better
than the currently-packaged `2026-09-08_seed-sweep-1000` checkpoint
(17.19s, 6.50 laps, 0.0013 damage) -- see the lab notebook entry for the
full 5-seed comparison and why seed 1000 (last sweep's winner) is now
the worst performer of this sweep under its own real initialization.
