# Multi-training-seed sweep: seed 1000 (2026-09-08)

Part of the first genuine multi-training-seed sweep on this track (see
`docs/lab_notebook.md`'s 2026-09-08 entry for the full 5-seed comparison
and analysis). Locked-in config identical to the `seed-sweep-909` entry;
seed 1000 is a fresh training seed never used on this track before.

**Result:** the best-performing seed in the sweep. 0/20 eliminated, 20/20
wins, 0.0013 avg damage, 0.02s off-track, 0.00s wall-contact, 6.50 avg
laps (highest in the sweep), 17.19s avg best lap time (fastest in the
sweep, tied with seed 3000), 31.0 m/s avg max speed. Combines the best
safety profile of the two fast seeds (1000, 3000) with matching pace --
see the lab notebook entry for the full cross-seed table and the
recommendation to use this checkpoint for `controllers.race_faster`.
