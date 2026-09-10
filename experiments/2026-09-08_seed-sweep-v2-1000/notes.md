# Multi-training-seed sweep, v2: seed 1000, genuine network initialization (2026-09-08)

Part of the v2 sweep (see `2026-09-08_seed-sweep-v2-110/notes.md` and
`docs/lab_notebook.md`'s 2026-09-08 entries for the full context and
5-seed comparison). Same config as every other v2 sweep run.

**Result: the worst performer of this sweep, a striking reversal.**
Seed 1000 was the winner of the original (buggy, shared-initialization)
sweep -- 17.19s avg lap time, the checkpoint currently packaged in
`race_faster.py`. With its own genuine network initialization, it is now
the *slowest* of the 5 seeds tested: 48.86s avg best lap time, 8.9 m/s
avg max speed, only 2.00 avg laps. Still fully safe (0/20 eliminated,
0.0000 damage, 0.00s off-track/wall-contact, 20/20 wins) -- just slow,
not broken. Direct evidence that the seed argument's effect on network
initialization matters as much or more than its effect on training
trajectory (spawn order / replay sampling), and that "seed 1000" as a
label carries no special meaning across the fix.
