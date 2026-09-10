# Best-trajectory reward bonus, timing bug fixed (2026-09-02)

Follow-up to `2026-09-02_trajectory-bonus-seed110`, which found a severe
regression caused by a same-tick, same-race cross-copy bug in
`BestTrajectoryTracker`'s update timing. Fixed by decoupling reads from
writes: each episode now takes a frozen `snapshot()` of the tracker at its
own first tick and reads bonuses from that snapshot for its whole run,
while `update()` still writes to the live tracker for *future* episodes
(see `src/training/trajectory.py`'s updated module docstring and
`src/training/controller.py`'s snapshot wiring). Same seed (110),
races=40, round_seconds=120, buffer_capacity=800000, `--trajectory-bonus`
as the buggy run.

## Result: the bug is fixed, but the mechanism still doesn't beat the reference

| | races=40 reference (no bonus) | trajectory-bonus, buggy | trajectory-bonus, fixed |
| --- | --- | --- | --- |
| avg damage | 0.000 | 0.337 | 0.111 |
| eliminated | 0/20 | 3/20 | **2/20** |
| avg off-track | 0.00s | 3.07s | 0.11s |
| avg wall contact | 0.00s | 2.64s | 0.09s |
| avg marshal/race | 0.15 | **22.25** | 0.50 |
| avg laps | 4.10 | 0.05 | **2.70** |
| avg best lap time | 24.76s | 99.32s | **32.73s** |
| avg max speed | 15.53 m/s | 14.76 m/s | 17.33 m/s |
| self-play training distance | -- | 3.0m/0.0m (40 races) | 23,603m/25,141m (40 races) |

The fix clearly worked: marshal recoveries dropped from 22.25/race back to
0.50/race (near the reference's 0.15), off-track and wall-contact time are
back near zero, and self-play's own training distance is back to a normal
order of magnitude. This is not the same broken, unstable training run
anymore.

But the fixed mechanism still doesn't beat the plain races=40 reference:
fewer laps (2.70 vs. 4.10), slower lap times (32.73s vs. 24.76s), and 2 of
20 races still end in elimination (vs. 0/20). Max speed is marginally
higher (17.33 vs. 15.53 m/s) but doesn't translate into better lap times
or reliability. Wins: 19/20 (one loss, seed 8675309 vs.
`default_student_controller`, 1/2) -- still generally competitive, just
not better than the existing best.

## What this means

This is the **sixth** consecutive reward-tuning/mechanism attempt today
(cap=12.0, cap=20.0, more training at cap=10.0 [races=80], steering
smoothness, trajectory-bonus buggy, trajectory-bonus fixed) that has
failed to beat the plain races=40 checkpoint on speed/lap-time, though
this is the first of those six that is a *design/implementation* result
rather than purely a hyperparameter one -- the underlying bug is real and
now fixed, but a single from-scratch training run with the corrected
mechanism still landed worse than the reference. Given only n=1 per
variant tried today, this specific comparison can't yet distinguish "the
trajectory-bonus idea doesn't help on top of the existing reward" from
"this particular from-scratch run had worse luck than the reference run
did" -- every experiment today has been one fresh random-init training run
per condition.

## Decision and rationale

Not adopting this checkpoint. Keeping the trajectory-bonus code and its
fix (default-off via `--trajectory-bonus`, no impact on default behavior)
since the underlying mechanism is now correctly implemented and could be
revisited later, e.g. combined with `--resume-from` (fine-tuning the
already-good races=40 policy with the bonus, rather than another
from-scratch run) or a smaller `WEIGHT_TRAJECTORY_BONUS`.

Given six straight attempts at beating races=40 via reward changes (all
trained from scratch) have not succeeded, shifting the next experiment to
a genuinely different mechanism: `--resume-from`, added to
`scripts/train_sac.py` in this same session, to continue training the
already-good races=40 policy directly instead of re-deriving a new policy
from random initialization under every reward variant.

## Next steps

1. **(next, in progress)** Test `--resume-from` on the races=40 checkpoint
   with the plain (non-trajectory) reward, continuing training rather than
   restarting -- isolates whether continued optimization from a strong
   prior beats another from-scratch run, independent of the trajectory
   question.
2. If that helps, a natural follow-up is resuming *with*
   `--trajectory-bonus` enabled once the fine-tuned baseline is
   established, to see whether the bonus helps as a refinement on top of
   an already-competent policy rather than from scratch.
3. Otherwise, treat races=40 as the practical best result for this
   project's remaining time and shift focus to other open items (broader
   seed testing, repackaging `controllers.race_faster`, the
   `controllers.minimum_viable` module gap).
