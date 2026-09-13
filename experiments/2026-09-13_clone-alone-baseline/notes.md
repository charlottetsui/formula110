# controllers.imitation run completely alone: the baseline every combined-approach variant must beat (2026-09-13)

A one-off evaluation (not a training run -- `controllers.imitation.Controller`
is already a frozen, trained artifact) establishing what "just sticking to
one approach" actually means, quantitatively, before judging whether any
combined (SAC + clone) variant improves on it. Uses the exact same protocol
(5-seed set, 2 races/seed, 120s rounds) as every residual-checkpoint
evaluation this session, run via `run_headless_head_to_head` directly
against all three available opponents (`crash_fast`,
`default_student_controller`, `leaderboard_expert` as a live opponent).

## Results

| opponent | eliminated | wins | best lap | laps |
| --- | --- | --- | --- | --- |
| crash_fast | 0/10 | 10/10 | 8.89s | 14.30 |
| default_student_controller | **2/10** | 9/10 | 8.77s | 11.30 |
| leaderboard_expert (live) | 0/10 | **6/10** | 8.93s | 14.60 |

## Read

The raw clone, with no SAC involved at all, is already an excellent
controller -- fast (best lap consistently ~8.8-8.9s, far faster than any
combined variant tried this session) and safe against two of the three
opponents, including winning a majority of races against
`leaderboard_expert` itself (the controller it was trained to imitate).
Its one clear, real weakness is a 2/10 elimination rate against
`default_student_controller` specifically. This sets a high, honest bar:
every subsequent combined-approach attempt
(`experiments/2026-09-13_residual-clone-base-seed8000/` and its
scale/hazard-gating follow-ups) is compared against these numbers, not
just against each other.

## Decision and rationale

n/a -- an evidence-gathering run, not a design decision.

## Next steps

Referenced from every combined-approach clone-base entry in
`docs/lab_notebook.md`'s 2026-09-13 sessions as the baseline to beat.
