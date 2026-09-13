# Uniform residual scale sweep, wide end (scale=0.5): fixes the target elimination, costs even more pace (2026-09-13)

Part of the same residual-scale sweep as
`experiments/2026-09-13_residual-clone-base-scale010-seed8000/`
(scale=0.1) and `experiments/2026-09-13_residual-clone-base-seed8000/`
(scale=0.3). Matched config identical to the scale=0.3 run except
`--residual-action-scale 0.5`.

## Results (120s-round protocol)

| | clone alone | scale=0.3 | scale=0.5 (this run) |
| --- | --- | --- | --- |
| default_student_controller eliminated | 2/10 | 0/10 | 0/10 |
| crash_fast eliminated | 0/10 | 0/10 | 0/10 |
| leaderboard_expert eliminated | 0/10 | 2/10 | 1/10 |
| default_student_controller best lap | 8.77s | 12.77s | **13.71s (worse)** |
| leaderboard_expert best lap | 8.93s | 13.12s | 14.06s (worse) |

## Read

Also fixes the `default_student_controller` elimination (0/10, matching
scale=0.3) and is marginally safer against `leaderboard_expert` (1/10 vs.
scale=0.3's 2/10 eliminated), but at a further pace cost on top of an
already-large one (worse best lap than scale=0.3 in every matchup tested).
Widening the uniform correction doesn't reveal a better trade-off point
than scale=0.3 -- it's a strictly slower variant with a similar or
marginally better safety profile, not a clear improvement. Superseded by
the hazard-gated design
(`experiments/2026-09-13_residual-clone-hazard-gated-seed8000/`), which
gets most of this run's safety benefit at roughly scale=0.3's pace cost by
restricting *when* the correction fires instead of only its magnitude.

## Decision and rationale

Not adopted. Confirmed the uniform-correction approach (regardless of
scale) trades away too much of the clone's pace to be worth pursuing
further as a family; motivated moving to hazard-gating instead.

## Next steps

None specific to this run -- superseded by the hazard-gated line of
experiments.
