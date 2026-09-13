# Uniform residual scale sweep, narrow end (scale=0.1): worse than both neighbors (2026-09-13)

Part of a residual-scale sweep on the clone-base checkpoint
(`experiments/2026-09-13_residual-clone-base-seed8000/`, scale=0.3), run
alongside `experiments/2026-09-13_residual-clone-base-scale050-seed8000/`
(scale=0.5) to see whether a narrower or wider uniform correction trades
pace against `default_student_controller` eliminations better than 0.3
does. Matched config identical to the scale=0.3 run except
`--residual-action-scale 0.1`.

## Results (120s-round protocol)

| | clone alone | scale=0.1 (this run) | scale=0.3 | scale=0.5 |
| --- | --- | --- | --- | --- |
| default_student_controller eliminated | 2/10 | **2/10 (no fix)** | 0/10 | 0/10 |
| crash_fast eliminated | 0/10 | **2/10 (worse, new)** | 0/10 | 0/10 |
| leaderboard_expert eliminated | 0/10 | 1/10 | 2/10 | 1/10 |
| default_student_controller best lap | 8.77s | 11.53s | 12.77s | 13.71s |

## Read

Strictly worse than both clone-alone and scale=0.3: it doesn't fix the
`default_student_controller` weakness at all (still 2/10 eliminated) while
*introducing* a new elimination mode against `crash_fast` (2/10) that
neither the clone alone nor scale=0.3 ever had, and it's still
meaningfully slower than the clone alone. This isn't read as a clean
"scale needs to be even smaller" signal -- more likely a single-seed,
too-weak-to-stabilize draw (a correction too small to reliably counteract
the specific failure mode, but still large enough to perturb the clone's
otherwise-good behavior into new mistakes). Superseded by the hazard-gated
design (`experiments/2026-09-13_residual-clone-hazard-gated-seed8000/`),
which fixed the actual problem (restricting *when* the correction fires,
not just its magnitude).

## Decision and rationale

Not adopted; ruled out immediately by the results above.

## Next steps

None specific to this run -- superseded by the hazard-gated line of
experiments.
