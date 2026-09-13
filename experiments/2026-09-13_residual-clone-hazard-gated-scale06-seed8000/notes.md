# Widening the hazard-gated correction to scale=0.6: worse across the board (2026-09-13)

Immediate follow-up to `experiments/2026-09-13_residual-clone-hazard-gated-seed8000/`
(hazard-gated residual RL on the imitation clone, scale=0.3): now that the
correction only fires on hazard ticks, does a *stronger* correction during
those rarer moments more decisively fix `default_student_controller`
eliminations without the pace cost that made a stronger *uniform*
correction (scale=0.5, `experiments/2026-09-13_residual-clone-base-scale050-seed8000/`)
unattractive? Matched config identical to the scale=0.3 hazard-gated run
except `--residual-action-scale 0.6`.

## Results

| | clone alone | hazard-gated 0.3 | hazard-gated 0.6 (this run) |
| --- | --- | --- | --- |
| default_student_controller eliminated | 2/10 | 1/10 | **3/10 (worse)** |
| crash_fast eliminated | 0/10 | 0/10 | **1/10 (worse, new)** |
| leaderboard_expert eliminated | 0/10 | 1/10 | 1/10 (same) |
| default_student_controller best lap | 8.77s | 10.33s | 10.62s (about the same) |

## Read

No. Widening the correction's magnitude during hazard moments made things
worse, not better -- both `default_student_controller` eliminations
roughly tripled and a brand-new `crash_fast` elimination appeared that
neither the clone alone nor either 0.3 variant (uniform or gated) ever
had, for essentially no pace benefit over scale=0.3. A larger correction
evidently overshoots/destabilizes during exactly the close-proximity
moments where precision matters most, rather than more decisively
resolving them. Magnitude is not the productive lever once gating is
already in place; the gate itself (deciding *when* to correct) did the
real work in the scale=0.3 result.

## Decision and rationale

Not adopted. `hazard-gated 0.3` remains the best combined-approach
checkpoint found this session, and is what `combined_candidate.py` uses.

## Next steps

Same as `2026-09-13_residual-clone-hazard-gated-seed8000/notes.md`'s: a
seed sweep at scale=0.3, or widening the hazard *detection* window rather
than the correction's magnitude, are the more promising untried levers.
