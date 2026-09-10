# Increased self-play traffic as a controlled fine-tune -- zero effect on the target metric, still regresses pace (2026-09-08)

Follow-up to `2026-09-08_copies2-seed8000/notes.md` (from-scratch
copies_per_side=2, confounded by ~2x the effective training budget).
This time, fine-tuned the *already-good* current checkpoint
(`2026-09-08_seed8000-resumed-short`) via `--resume-from` with
`--copies-per-side 2 --races 10` (a small dose, matching the successful
"+10 races" pattern from the earlier near-miss fix), rather than
retraining from scratch -- specifically to avoid conflating "denser
traffic" with "more total training."

## Result: car-contact time literally unchanged; pace still regresses

| | current best (reference) | copies=2 fine-tune (+10 races) |
| --- | --- | --- |
| avg low-progress | 3.30s | 3.27s (flat) |
| **avg car-contact** | **1.70s** | **1.72s (flat -- no effect)** |
| avg laps | 6.75 | 4.50 |
| avg best lap time | 15.54s | 24.37s |
| avg max speed | 30.3 m/s | 18.7 m/s |
| avg damage | 0.0004 | 0.0000 |
| eliminated | 0/20 | 0/20 |

Unlike the from-scratch version (which showed a modest ~19% car-contact
improvement alongside a large pace cost, confounded by extra training
budget), this controlled version shows **no car-contact improvement at
all** -- the metric this experiment was specifically testing is
completely flat -- while still costing significant pace. Safety
otherwise unaffected (still 0/20 eliminated, near-zero damage).

## Read

With the training-budget confound removed, the result is unambiguous:
10 additional races of denser self-play traffic did not teach any
measurable collision-avoidance improvement on this already-competent
checkpoint, at any dose tested. Combined with the from-scratch result,
self-play traffic density is not a productive lever for this specific
goal on this checkpoint.

## Decision and rationale

Not adopted. `2026-09-08_seed8000-resumed-short` remains the reference
checkpoint. This is the fifth consecutive hesitation-reduction attempt
this session (three reward-shaping variants, two self-play-traffic
variants) to fail -- three regressed pace for a real if partial
improvement, one regressed pace with no improvement at all, and one
(the reward-shaping angle-restricted attempt) collapsed entirely.
Recommending this thread be closed: the accumulated evidence across five
independently-designed experiments, using different mechanisms (reward
terms, training procedure), points at the current checkpoint's
hesitation level being a practical floor for this reward/observation/
architecture combination, not a gap closeable by the levers tried so
far.

## Next steps

1. If pursued further, the two remaining genuinely untried levers are:
   a closing-speed-scaled competitor-proximity reward term (may have a
   similar "maintain constant distance forever" escape hatch as the two
   failed distance-based variants, so lower confidence), or a
   controller-level deterministic safety backstop (structurally
   different -- an inference-time override, not a training-time
   incentive, so immune to the "policy learns to avoid the situation
   entirely" failure mode every reward-based attempt has hit).
2. Otherwise, accept the current checkpoint's hesitation level as final
   for this track.
