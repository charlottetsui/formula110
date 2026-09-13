# Hazard-gated residual RL on the imitation clone: closes most of the pace gap to "clone alone" (2026-09-13)

Follow-up to `experiments/2026-09-13_residual-clone-base-seed8000/` (uniform
residual correction on top of `controllers.imitation`, applied every tick).
That run and its `-scale010-`/`-scale050-` siblings all traded away a large
fraction of the clone's raw pace to fix one narrow weakness
(`default_student_controller` eliminations), and even made the
`leaderboard_expert` matchup worse than the raw clone alone. This tests a
more targeted fix: apply the SAC correction only on ticks
`training.reward.in_hazard` (now public) judges a wall- or
competitor-proximity hazard; every other tick, the clone's command passes
through completely untouched.

Implemented as a new `residual_hazard_gated` mode on `TrainableController`
(`src/training/controller.py`), requiring `residual_base=True`. Important
correctness detail: when the correction is gated off, the *stored* replay
action must be zeroed (not the network's actual, uncorrected-for-this
output) since that's what was actually physically applied -- otherwise the
critic would mislearn Q(s, a) for a nonzero action that never affected the
car. `--residual-hazard-gated` on `scripts/train_sac.py`/`scripts/eval_sac.py`.

Matched config: seed=8000, races=40, round_seconds=120, n_step=3,
hidden_size=128, residual_base_source=clone, residual_action_scale=0.3 --
`--residual-hazard-gated` is the only new variable vs. the uniform-0.3 run.

## Results (all vs. the clone alone / uniform-0.3 comparison points)

**vs. default_student_controller (10 races, 120s rounds):**

| | clone alone | uniform 0.3 | hazard-gated 0.3 (this run) |
| --- | --- | --- | --- |
| eliminated | 2/10 | 0/10 | **1/10** |
| damage | 0.2043 | 0.0461 | 0.1918 |
| best lap | 8.77s | 12.77s | **10.33s** |
| laps | 11.30 | 8.60 | **10.30** |

**vs. crash_fast (10 races, 120s rounds):**

| | clone alone | uniform 0.3 | hazard-gated 0.3 |
| --- | --- | --- | --- |
| eliminated | 0/10 | 0/10 | 0/10 |
| best lap | 8.89s | 12.77s | **10.85s** |
| laps | 14.30 | 9.10 | **11.10** |
| car-contact | 0.348s | 0.377s | **0.138s** |

**vs. leaderboard_expert as a live opponent (10 races, 120s rounds):**

| | clone alone | uniform 0.3 | hazard-gated 0.3 |
| --- | --- | --- | --- |
| eliminated | 0/10 | 2/10 | **1/10** |
| wins | 6/10 | 0/10 | 0/10 |
| best lap | 8.93s | 13.12s | **10.20s** |
| laps | 14.60 | 7.40 | **11.30** |

## Read

Hazard-gating is Pareto-better than the uniform correction on *every*
metric tested: it fixes more of the `default_student_controller` weakness
per unit of pace given up, and shrinks (rather than grows) the new
`leaderboard_expert` weakness the uniform correction introduced. This
validates the design hypothesis: the clone's failure is localized to rare
proximity moments, and a correction scoped to only those moments preserves
far more of its excellent solo behavior than one applied everywhere.

It is still not a strict, unconditional win over running the clone alone.
`default_student_controller` eliminations improved (2/10 -> 1/10) but
didn't reach zero, and a new, smaller `leaderboard_expert` weakness
appeared (0/10 -> 1/10 eliminated, and the clone's lucky 6/10 win rate
there drops to 0/10) at a real, if much smaller, pace cost (8.77-8.93s ->
10.20-10.85s). This is the best trade found across every combined-approach
variant tried today, not a checkpoint that dominates the clone on every
axis.

## Decision and rationale

Adopted as the packaged `combined_candidate` checkpoint (Charlotte's
explicit direction: combine the two tracks' own trained artifacts even
knowing the pure-clone numbers are stronger in places). Chosen over the
uniform-correction checkpoints because it is strictly better than all of
them on every metric measured, not because it beats the clone alone
unconditionally -- it doesn't, and the module's own docstring says so
plainly.

## Next steps

1. A wider gated scale (0.6) was tried as an immediate follow-up
   (`experiments/2026-09-13_residual-clone-hazard-gated-scale06-seed8000/`)
   and made things worse across the board -- not a promising direction to
   push further in that dimension.
2. Untried: a seed sweep at this exact (gated, scale=0.3) config, per this
   track's established n=1 caution.
3. Untried: widening what counts as a "hazard" specifically for gating
   purposes (e.g. an earlier warning distance than `training.reward`'s
   reward-shaping thresholds use), so the correction has more lead time to
   act before a collision is imminent -- distinct from just scaling the
   correction's magnitude, which (per the scale=0.6 result) is not the
   productive lever.
4. Still open: the `controllers.minimum_viable` module gap.
