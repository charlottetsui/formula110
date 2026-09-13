# Widening the residual scale to 0.45: a clean regression, not a speed gain (2026-09-11)

Direct follow-up to v2 (`experiments/2026-09-11_residual-expert-base-v2-
seed8000/`), which closed the recovery-passthrough bug but still laps
noticeably slower (11.37-11.78s) than Lucy's raw expert alone (8.94s,
same protocol). Tested whether giving the policy more room to deviate
from the expert's baseline action would close that gap: made
`RESIDUAL_ACTION_SCALE` configurable (`--residual-action-scale` on
`scripts/train_sac.py`, threaded through `TrainableController`,
`evaluate_against_baselines`, and `scripts/eval_sac.py`) and trained
fresh at `0.45` (up from `0.3`), otherwise identical config (seed=8000,
races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, recovery-passthrough fix already in code).

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v2 (scale=0.3) | scale=0.45 |
| --- | --- | --- |
| avg damage | 0.0254 | 0.1253 (5x worse) |
| avg off-track | 0.325s | **5.737s (17.6x worse)** |
| avg wall-contact | 0.140s | **3.305s (23.6x worse)** |
| avg car-contact | 1.468s | 1.867s (worse) |
| avg laps | 10.15 | 8.70 (worse) |
| avg best lap time | 11.37s | 11.83s (**slower, not faster**) |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 19/20 (lost one race) |

**vs. leaderboard_expert as a live opponent (10 races):**

| | v2 (scale=0.3) | scale=0.45 |
| --- | --- | --- |
| avg damage | 0.2204 | 0.1672 |
| avg off-track | 1.320s | **9.92s (7.5x worse)** |
| avg wall-contact | 0.595s | **5.24s (8.8x worse)** |
| avg laps | 9.80 | 8.80 |
| avg best lap time | 11.78s | 12.47s (**slower, not faster**) |
| eliminated | 0/10 | 0/10 |

`metrics.csv` shows no training divergence (critic loss bounded, alpha
settling smoothly to ~0.024-0.026) -- this isn't an unstable-training
artifact, the policy converged cleanly to a genuinely worse driving
style.

## Read

This is a clean, consistent regression across both evaluation protocols,
not noise or an unstable run. More freedom to deviate from the expert's
line did not translate into a faster racing line -- if anything, lap time
got marginally *slower* in both matchups -- while off-track and
wall-contact time exploded by 7-24x. The extra correction budget appears
to be spent on erratic/oscillating deviations rather than a genuinely
better line, the same failure family as other "loosen a constraint and
hope for a faster line" attempts elsewhere on this track (e.g. the
`WEIGHT_CENTER_OFFSET` halving experiment) -- widening how far the
learned part can push doesn't automatically translate that freedom into
useful pace.

## Decision and rationale

**Not adopted; reverted.** `2026-09-11_residual-expert-base-v2-seed8000`
(`residual_scale=0.3`, the default) remains the best combined-approach
checkpoint. `--residual-action-scale` is kept as a configurable, tested
parameter (default unchanged at `RESIDUAL_ACTION_SCALE = 0.3`) per this
track's convention of preserving tested-but-rejected mechanisms rather
than deleting the code that produced them.

## Next steps

1. This specific lever (scale) is treated as exhausted after one clear
   negative result in both directions tested (0.3 baseline, 0.45 failed) --
   not continuing to search this axis without a different underlying idea,
   consistent with this project's established practice.
2. The remaining gap to Lucy's raw pace (8.94s) may partly be structural:
   her expert explicitly accepts more risk to maximize speed, while the
   RL reward function balances speed against safety/damage/off-track
   penalties by design -- a controller optimizing for both may have a
   genuine ceiling below a controller optimizing for speed alone.
3. If pursued further, a different lever (more training at the existing
   scale, or reward-weight adjustments specific to residual-mode
   training) would need to be tried, not more of this same axis.
