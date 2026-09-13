# Residual RL on top of the imitation-learning clone instead of the expert: a negative result (2026-09-13)

Tests whether `TrainableController`'s `residual_base` mode should use
`controllers.imitation.Controller` (Lucy's behavioral clone, trained via
behavior cloning on `leaderboard_expert`'s trajectories) instead of
`controllers.leaderboard_expert.Controller` (the hand-written expert
itself) as the base action every tick -- the combination
`combined_candidate.py` and every causal-test-37 run has used so far.
Added a `residual_base_source` option (`"expert"` default / `"clone"`) to
`training/controller.py` to make this a single-flag change
(`--residual-base-source clone` on `scripts/train_sac.py`/
`scripts/eval_sac.py`).

Matched the original reference config exactly (seed=8000, races=40,
round_seconds=120, n_step=3, hidden_size=128, buffer_capacity=800000,
matching `2026-09-08_seed-sweep-v2-8000` / `2026-09-11_residual-expert-
base-seed8000`) -- `--residual-base-source clone` is the only new
variable.

## Results

**vs. crash_fast + default_student_controller (20 races, 120s rounds):**

| | expert-base residual RL (seed8000) | clone-base residual RL (seed8000) |
| --- | --- | --- |
| avg damage | 0.0662 | 0.0231 |
| avg off-track | 0.537s | 0.380s |
| avg wall-contact | 0.256s | 0.210s |
| avg car-contact | 1.669s | 0.661s |
| avg laps | 10.10 | 8.85 |
| avg best lap time | 11.25s | 12.77s |
| avg max speed | 37.33 m/s | 35.28 m/s |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races, 120s rounds -- the
decisive stress test, same protocol as causal test 37):**

| | expert-base residual RL | clone-base residual RL |
| --- | --- | --- |
| eliminated | 1/10 | **2/10** |
| avg damage | 0.2278 | 0.2194 |
| avg car-contact | 6.147s | 4.605s |
| avg off-track | 1.320s | 1.453s |
| avg wall-contact | 0.595s | 0.843s |
| avg laps | 9.50 | **7.40** |
| avg best lap time | 12.08s | **13.12s** |
| wins | 0/10 | 0/10 |

Raw data: `vs_standard_120s/eval_results.json`,
`vs_leaderboard_expert_120s/eval_results.json` (the comparable, correct-
round-length runs). `eval_results.json` and `vs_leaderboard_expert/` at
this directory's top level are from an initial pass that mistakenly used
the eval script's 20s default round length instead of the required 120s
match -- kept for transparency, not comparable to the table above.

## Read

Safer than the expert-base variant against the two easy standard
baselines (both already sat at 0/20 eliminated, so not very
differentiating), but worse against `leaderboard_expert` itself on every
metric not confounded by falling behind: elimination rate doubled,
laps completed dropped ~22%, best lap time got slower, and off-track/
wall-contact time both rose. The lower car-contact figure against the
expert is most likely explained by the car falling further behind and
spending less time near the opponent at all, not by being genuinely
safer.

Plausible mechanism: the clone is a lossy approximation of the expert
(behavior cloning typically regresses toward more conservative/averaged
behavior relative to its teacher), so residual RL on top of it starts
from a slower, less-refined base than the exact expert provides, and the
learned correction does not fully close that gap -- it does close some of
it against slow/simple baselines, but not against the expert itself,
where pace and precision matter most.

## Decision and rationale

Not adopted. `combined_candidate.py` stays on `leaderboard_expert` as its
residual base. Single run/seed, per this track's established n=1 caution,
but the result is directionally consistent across every non-confounded
metric on the specific test built to be decisive for this kind of change,
so this isn't read as a close call a seed sweep would likely flip. The
`residual_base_source` option itself is kept as tested, backward-
compatible infrastructure (default unchanged), since it turned an
otherwise-unverifiable design question into a real answer.

## Next steps

1. `src/training/imitation_handoff.py` already has unfinished
   infrastructure for the more standard way to combine BC and RL as a
   single artifact -- warm-starting the SAC actor's own weights from the
   clone, then continuing normal training -- rather than a frozen-clone-
   plus-residual composition. Worth pursuing if "combine two learned
   models" specifically (rather than "expert + SAC") remains a goal.
2. A seed sweep would firm up robustness but isn't prioritized given the
   directional consistency of this single run's live-expert-matchup
   result.
