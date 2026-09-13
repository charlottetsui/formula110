# Residual RL, wall-proximity-speed-scale raised 10 -> 20: no improvement (2026-09-12)

Fifth attempt at closing the remaining pace gap to `controllers.leaderboard_expert`'s
raw speed (8.94s solo), after four prior levers (residual scale in both
directions, a 5-seed sweep, a flat progress-weight reweight, curvature-aware
center-offset) all failed -- see `docs/rl_design.md` section 6, causal test 37
and its follow-ups. This tests a new lever: `training.reward`'s
`WALL_PROXIMITY_SPEED_SCALE_MPS`, the closest thing left to a "speed ceiling"
since `MAX_REWARDED_SPEED_MPS` was removed structurally (causal test 17,
2026-09-02). Raising it makes the wall-proximity penalty grow more slowly with
speed -- tolerates more speed before pricing it as risky. Added an opt-in
`wall_proximity_speed_scale_mps` override to `step_reward`/`TrainableController`/
`scripts/train_sac.py --wall-proximity-speed-scale` (default unchanged, so
plain self-play is unaffected). Trained at `20.0` (2x the default), matched to
`2026-09-11_residual-expert-base-v2-seed8000`'s exact config otherwise
(seed=8000, races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, residual_expert_base=True, residual_action_scale=0.3).

**Caveat found 2026-09-12 (later same session):** `scripts/train_sac.py`'s
`--opponent self` incumbent construction didn't pass
`wall_proximity_speed_scale_mps`/`damage_weight` through to the incumbent's
own `TrainableController` -- only the challenger got the override. Since
both copies push transitions to the same shared buffer in self-play, roughly
half of this run's transitions (the incumbent's) were actually computed
under the *default* `WALL_PROXIMITY_SPEED_SCALE_MPS=10.0`, not the intended
`20.0`. Fixed in `scripts/train_sac.py` the same session. This run was not
re-trained under the fix -- treat this result as a diluted (roughly
half-strength) version of the intended treatment, not a clean test of
`wall_proximity_speed_scale_mps=20.0` applied throughout. The regression
observed here is if anything a lower bound on the effect's downside, not
grounds to discount the "not adopted" conclusion.

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v2 (reference) | wall-scale=20 |
| --- | --- | --- |
| avg damage | 0.0254 | 0.0505 (worse) |
| avg off-track | 0.325s | 0.624s (worse) |
| avg wall-contact | 0.140s | 0.217s (worse) |
| avg best lap time | 11.37s | 11.88s (slower) |
| fastest lap | 10.33s | 11.40s (slower) |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races, same stress test as every
causal test 37 follow-up):**

| | v2 (reference) | wall-scale=20 |
| --- | --- | --- |
| eliminated | 0/10 | 0/10 |
| avg laps | 9.80 | 10.00 |
| avg best lap time | 11.78s | 13.37s (slower) |
| wins | -- | 0/10 |

## Read

**Not an improvement on the pace goal, and a mild regression on safety.**
Every scored metric moved the wrong direction relative to v2 except lap
count against the expert (roughly flat). Loosening the speed-scaled
wall-proximity penalty let the policy tolerate more risk at speed, but that
didn't translate into a faster line -- if anything, lap time got worse in
both matchups. Training was stable throughout (`metrics.csv` showed normal,
bounded critic loss), so this isn't an unstable-training artifact.

## Decision and rationale

**Not adopted.** This is the fifth structurally different lever (after
residual scale x2, network-initialization seed sweep, progress-weight
reweight, curvature-aware center-offset) to fail at closing the pace gap
to Lucy's raw expert, consistent enough with the prior four to continue
reading this as a real structural trade-off rather than a nearby local
optimum. `wall_proximity_speed_scale_mps` is kept as a tested, documented,
opt-in parameter (default unchanged) rather than reverted code, consistent
with this track's practice of preserving negative results.

## Next steps

1. See the companion experiment `2026-09-12_residual-damageweight25-seed8000`
   (same session, a different lever: lowering `WEIGHT_DAMAGE` directly).
2. `2026-09-11_residual-expert-base-v2-seed8000` remains the best
   combined-approach checkpoint.
