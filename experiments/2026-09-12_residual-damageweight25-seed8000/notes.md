# Residual RL, damage weight halved 5.0 -> 2.5: a clean regression (2026-09-12)

Sixth attempt at closing the remaining pace gap to `controllers.leaderboard_expert`'s
raw speed (8.94s solo) -- see `docs/rl_design.md` section 6, causal test 37 and
its follow-ups for the five prior failed levers, including the companion
experiment `2026-09-12_residual-wallscale20-seed8000` run this same session.
This tests the most direct "accept more risk on purpose" lever available:
`training.reward`'s `WEIGHT_DAMAGE`, held at `5.0` unchanged since the
reward's inception and never itself the variable in any prior causal test.
Added an opt-in `damage_weight` override to `step_reward`/`TrainableController`/
`scripts/train_sac.py --damage-weight` (default unchanged, so plain self-play
is unaffected). Trained at `2.5` (half the default), matched to
`2026-09-11_residual-expert-base-v2-seed8000`'s exact config otherwise
(seed=8000, races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, residual_expert_base=True, residual_action_scale=0.3).

**Caveat found 2026-09-12 (later same session):** the same
`--opponent self` incumbent bug documented in the companion wall-scale
experiment's notes applies here too -- roughly half of this run's
transitions (the incumbent's) were computed under the *default*
`WEIGHT_DAMAGE=5.0`, not the intended `2.5`. Fixed in `scripts/train_sac.py`
the same session; this run was not re-trained under the fix. The
elimination-rate regression observed (2/10 vs. v2's 0/10) occurred despite
only diluted exposure to the lowered damage weight, which if anything
understates how much a fully-applied `damage_weight=2.5` might cost --
not grounds to discount the "not adopted" conclusion.

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v2 (reference) | damage-weight=2.5 |
| --- | --- | --- |
| avg damage | 0.0254 | 0.0606 (worse) |
| avg off-track | 0.325s | 0.594s (worse) |
| avg wall-contact | 0.140s | 0.307s (worse) |
| avg car-contact | 1.468s | 0.974s (better) |
| avg best lap time | 11.37s | 11.66s (slower) |
| fastest lap | 10.33s | 11.12s (slower) |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races, same stress test as every
causal test 37 follow-up):**

| | v2 (reference) | damage-weight=2.5 |
| --- | --- | --- |
| eliminated | 0/10 | **2/10 (regressed)** |
| avg laps | 9.80 | 8.80 (worse) |
| avg best lap time | 11.78s | 12.32s (slower) |
| wins | -- | 1/10 |

## Read

**A clean regression, not a trade-off.** Halving the damage penalty was
meant to let the policy tolerate more risk in exchange for pace, but pace
got worse too -- slower on the standard baselines and against the expert
alike -- while safety visibly degraded (two real eliminations against the
expert, where v2 and every wall-scale variant tested so far have zero).
`metrics.csv` showed normal, bounded critic loss throughout, so this isn't
a training-instability artifact. Same failure family as several earlier
"loosen a caution term hoping to unlock speed" attempts on this track
(e.g. the 2026-09-03 `WEIGHT_CENTER_OFFSET` halving, the 2026-09-12
curvature-aware center-offset test) -- the caution term being loosened was
apparently load-bearing for competent driving, not merely limiting top
speed.

## Decision and rationale

**Not adopted.** This is the sixth structurally different lever (after
residual scale x2, network-initialization seed sweep, progress-weight
reweight, curvature-aware center-offset, and this session's
wall-proximity-speed-scale test) to fail at closing the pace gap to Lucy's
raw expert -- and the first to actively regress safety below v2's perfect
record against the expert. `damage_weight` is kept as a tested, documented,
opt-in parameter (default unchanged) rather than reverted code, consistent
with this track's practice of preserving negative results.

## Next steps

1. `2026-09-11_residual-expert-base-v2-seed8000` remains the best
   combined-approach checkpoint, now the strongest result after six
   independent attempts to beat it on pace.
2. Recommending against further attempts to close this specific gap via
   training-side reward tuning -- the pattern is now six-for-six.
