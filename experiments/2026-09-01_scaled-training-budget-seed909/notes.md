# Repeated-seed check on the training-budget experiment (2026-09-01)

Identical configuration to `experiments/2026-09-01_scaled-training-budget`
(races=10, round_seconds=60, buffer_capacity=150000, eval_round_seconds=120,
same reward weights, same network size) with only `--seed` changed
(110 -> 909). Goal: find out whether the seed=110 run's pace regression
(1.84 -> 1.31 m/s vs. `crash_fast`) was signal or noise.

## Result: not noise around a similar mean -- a completely different, worse failure mode

| | seed=110 (prior run) | seed=909 (this run) |
| --- | --- | --- |
| avg scored distance vs crash_fast | 100.7 m | 2.2 m |
| avg damage | 0.118 | **0.000 (every race)** |
| avg off-track time | 14.1 s (11.8%) | **0.0 s (every race)** |
| avg wall contact time | 12.9 s | **0.0 s (every race)** |
| avg low-progress time | 20.9 s (17.4%) | **32.2 s (26.8%)** |
| race wins vs crash_fast | 10/10 | 5/10 (lost seed 7 outright, 0/2) |
| vs default_student_controller | 92.4-221.5 m | 0.0-69.5 m |

The seed=110 policy learned to drive fast and take some risk (damage,
off-track, wall contact all nonzero, but making real progress). The
seed=909 policy learned the opposite: **never take any risk** -- zero
damage, zero off-track time, zero wall contact in every single evaluation
race -- at the cost of spending 12-54% of each race essentially stationary
or crawling. These are two qualitatively different behaviors, not two
samples from the same underlying competence level with some variance.

## What this means

The reward function has a plausible "do nothing" trap: `WEIGHT_DAMAGE =
5.0` is large relative to `WEIGHT_PROGRESS = 1.0`, and there is no penalty
for near-zero speed that isn't reverse (`WEIGHT_REVERSE` only fires for
*negative* speed). Standing still or crawling never touches a wall, never
goes off-track, and never loses reward to `WEIGHT_CENTER_OFFSET` if it
stays near the spawn point -- it just forfeits the (comparatively small,
inconsistent) progress reward. Depending on what a given training run's
early self-play trajectory looks like, the policy can converge toward
either extreme. This looks like a genuine reward risk-asymmetry problem,
not primarily a training-budget problem -- the earlier "training budget is
the bottleneck" reading (`docs/rl_design.md` section 6, item 1) needs
revisiting.

## Decision and rationale

Do not adopt this checkpoint (it's strictly worse than seed=110's by every
metric that matters for racing). More importantly: do not treat the
seed=110 run's numbers as representative of "the" scaled-training-budget
result -- with n=2 seeds, we've now seen both an aggressive-but-crashy
policy and a frozen-but-safe policy, which is itself the finding. Do not
change reward weights yet based on n=2 -- see next steps for options.

## Next steps (proposed, not yet run)

1. Test the specific hypothesis directly: re-run with **seed=909 held
   fixed** and a small reward change targeting the "never move" trap (e.g.
   a mild idle/near-zero-speed penalty, or reducing `WEIGHT_DAMAGE`
   relative to `WEIGHT_PROGRESS`) -- a clean causal test, since the seed
   that produced the pathological freeze is held constant.
2. Alternatively, run 2-3 more seeds at the current config (cheap, ~1-2
   min each) to characterize how often each regime occurs before changing
   anything -- more data before more variables.
3. Either way, `docs/rl_design.md` section 6 needs re-prioritizing: reward
   risk-asymmetry is now competing with training budget as the top lever,
   not clearly subordinate to it.
