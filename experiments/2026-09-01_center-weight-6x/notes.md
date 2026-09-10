# Reward reweight: center-offset penalty 0.05 -> 0.3 (2026-09-01)

Single-variable experiment following the triage of the 2026-08-31 minimum
experiment (see chat/lab notebook): the trained controller spent 7.5-27% of
each race off-track. Hypothesis: `WEIGHT_CENTER_OFFSET` (0.05) was too
small relative to `WEIGHT_PROGRESS` (1.0) for the policy to prefer staying
centered over cutting corners at speed.

## What changed

`src/training/reward.py`: `WEIGHT_CENTER_OFFSET = 0.05` -> `0.3`. Nothing
else — same hyperparameters, same base seed (`110`), same SAC network
random seed (`SACAgent`'s default `seed=0`, unchanged by `args.seed`) as
the 2026-08-31 baseline run. Everything except the reward's center-offset
weight is identical between the two runs by construction, not just by
matching CLI flags.

## Result

Averaged over 10 evaluation races (5 seeds x 2 races) against `crash_fast`:

| | 2026-08-31 baseline (w=0.05) | 2026-09-01 (w=0.3) |
| --- | --- | --- |
| avg scored distance | 28.9 m | 24.8 m |
| avg off-track time | 3.64 s (18.2%) | 3.42 s (17.1%) |
| avg wall contact time | 3.12 s | 2.97 s |
| avg low-progress time | 4.55 s | 4.62 s |
| avg marshal count | 2.30 | 2.40 |

Same pattern against `default_student_controller` (avg distance 19.8m ->
17.7m, off-track 19.1%->19.5%, marshal 2.40->2.60). Full per-seed numbers
in `eval_results.json`.

## What we observed

No measurable improvement, and if anything a slightly worse scored
distance (within noise given only 10 races per condition). Off-track time,
wall contact, and marshal count are all statistically indistinguishable
between the two weights.

## Decision and rationale

A 6x reweight of one reward term produced no detectable behavioral change.
Given only ~2,448 gradient updates (most of the training budget spent in
warmup), the most likely explanation is that the policy hasn't had enough
updates to exploit *any* reshaped incentive yet — training budget, not
reward shape, is the current bottleneck. Not reverting the weight change
(it's still a reasonable prior and isn't measurably worse), but the next
experiment should scale up training budget as its own isolated variable
before re-testing reward shaping. See `docs/rl_design.md` section 6, item 0.
