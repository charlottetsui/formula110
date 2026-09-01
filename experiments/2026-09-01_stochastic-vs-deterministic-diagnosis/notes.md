# Stochastic vs. deterministic evaluation diagnosis (2026-09-01)

Direct test of the train/eval mismatch hypothesis from
`2026-09-01_longer-training-round-seed909/notes.md`: does the checkpoint's
dangerous *deterministic* (mean-action) behavior differ from its
*stochastic* (sampled, like training) behavior at the same weights?

No retraining -- loaded
`experiments/2026-09-01_longer-training-round-seed909/checkpoints/policy_final.pt`
(the 40.5 m/s / 100%-elimination checkpoint) and ran
`scripts/compare_stochastic_eval.py`, which evaluates the same checkpoint
twice via `training.evaluation.evaluate_against_baselines`: once with
`deterministic=True` (mean action, no noise -- what a packaged controller
would run), once with `deterministic=False` (sampled action, matching how
actions were chosen during training), across the same 5 fixed seeds vs.
`crash_fast`. Required adding a `deterministic` parameter to
`TrainableController`/`evaluate_against_baselines` to decouple action
selection from the `training` flag (previously coupled:
`deterministic = not training`).

## Result: hypothesis rejected

| metric | deterministic | stochastic |
| --- | --- | --- |
| avg damage | 1.000 | 1.000 |
| avg max speed | 40.457 m/s | 40.019 m/s |
| avg scored distance | 91.3 m | 94.5 m |
| elimination rate | 100% | 100% |

Essentially identical. Sampling actions with exploration noise (the same
way training chooses them) produces the same extreme speed and the same
100% elimination rate as the deterministic mean action. There is no
meaningful train/eval behavior gap here.

## Correction to the prior entry's reasoning

`2026-09-01_longer-training-round-seed909/notes.md` treated self-play's
own printed training total (himself "self-play scored distance a=913.4m,
b=852.9m over 10 races") as evidence the training process itself looked
"reasonable," in apparent contrast to eval's extreme behavior. That
reading doesn't hold up: 913.4m / 10 races = 91.3m/race, which matches
almost exactly this run's deterministic eval average (91.3m) and
stochastic eval average (94.5m). The self-play total was never evidence of
safe driving -- it was consistent with the same crash-after-a-fast-burst
pattern all along; it just wasn't divided by race count or checked against
elimination at the time.

## What this means

The policy did not learn a "safe" driving strategy that only looks
dangerous when the mean action is taken without noise -- it learned to
drive very fast and crash, as its actual, substantively trained behavior,
under both action-selection modes. The eval-mode artifact hypothesis
(option 1 from the prior turn's proposed next steps) is closed.

## Decision and rationale

Do not pursue further eval-mode-specific fixes (e.g. tuning inference
noise/temperature) -- the problem is upstream, in what the policy actually
learned to value. Leading remaining candidate not yet tested: the reward's
`forward_progress_m` term (`speed_mps * cos(heading_error) * dt_s`) has no
upper bound -- nothing in `src/training/reward.py` caps the benefit of
going faster, so a policy that discovers "more speed = more reward,
monotonically" has no structural reason to stop pushing speed higher, and
the one-time `WEIGHT_TERMINAL_PENALTY = 10.0` may simply be smaller than
the cumulative reward from a sustained high-speed burst before crashing.
Not yet tested -- would need its own causal test (e.g. capping/saturating
the progress reward's speed term, or reweighting `WEIGHT_TERMINAL_PENALTY`
much higher) before concluding this is the cause.

## Next steps (options, not yet decided)

1. **(new)** Test whether capping/saturating the speed component of
   `forward_progress_m` (e.g. reward plateaus above some speed rather than
   scaling linearly forever) removes the incentive to push speed
   indefinitely -- directly targets the mechanism above.
2. Try the current reward (idle + terminal penalties) on seed 110 (the
   "good" seed) to see whether this failure is seed-909-specific or
   general -- still open from the prior entry.
3. Run a small seed sweep (3-5 seeds) at the current reward to see the
   real outcome distribution.
4. Add a hard action/speed cap at the controller level as a safety
   backstop independent of reward shaping.
