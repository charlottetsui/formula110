# Residual RL seed sweep, seed=9000: worse than v2, not adopted (2026-09-12)

First draw in a seed sweep at the locked-in v2 config (residual_base=True,
residual_action_scale=0.3, recovery-passthrough fix), following the
residual-scale axis being closed out as exhausted. Fresh network
initialization (seed=9000) instead of the seed=8000 used for every
residual experiment so far, otherwise identical config.

## Result

vs. v2 (seed=8000) on the standard baselines: avg damage worse (0.0682
vs. 0.0254), avg laps worse (9.45 vs. 10.15), avg best lap time worse
(11.64s vs. 11.37s), and **one elimination reappeared (1/20 vs. v2's
0/20)** -- off-track and wall-contact were marginally better, but the
elimination and damage/laps/pace regressions dominate. `metrics.csv`
showed stable training (no divergence) -- this is a genuinely worse
network initialization for this task, not a broken run.

## Decision and rationale

Not adopted. `2026-09-11_residual-expert-base-v2-seed8000` remains the
best checkpoint. One failed seed is weak evidence on its own -- continuing
the sweep with additional fresh seeds before drawing a conclusion about
whether this lever can beat v2.
