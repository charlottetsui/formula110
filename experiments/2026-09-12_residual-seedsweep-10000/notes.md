# Residual RL seed sweep, seed=10000: mixed, still not better than v2 (2026-09-12)

Second draw in the seed sweep at the locked-in v2 config
(residual_base=True, residual_action_scale=0.3, recovery-passthrough
fix), fresh seed=10000.

## Result

Standard baselines: 0/20 eliminated (matches v2), avg car-contact
actually improved (0.77s vs. v2's 1.468s), but avg off-track (0.935s vs.
0.325s) and avg wall-contact (0.567s vs. 0.140s) both got meaningfully
worse, avg laps dropped (9.25 vs. 10.15), and avg best lap time got
slower (12.31s vs. 11.37s).

vs. leaderboard_expert: 0/10 eliminated (matches v2), lower avg damage
(0.1307 vs. 0.2204), but fewer laps (9.3 vs. 9.8) and slower lap time
(12.55s vs. 11.78s).

`metrics.csv` showed stable training throughout.

## Decision and rationale

Not adopted -- a mixed result (better on some safety metrics, worse on
pace and other safety metrics) rather than a clear win. Combined with
seed=9000 (clearly worse), two of two sampled seeds fail to beat v2
overall. `2026-09-11_residual-expert-base-v2-seed8000` remains the best
checkpoint.

## Next steps

Two seeds sampled, zero improvements found -- consistent with this
project's own observed per-seed hit rate for finding a strict
improvement (roughly 10-20% on the plain-SAC side, meaning several more
draws could plausibly be needed). Awaiting direction on whether to
continue sampling or conclude the seed sweep here.
