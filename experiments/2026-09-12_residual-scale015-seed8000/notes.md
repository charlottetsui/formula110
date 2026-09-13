# Narrowing the residual scale to 0.15: also worse, closes out the axis (2026-09-12)

Tested the opposite direction from the failed `scale=0.45` experiment: on
the hypothesis that a wider correction produced erratic, wall-hugging
driving (17-24x worse off-track/wall-contact, slightly slower lap time),
would a *narrower* correction (0.15, down from v2's 0.3) reduce erratic
driving further and possibly cost little speed, since the base expert's
line is already fast? Same matched config as v2 otherwise (seed=8000,
races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, recovery-passthrough fix already in code).

## Results

**vs. crash_fast + default_student_controller (20 races):**

| | v2 (scale=0.3) | scale=0.15 (new) | scale=0.45 (prior failure) |
| --- | --- | --- | --- |
| avg damage | 0.0254 | **0.0100 (better)** | 0.1253 |
| avg off-track | 0.325s | 0.593s (worse) | 5.737s |
| avg wall-contact | 0.140s | 0.280s (worse) | 3.305s |
| avg car-contact | 1.468s | 1.582s (worse) | 1.867s |
| avg laps | 10.15 | 9.45 (worse) | 8.70 |
| avg best lap time | 11.37s | 12.08s (worse) | 11.83s |
| eliminated | 0/20 | 0/20 | 0/20 |
| wins | 20/20 | 20/20 | 19/20 |

**vs. leaderboard_expert as a live opponent (10 races):**

| | v2 (scale=0.3) | scale=0.15 (new) |
| --- | --- | --- |
| eliminated | **0/10** | **1/10 (regressed)** |
| avg damage | 0.2204 | 0.1060 |
| avg car-contact | 4.623s | 5.698s (worse) |
| avg laps | 9.80 | 9.10 (worse) |
| avg best lap time | 11.78s | 12.77s (worse) |

`metrics.csv` showed normal, stable training (no divergence) -- this is
a real behavioral difference, not an unstable run.

## Read

Narrower is not simply safer. On the standard baselines, per-tick average
damage improved, but every other metric got worse, including a real
elimination reappearing in the harder live-opponent test (1/10, which v2
had eliminated entirely) alongside higher opponent-contact time. A
correction with less range has less power to react meaningfully when a
real course change is actually needed (e.g. avoiding a closing
competitor) -- it can still shave off the occasional risky moment (hence
the lower average damage) but can't reliably resolve the more demanding
situations, unlike v2's wider (but not too wide) correction.

**Combined with the `scale=0.45` result, this closes out the residual-
scale axis:** three points now tested (0.15, 0.3, 0.45), and `0.3` (v2)
wins outright or ties on nearly every metric in both directions tested.
This is a genuine local optimum for this axis, not an arbitrary starting
guess that happened to be first.

## Decision and rationale

**Not adopted.** `2026-09-11_residual-expert-base-v2-seed8000`
(`residual_scale=0.3`) remains the best combined-approach checkpoint.
Recommending the residual-scale axis be treated as closed -- further
concrete improvement should come from a different lever entirely (e.g. a
seed sweep for genuine network-initialization diversity, which is the
project's own best-precedented way of finding real gains, per the plain
SAC track's causal tests 24-32), not further points on this axis.

## Next steps

1. Seed sweep: sample a few fresh genuine-init seeds at the locked-in
   v2 config (residual_base, scale=0.3, recovery-passthrough fix) to
   look for a checkpoint that's faster and/or safer than seed=8000's
   result, mirroring the SAC track's own successful methodology.
2. Otherwise, treat v2 as the practical best combined-approach result.
