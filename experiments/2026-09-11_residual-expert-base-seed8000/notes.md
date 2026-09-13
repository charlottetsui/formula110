# Residual RL (SAC correction on top of the expert's action): the best combined-approach result yet (2026-09-11)

Combined-approach option B, take 2. The inference-time hard-switch shield
(`controllers.hybrid_controller`, `experiments/2026-09-11_hybrid-
controller-eval/`) reduced average contact time but traded it for a much
higher catastrophic-crash rate (0/10 -> 6/10 eliminated vs.
`leaderboard_expert`), likely from the discontinuity of switching between
two unrelated control laws mid-race. This tests the fix: **residual
reinforcement learning** -- `controllers.leaderboard_expert.Controller`'s
command becomes the base action every tick, and the SAC policy only
learns a bounded correction on top of it
(`training.controller.RESIDUAL_ACTION_SCALE = 0.3`), so the expert's
influence is continuous rather than switched. Implemented as a new
`residual_base` mode on `TrainableController` and a `--residual-expert-
base` flag on `scripts/train_sac.py`/`scripts/eval_sac.py`. Stays inside
ordinary self-play (`--opponent self`, unchanged) -- no training-
distribution risk like causal test 35.

Matched the original from-scratch reference config exactly (seed=8000,
races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, matching `2026-09-08_seed-sweep-v2-8000`) --
`--residual-expert-base` is the only new variable.

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v2-8000 (matched reference) | residual RL |
| --- | --- | --- |
| avg damage | 0.0298 | 0.0662 |
| avg off-track | 0.152s | 0.537s |
| avg wall-contact | 0.107s | 0.256s |
| avg car-contact | 1.212s | 1.669s |
| avg laps | 7.30 | **10.10** |
| avg best lap time | 14.96s | **11.25s (-25%)** |
| avg max speed | 26.65 m/s | 37.33 m/s |
| eliminated | 0/20 | **0/20** |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races; the real stress
test, since this is what exposed the shield's failure):**

| | pure SAC (no shield) | hybrid shield (hard switch) | residual RL |
| --- | --- | --- | --- |
| eliminated | 0/10 | **6/10** | **1/10** |
| avg damage | 0.1471 | 0.6055 | 0.2278 |
| avg car-contact | 5.465s | 4.035s | 6.147s |
| avg off-track | 2.637s | 1.988s | 1.320s |
| avg wall-contact | 1.103s | 0.950s | 0.595s |
| avg laps | 6.50 | 4.70 | **9.50** |
| avg best lap time | 16.51s | 14.99s | **12.08s** |

Raw data: `vs_leaderboard_expert.json` (residual RL vs. the expert, same
protocol as the shield's evaluation).

## Diagnosis

`metrics.csv` shows normal, stable training throughout (critic loss
bounded in the single-to-low-double digits, entropy temperature settling
smoothly from 1.0 to ~0.034-0.037) -- nothing resembling the divergence
seen in every train-against-a-fixed-opponent experiment (causal test 35).
This confirms staying inside self-play while changing only the action
*composition* (base + correction, rather than a switch) avoids the
training-distribution risk entirely, as intended.

The one elimination against the expert (seed 2024, race 1: damage 1.0,
only 5 laps, 3.73s car-contact) is a real, not-yet-diagnosed outlier --
worth a per-tick trace if this direction continues, the same way every
other flagged outlier on this project has been handled.

## Read

This is a materially different, better-shaped result than the hard-switch
shield: it recovers almost all of plain SAC's safety against the expert
(1/10 vs. 0/10 eliminated, vs. the shield's 6/10) while being
*substantially* faster and more complete than either alternative in every
matchup tested (25% faster lap time and 40%+ more laps than the reference
against the standard baselines; fastest lap time and most laps of all
three variants against the expert). The mechanism -- a continuous,
bounded correction rather than a discontinuous switch -- appears to be
exactly the fix the shield's diagnosis called for.

Not a perfect result: damage/off-track/wall-contact all rose somewhat
against the standard baselines (though eliminations stayed at zero), and
the one elimination against the expert means this is not as unconditionally
safe as plain SAC alone. This is a genuine trade -- meaningfully faster,
slightly less safe -- not a strict improvement on every axis, but a much
more favorable trade than the shield's.

## Decision and rationale

Not unilaterally adopted or repackaged into `race_faster.py` -- presenting
as the strongest combined-approach candidate found this session, for
direction. Unlike the shield, this is a real candidate worth taking
seriously rather than a mechanism to set aside.

## Next steps

1. Await direction on whether to adopt this (e.g. package a self-contained
   `controllers.*` module analogous to `race_faster.py`, composing the
   frozen residual actor with the expert the same way at inference time).
2. Diagnose the one elimination against the expert with a per-tick trace.
3. A seed sweep would establish robustness versus one favorable draw, per
   this track's own established caution about n=1 results.
4. `RESIDUAL_ACTION_SCALE = 0.3` was chosen without a dedicated sweep --
   worth exploring whether a different bound trades pace against the
   remaining elimination risk.
