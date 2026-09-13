# Weighting progress more heavily backfires: slower AND less safe (2026-09-12)

Tested a genuinely different lever from the residual-scale sweep and the
seed sweep: reweight `training.reward`'s speed-vs-caution balance
specifically for training, via the new `--progress-weight` flag
(`WEIGHT_PROGRESS` override, training-only -- eval never computes reward).
Rationale: the caution terms (`WEIGHT_CENTER_OFFSET`,
`WEIGHT_WALL_PROXIMITY`, `WEIGHT_DAMAGE`, etc.) were tuned entirely for
plain self-play, where the network supplies 100% of its own collision
avoidance -- in residual mode the base action already comes from a
competent expert, so the hypothesis was that the correction might not
need as much built-in caution relative to speed. Tested
`progress_weight=1.5` (up from the default 1.0), same matched config as
v2 otherwise (seed=8000, races=40, round_seconds=120, n_step=3,
hidden_size=128, buffer_capacity=800000, residual_action_scale=0.3,
recovery-passthrough fix already in code).

## Result

| | v2 (progress_weight=1.0) | progress_weight=1.5 |
| --- | --- | --- |
| avg damage | 0.0254 | 0.0569 (worse) |
| avg off-track | 0.325s | 0.532s (worse) |
| avg wall-contact | 0.140s | 0.252s (worse) |
| avg laps | 10.15 | 9.70 (worse) |
| avg best lap time | 11.37s | **12.03s (slower, not faster)** |
| fastest individual lap | 10.33s | **11.48s (also slower)** |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

`metrics.csv` showed no instability (critic loss bounded, alpha settling
smoothly ~0.04) -- a real behavioral regression, not a broken run.

## Read

The opposite of the intended effect on every axis, including the one
this was specifically meant to improve. Weighting progress more heavily
did not make the policy drive faster -- it made it drive both slower and
less safely. A plausible mechanism: increasing the progress term's
relative magnitude doesn't teach the correction to find a faster line
through corners; it likely just makes the *reward* less sensitive to the
caution terms in relative terms, so the policy's tendency to hedge
against risk (which, per the residual-scale sweep, is largely what the
correction seems to specialize in) gets diluted without being replaced
by anything that improves the racing line itself.

This is the **third** consecutive combined-approach lever aimed at
closing the pace gap to Lucy's raw expert (8.94s) to fail cleanly: the
residual-scale sweep (wider and narrower both failed), the
network-initialization seed sweep (5 seeds, none closed the gap), and
now reward reweighting. All three are structurally different mechanisms
(how much the correction can act; which policy the correction started
from; what the correction is trained to optimize for), and all three
point the same direction.

## Decision and rationale

Not adopted. `2026-09-11_residual-expert-base-v2-seed8000` remains the
best combined-approach checkpoint (or `2026-09-12_residual-seedsweep-
12000` if prioritizing safety over pace). `--progress-weight` kept as a
tested, configurable parameter (default unchanged).

Recommending against further from-scratch training runs purely aimed at
beating Lucy's raw pace via reward/scale/seed tuning -- three genuinely
different axes have now converged on the same read: the current
balance point is not a local minimum that a nearby tweak escapes, it
looks like a real structural trade-off between the pace a
safety-unconstrained controller can hit and what a controller
balancing speed against damage/off-track/wall-contact can sustain.

## Next steps

1. If pace remains the priority, a genuinely different mechanism (not
   another value on scale/seed/reward-weight) would be needed -- e.g.
   reward shaping that specifically targets cornering technique rather
   than a global speed/caution trade, or accepting a higher damage/
   elimination rate as the deliberate cost of matching Lucy's pace.
2. Otherwise, treat this line of investigation as concluded: v2 (pace-
   balanced) and seed=12000 (safety-focused) are the two combined-
   approach checkpoints worth keeping, and neither is expected to close
   the gap to Lucy's raw, safety-unconstrained speed without giving up
   the safety balance the whole combined-approach effort was built
   around.
