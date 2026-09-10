# Extend training round to match eval length (2026-09-01)

Follow-up to `2026-09-01_terminal-penalty-seed909`'s finding that the
terminal penalty never fired during a 60s training round. Same seed (909)
and reward (idle + terminal penalties) held fixed; only
`--round-seconds 60 -> 120` changed (matching the 120s eval length),
`--buffer-capacity` raised 150000 -> 200000 for the larger run.

## Hypothesis (rejected)

If training rounds reach the same duration as evaluation, the terminal
penalty should get a chance to apply, and the policy should learn to avoid
whatever leads to elimination.

## Result: the opposite happened

| | 60s training (prior) | 120s training (this run) |
| --- | --- | --- |
| avg scored distance vs crash_fast | 31.8 m | 91.3 m |
| eliminated | 10/10 | **10/10 (unchanged)** |
| avg max speed | 15.9 m/s | **40.5 m/s** |
| avg off-track time | 5.6s (4.6%) | 1.0s (0.9%) |
| avg wall contact time | 4.2s | 0.6s |
| avg low-progress time | 5.8s (4.8%) | 0.8s (0.7%) |

Still eliminated in every evaluation race -- unchanged. But now reaching
**40.5 m/s average max speed** (up to ~146 km/h), roughly 6x the "good"
seed-110 reference run's ~6.9 m/s, and the very low off-track/wall-
contact/low-progress times suggest it crashes almost immediately (within
the first couple of seconds), likely by flooring the throttle in a
straight line into the nearest wall.

Sanity-checked this isn't a simulator artifact: read
`reset_robot_vehicle` (`src/racing/race/runtime.py`) -- marshal resets
correctly zero linear/angular velocity and clear forces on teleport, so
this isn't a spurious velocity spike from a reset. It looks like a real,
physically-realized outcome of the policy's chosen actions.

## What this means (best current hypothesis, not verified)

Extending the training round length increased self-play's own in-training
scored distance dramatically (0.0m in every prior run -> 913m/853m this
run, since self-play is stochastic/exploring and covers real ground). But
the *deterministic* evaluation policy (the tanh-squashed mean action, no
exploration noise) may have converged toward an extreme, rarely-actually-
sampled action (max throttle, minimal steering correction) that produces
very different, much more dangerous behavior than what was actually
experienced during noisy training exploration. This is a plausible
train/eval behavior mismatch, not obviously fixable by more training time
at the same settings -- it may need attention to the policy's entropy/
exploration schedule, or an explicit inference-time bound (e.g. clipping
top speed in the observation/action mapping) rather than only reward
changes. **Not yet investigated further.**

## Decision and rationale

Reverting course: this specific hypothesis (longer training round fixes
survival) is rejected by direct evidence -- it made the safety problem
worse on the metric that matters most (average max speed, a proxy for how
violent the eventual crash is), while leaving the elimination rate
unchanged. Not adopting this checkpoint. Pausing the seed-909 causal-test
chain here to get direction on which of several plausible next steps to
pursue, rather than continuing to guess -- see
`docs/lab_notebook.md`'s 2026-09-01 entry for the options laid out.

**Best checkpoint across all of today's SAC-track experiments remains**
`experiments/2026-09-01_scaled-training-budget` (seed 110): 0/10
eliminations, ~6.9 m/s average max speed, 10/10 wins vs. `crash_fast`.
`controllers.sac_candidate` auto-selects the newest checkpoint by file
time, which is now this run's (the dangerous one) -- point
`FORMULA110_SAC_CHECKPOINT` at the seed-110 run explicitly if watching it
live.
