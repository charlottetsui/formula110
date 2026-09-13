# Curvature-aware center-offset penalty: fourth consecutive clean failure (2026-09-12)

The "more invasive" combined-approach direction, tried after three
consecutive clean failures on the residual-scale sweep, the network-
initialization seed sweep, and a flat progress-weight reweight (all in
docs/rl_design.md causal test 37). This targeted cornering technique
specifically rather than a global speed/caution dial: scaled
`training.reward`'s `WEIGHT_CENTER_OFFSET` penalty by upcoming bend
sharpness (`_bend_score`, the same formula `controllers.leaderboard_expert`
itself uses from lookahead offsets) instead of applying it uniformly --
full strength approaching a real corner, reduced to 30%
(`MIN_CENTER_OFFSET_SCALE`) on straights. Unlike the 2026-09-03 attempt on
the plain-SAC track (which halved this weight *uniformly*, including in
corners, and regressed badly), this only loosens the penalty where the
track is straight, leaving the load-bearing cornering signal untouched.
Same matched config as v2 otherwise (seed=8000, races=40,
round_seconds=120, n_step=3, hidden_size=128, buffer_capacity=800000,
residual_action_scale=0.3, recovery-passthrough fix, progress_weight=1.0
default).

## Result

**Standard baselines (20 races):**

| | v2 (uniform penalty) | curvature-aware |
| --- | --- | --- |
| avg damage | 0.0254 | 0.0603 (worse) |
| avg off-track | 0.325s | 0.563s (worse) |
| avg wall-contact | 0.140s | 0.188s (worse) |
| avg laps | 10.15 | 9.75 (worse) |
| avg best lap time | 11.37s | 11.56s (slightly slower) |
| fastest individual lap | 10.33s | 11.20s (slower) |
| eliminated | 0/20 | **1/20 (reintroduced)** |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert (10 races):**

| | v2 | curvature-aware |
| --- | --- | --- |
| eliminated | 0/10 | **1/10 (reintroduced)** |
| avg damage | 0.2204 | 0.1426 (better) |
| avg laps | 9.80 | 9.60 (worse) |
| avg best lap time | 11.78s | 11.98s (slower) |
| fastest individual lap | ~10.7-10.9s | 10.28s (close to the record, but not a new one) |

`metrics.csv` showed a critic-loss spike early in training (30.0 at the
first checkpoint, well above every other residual run's typical
single-to-low-double-digit range) that settled down by the end (6.5) --
not a clean, stable run throughout, though it didn't diverge outright.

## Read

The fourth consecutive lever to fail cleanly, and again not even a clean
trade-off -- it lost safety (an elimination reappeared in *both* test
protocols, which v2 had eliminated) without gaining speed on average
(both average and worst-case lap times got slower on the standard
baselines). The one genuinely encouraging data point -- a 10.28s
individual lap against the expert, close to the existing 10.23s record --
isn't enough to outweigh the reintroduced crashes and the early training
instability.

This result is informative beyond just "reject this lever": the
hypothesis was specifically that the *uniform* nature of the original
2026-09-03 failure (touching corners too) was why it regressed, and that
a curvature-gated version would avoid that failure mode. It didn't -- it
regressed in a similar direction anyway, just less severely. That
suggests the center-offset penalty (uniform or curvature-gated) isn't the
actual bottleneck on pace; something else about how the residual
correction handles full-track-width driving is the limiting factor, not
this specific penalty's shape.

## Decision and rationale

Not adopted; reverted. `2026-09-11_residual-expert-base-v2-seed8000`
remains the best combined-approach checkpoint (or `2026-09-12_residual-
seedsweep-12000` for the safety-focused alternative).
`--curvature-aware-center-offset` kept as a tested, configurable
parameter (default unchanged).

**Four structurally different levers have now failed to close the pace
gap to Lucy's raw expert (8.94s):** how much the correction can act
(residual scale, both directions), which network initialization it
starts from (5 seeds), what the reward optimizes for globally (progress
weight), and now a targeted cornering-specific reward shape. This is a
strong, consistent pattern across genuinely different mechanisms, not
one unlucky axis. Recommending this line of investigation be concluded
rather than attempting a fifth variation.

## Next steps

1. Recommending against further from-scratch experiments purely aimed at
   beating Lucy's raw pace via training-side tuning -- four consecutive,
   structurally different attempts have all failed, several in ways that
   actively regressed safety without any compensating speed gain.
2. v2 (pace-balanced, ~11.4s laps, 0 eliminations everywhere tested) and
   seed=12000 (safety-focused, first checkpoint to beat the expert
   outright) remain the two combined-approach checkpoints worth keeping.
3. If pace is still a priority going forward, it likely needs a
   fundamentally different approach outside this reward-tuning family --
   e.g. a genuinely different residual-composition scheme, or accepting
   that matching Lucy's raw pace requires giving up the safety balance
   this whole combined-approach effort was built around.
