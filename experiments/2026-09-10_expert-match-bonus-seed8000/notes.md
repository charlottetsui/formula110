# Expert-match reward bonus, restricted to hazard states: a real, modest improvement (2026-09-10)

Combined-approach idea #1, implemented after every training-*opponent*
combination (causal test 35 and its fine-tuning follow-up) failed: reward
the SAC policy for choosing an action close to what
`controllers.leaderboard_expert.Controller` (Lucy's rule-based expert)
would choose, but only during the exact wall/robot-proximity hazard states
`training.reward` already tracks (`_wall_proximity_penalty`/
`_robot_proximity_penalty` > 0). Implemented as `WEIGHT_EXPERT_MATCH` in
`src/training/reward.py`, a `--expert-match-bonus` flag on
`scripts/train_sac.py`, and a private, non-controlling shadow instance of
the expert inside `TrainableController` (`src/training/controller.py`) fed
the same sensors every tick purely to compute the bonus.

**Critically, this run uses ordinary self-play (`--opponent self`, the
default) — nothing about who SAC races against changed.** This was the
whole point: every failure today came from changing the training
*distribution* (a fixed, unfamiliar opponent); this mechanism only adds a
reward term inside the proven-stable self-play loop.

Matched the *original* from-scratch config of the current reference
checkpoint's lineage (`2026-09-08_seed-sweep-v2-8000`: seed=8000, races=40,
round_seconds=120, n_step=3, hidden_size=128, buffer_capacity=800000) —
`--expert-match-bonus` is the only new variable versus that run.

## Result

| | v2-8000 (from-scratch reference) | expert-match-bonus (this run) |
| --- | --- | --- |
| avg damage | 0.0298 | **0.0129 (-57%)** |
| avg off-track | 0.152s | 0.372s (higher) |
| avg wall-contact | 0.107s | 0.236s (higher) |
| **avg car-contact (target metric)** | 1.212s | **1.136s (-6%)** |
| avg laps | 7.30 | 7.30 (tied) |
| avg best lap time | 14.96s | 14.98s (tied) |
| avg max speed | 26.65 m/s | **35.11 m/s (+32%)** |
| eliminated | 0/20 | 0/20 (tied) |
| wins vs. both baselines | 20/20 | 20/20 (tied) |

Also compares favorably to `2026-09-08_seed8000-resumed-short` (the
checkpoint currently packaged in `race_faster.py`): car-contact 1.136s vs.
1.702s (better), lap time 14.98s vs. 15.54s (better), max speed 35.11 vs.
30.33 m/s (higher) — at the cost of avg damage 0.0129 vs. 0.0004 (both
negligible in absolute terms; neither approaches
`NEAR_ELIMINATION_DAMAGE = 0.9`, and both are 0/20 eliminated).

## Diagnosis

`metrics.csv` shows normal, stable training throughout — critic loss stays
in a bounded range (2.7 -> 11.7 -> 8.8 -> 7.9 -> 14.3 at even checkpoints)
and the entropy temperature settles smoothly (1.0 -> ~0.05 -> ~0.04),
nothing resembling the divergence seen in every expert-*opponent*
experiment earlier today. Per-race detail confirms the off-track/wall-
contact increase isn't one catastrophic outlier: 5 of 10 races evaluated
against `crash_fast` have exactly zero off-track/wall-contact time, and
the other 5 have modest amounts (0.4-1.6s off-track, 0-1.5s wall-contact)
spread across different seeds, with laps (7-8) and damage (max 0.049,
nowhere near the elimination threshold) staying solid throughout. Reads as
a genuinely faster policy (max speed up 32%) taking slightly more
incidental risk in some corners, not a hidden crash mode.

## Read

This is the first combined-approach experiment today where the target
metric (opponent-collision time) actually improved, alongside real gains
in damage and top speed, at essentially zero lap-time cost. The trade
(more off-track/wall-contact seconds, both still under 2s/race) looks
favorable given laps, lap time, damage, and elimination rate all held or
improved. This is a single run (one seed, one training run) -- per this
track's own established caution about n=1 results, this shows the
mechanism *can* produce a real improvement, not that it reliably will
every time.

## Decision and rationale

Not unilaterally repackaging `race_faster.py` -- presenting this as a
genuine candidate for the user's direction, consistent with how prior
judgment-call improvements on this track (e.g. causal test 27's seed 8000)
were handled. `--expert-match-bonus` is a real, working, tested addition
to the training loop regardless of whether this specific checkpoint is
adopted.

## Next steps

1. Await direction on whether to adopt this checkpoint (or repackage
   `controllers.race_faster` from it).
2. A seed sweep (matching the rigor already applied to the base self-play
   reward) would establish whether this is a robust improvement or one
   favorable draw -- not yet done.
3. The off-track/wall-contact increase, while small, is unexplained --
   worth a per-tick diagnostic on one of the nonzero races if this
   direction is pursued further.
4. `WEIGHT_EXPERT_MATCH`'s value (0.5) was chosen by analogy to
   `WEIGHT_WALL_PROXIMITY`'s scale, not tuned -- a dedicated sweep of this
   weight is untried.
