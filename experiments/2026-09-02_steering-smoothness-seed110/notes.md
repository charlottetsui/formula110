# Penalize oscillating steering (2026-09-02) -- regression

After three consecutive failed attempts to raise speed via
`MAX_REWARDED_SPEED_MPS` (12.0, 20.0, and more training at 10.0 all
failed to beat the races=40 reference), tried a different lever: added
`WEIGHT_STEERING_SMOOTHNESS = 0.1` to `src/training/reward.py`, penalizing
tick-to-tick change in `imu.yaw_rate_degrees_per_s` (scaled by
`YAW_RATE_CHANGE_SCALE_DEGREES_PER_S = 200.0`, capped at 1.0) as a proxy
for jerky/oscillating steering -- `step_reward` only sees sensor
transitions, not the raw steer action, so this doesn't require threading
action data through `TrainableController`. Same seed (110), races=40,
round_seconds=120, buffer_capacity=800000 as the reference, trained from
scratch.

## Result: another clear regression

| | races=40, reference (no smoothness term) | races=40, +steering smoothness (this run) |
| --- | --- | --- |
| avg damage | 0.000 | 0.000 (unchanged, still perfect) |
| avg off-track / wall contact | 0.00s / 0.00s | 0.00s / 0.00s (unchanged, still perfect) |
| laps completed | 4.10 avg (4-5) | **2.00 (exactly 2, every single race)** |
| avg best lap time | 24.76s (23.6-27.0s) | **45.48s (43.0-61.3s) -- nearly double** |
| avg max speed | 15.53 m/s | **10.75 m/s (-31%)** |
| avg raw distance/race | 891.2m | 489.2m (-45%) |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

Safety is unaffected (still perfect). But speed, lap count, and lap time
all got substantially worse -- the clearest, most consistent regression
of any experiment tried today. Every one of the 10 evaluated races vs.
`crash_fast` completed exactly 2 laps at 10.6-10.8 m/s.

## Why, most likely

Penalizing raw tick-to-tick yaw-rate *change* can't distinguish jittery,
wasteful oscillation from a legitimate, necessary steering input for
cornering -- both involve the yaw rate changing quickly. The penalty
applied to real cornering just as much as to hesitation, so the policy
learned to turn more gradually and cautiously everywhere, which is much
slower through every corner, not smoother-but-still-fast driving. The
hypothesis that "hesitation" was costing lap time wasn't wrong in
principle, but this specific proxy (yaw-rate delta) measures "any turning
input," not "wasted/oscillating turning input" -- it needed a way to
distinguish the two that this implementation didn't have.

## Decision and rationale

Reverted `WEIGHT_STEERING_SMOOTHNESS` to `0.0` (kept the mechanism in the
code, disabled by weight, rather than removing it -- consistent with how
`MAX_REWARDED_SPEED_MPS` experiments were handled). Not adopting this
checkpoint. `2026-09-01_more-training2-seed110` (races=40, original
reward) remains the best checkpoint.

This is the fourth consecutive reward-tuning attempt aimed at the speed
goal (cap=12.0, cap=20.0, more training at cap=10.0, this steering-
smoothness term) that has failed to beat the plain races=40 reference.
That consistency is itself informative: the races=40 checkpoint appears
to sit in a fairly strong local optimum for this reward structure that
isolated reward tweaks, each tested as a fresh from-scratch training run,
haven't been able to improve on.

## Next steps (proposed)

1. **Recommend pausing further reward-tuning attempts at pure speed
   optimization** -- four attempts across two different levers (speed
   cap, steering smoothness) have now failed. Further attempts along
   similar lines have a low prior of success without new information.
2. If a smarter steering-smoothness proxy is wanted later: distinguish
   *sustained* turning (real cornering, large but consistent yaw rate)
   from *oscillating* turning (yaw rate repeatedly changing sign or
   direction in a short window) -- e.g. penalize sign changes in yaw rate
   rather than raw magnitude of change, which wouldn't penalize a held
   turn the way this implementation did.
3. Consider the `--resume-from` checkpoint-fine-tuning capability
   (`scripts/train_sac.py` currently always initializes fresh) as a
   fundamentally different mechanism -- continuing gradient descent from
   the already-good races=40 policy, rather than re-deriving a whole new
   policy from a random init under a modified reward each time.
4. Otherwise, treat races=40 as a strong, practical result and shift
   focus to other open items: broader seed testing, repackaging
   `controllers.race_faster` from the current best checkpoint, or the
   `controllers.minimum_viable` module gap.
