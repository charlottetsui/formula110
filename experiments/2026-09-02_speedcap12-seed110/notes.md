# Smaller speed-cap increase: 10.0 -> 12.0 (2026-09-02)

Follow-up to the 2026-09-01 finding that doubling the cap (10.0 -> 20.0)
regressed badly, and that more training at cap=10.0 converges speed
*down* toward the cap rather than past it. Tested a much smaller step:
`MAX_REWARDED_SPEED_MPS` 10.0 -> 12.0 (20% increase, still below the
races=40 reference's own observed peak speed, ~16 m/s). Same seed (110),
races=40, round_seconds=120, buffer_capacity=800000, eval_round_seconds=120
as the reference -- trained from scratch.

## Result: essentially a tie with the reference, not an improvement

| | races=40, cap=10.0 (reference) | races=40, cap=12.0 (this run) |
| --- | --- | --- |
| avg damage | 0.000 | 0.000 (unchanged, still perfect) |
| avg off-track / wall contact | 0.00s / 0.00s | 0.00s / 0.00s (unchanged, still perfect) |
| laps completed | 4-5 (mixed) | **4 (every single race, more consistent)** |
| avg best lap time | 24.76s (range 23.6-27.0s) | **28.20s (range 26.2-31.7s) -- slower** |
| avg max speed | 15.53 m/s | 15.88 m/s (~2%, not meaningfully different) |
| avg raw distance/race | 891.2m | 792.5m (lower) |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

Safety is identical (still perfect). Lap completion is if anything more
consistent (every race hit exactly 4 laps, vs. the reference's mix of 4s
and 5s). But top speed did not measurably increase, and average lap time
got *slower*, not faster. This is not the improvement the change was
testing for.

## What this means

Three points on the `MAX_REWARDED_SPEED_MPS` axis have now been tried
from this same seed/config: 10.0 (reference, the best result so far),
12.0 (this run, a statistical tie or mild regression on lap time), and
20.0 (2026-09-01, a severe regression). None of them beat the reference.
Each training run is a fresh run from scratch with the same network
initialization but a different reward from tick one, so results can
diverge for reasons beyond the intended "give more credit for speed"
effect -- run-to-run variance from a single trial at each cap value can't
be fully separated from a genuine causal effect with n=1 per value. But
across all three tested points, **no cap value found so far measurably
beats the original cap=10.0's lap times**, which is the more important
signal than any individual run's noise.

## Decision and rationale

Not adopting this checkpoint. `2026-09-01_more-training2-seed110`
(races=40, cap=10.0) remains the reference/best checkpoint. Recommending
against further tuning of `MAX_REWARDED_SPEED_MPS` specifically as a lever
for this goal -- three attempts (10, 12, 20) have not produced a
measurable win, and the two non-baseline attempts both came in worse on
lap time. This looks like a genuine dead end for the speed goal via
reward-cap tuning, not a matter of finding the exact right value.

## Next steps (proposed, not yet run)

1. Treat `MAX_REWARDED_SPEED_MPS` tuning as exhausted for now; pursue a
   different lever if further speed gains are wanted:
   - The standing "reduce hesitation" refinement-plan item (penalize
     oscillating steering) targets lap time via smoother driving rather
     than raw speed incentive.
   - A `--resume-from` option for `scripts/train_sac.py` (not currently
     supported -- always initializes a fresh `SACAgent`) would let
     training continue from the races=40 checkpoint rather than
     restarting from scratch each time, which might refine cornering
     technique without the variance of a fresh random start.
   - Repeat cap=12.0 with a different seed to check whether the "no
     improvement" result generalizes or was this particular run's
     variance -- n=1 per cap value is thin evidence for a firm
     conclusion.
2. Alternatively, treat races=40 (cap=10.0) as a strong enough result on
   speed for now and shift focus to other open items: broader seed
   testing, packaging the current best checkpoint (still races=20 in
   `controllers.race_faster` as of the last packaging entry), or the
   `controllers.minimum_viable` module gap.
