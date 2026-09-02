# Raise the speed cap to increase throttle/speed (2026-09-01) -- regression

Attempt to optimize for speed while keeping the checkpoint safe: raised
`MAX_REWARDED_SPEED_MPS` 10.0 -> 20.0 in `src/training/reward.py`, leaving
every safety-side term (`WEIGHT_DAMAGE`, `WEIGHT_WALL_PROXIMITY`,
`WALL_WARNING_DISTANCE_M`, `WEIGHT_TERMINAL_PENALTY`, `WEIGHT_CONTACT`)
untouched. Rationale at the time: the `2026-09-01_more-training2-seed110`
checkpoint (races=40, the reference) already reached 15.4-16.1 m/s max
speed with the old 10.0 cap -- exceeding it -- while its average lap pace
(~6.8-7.7 m/s) stayed well under even the old cap, suggesting the cap
wasn't limiting top speed, just not crediting sustained higher speed.
Trained from scratch, same seed (110), same races=40/round_seconds=120
config as the reference.

## Result: a clear regression, not an improvement

| | reference (cap=10.0) | this run (cap=20.0) |
| --- | --- | --- |
| avg damage | 0.000 | 0.14 - 0.66 (real damage, though no full eliminations in this sample) |
| avg laps completed | 4.1 | 0 - 2 |
| best lap time | 21.2 - 27.9s | 36.3 - 95.9s (**slower**, despite higher top speed) |
| max speed | 15.4 - 16.1 m/s | **34.4 - 37.7 m/s** |
| avg low-progress time | ~0s | 12.2 - 57.9s (10-48% of the race) |
| marshal recoveries | ~0/race | **0 - 21/race** |
| wins vs default_student_controller | 10/10 | **5/10** (lost 5 races -- first losses since the races=40 breakthrough) |

Raw top speed roughly doubled, but every other metric got worse, several
dramatically. Lap times got *slower* on average despite the higher top
speed -- consistent with a policy that accelerates hard in bursts but then
struggles through corners, gets stuck, and needs frequent marshal
recovery (up to 21 times in one race).

## Why, most likely

Raising the cap doubled the maximum achievable per-tick progress reward
(`WEIGHT_PROGRESS * 20.0 / 60 ≈ 0.333`, vs. `0.167` before) while every
control/safety-relevant term (`WEIGHT_CENTER_OFFSET`,
`WEIGHT_WALL_PROXIMITY`, `WEIGHT_IDLE`) stayed the same absolute size.
That shifts the *relative* balance of the reward toward raw speed and away
from staying centered/controlled, even though the intent was only to
extend credit for speeds the policy was already reaching safely. The jump
(10.0 -> 20.0, 2x) was likely too large a single step -- it re-opened the
same kind of speed-vs-control trade-off that plagued the seed-909 causal
chain earlier today, just from a different starting point.

## Decision and rationale

**Reverting** `MAX_REWARDED_SPEED_MPS` back to `10.0`. The evidence is
unambiguous: the cap was not the bottleneck on top speed (the reference
checkpoint already exceeded it), and raising it produced a worse
controller by every metric that matters, not a faster-and-still-safe one.
Not adopting this checkpoint. The reference
(`2026-09-01_more-training2-seed110`) remains the current best.

## Next steps (proposed)

1. Pursue speed improvements via **more training on the existing,
   already-productive reward (cap=10.0)** instead of reshaping the reward
   further -- today's training-budget scaling (races 10 -> 20 -> 40)
   showed a clean, monotonic improvement in both safety *and* consistency
   without ever needing a reward change; lap times may well keep
   tightening with more training the same way damage/off-track/wall-
   contact did.
2. If reward-side speed tuning is revisited, use a much smaller step
   (e.g. 10.0 -> 12.0, just above the reference's own observed max speed)
   rather than doubling, and compare directly against the races=40
   reference rather than training from scratch again.
3. Consider the standing refinement-plan item on reducing hesitation
   (penalizing oscillating steering) as a more targeted way to improve
   lap times without touching the speed/control balance.
