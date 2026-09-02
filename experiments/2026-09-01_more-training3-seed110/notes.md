# Push training further again: races 40 -> 80 (2026-09-01)

Continues the training-budget trend: same seed (110), same reward
(`MAX_REWARDED_SPEED_MPS = 10.0`, reverted after the speedcap20
regression), same 120s round length. Only variable changed:
`--races 40 -> 80` (buffer capacity 800000 -> 1600000 to match). Took
849.0s training (1,106,262 transitions, 276,316 gradient updates) --
roughly 2x the races=40 run's time for 2x the races, consistent with
every prior scaling step today.

## Result: safety maintained at floor, but speed and lap count both dropped

| | races=40 (prior best) | races=80 (this run) |
| --- | --- | --- |
| avg damage | 0.000 | 0.000 (unchanged, still perfect) |
| avg off-track time | 0.00s | 0.00s (unchanged, still perfect) |
| avg wall contact | 0.00s | 0.00s (unchanged, still perfect) |
| max speed | 15.4 - 16.1 m/s | **13.1 - 14.7 m/s (slower)** |
| laps completed | 4 - 5 | **2 - 3 (fewer)** |
| best lap time | 21.2 - 27.9s | **34.3 - 42.1s (slower)** |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

Damage, off-track time, and wall contact are all still exactly zero
across every one of 20 evaluation races -- more training didn't break
anything. But average max speed and lap count both moved in the *wrong*
direction for the stated goal (optimize for speed): slower laps, fewer of
them, in the same 120s window.

## Why, most likely

`MAX_REWARDED_SPEED_MPS = 10.0` means the reward stops crediting forward
progress once speed exceeds 10 m/s -- there is no reward benefit to going
faster than that, only unrewarded (and, if it goes wrong, punished) risk.
races=40's checkpoint was still driving at 15.4-16.1 m/s, meaningfully
above the cap, likely a residual of earlier, less-refined training rather
than a reward-seeking choice. More training continued optimizing toward
the actual reward-maximizing point, which is at or near the cap, not
above it -- so speed converged *down* toward ~13-15 m/s. This is
consistent with (and now further confirms) the previous entry's finding
that the reward has no structural incentive to exceed `MAX_REWARDED_SPEED
_MPS`: more training doesn't push speed up past a point the reward
doesn't reward, it pushes speed toward exactly that point.

## Decision and rationale

This is not adopted as "the" best checkpoint for the stated goal (speed).
It's arguably the single safest/most consistent checkpoint produced today
(same perfect safety record as races=40, with tighter/slower and
presumably even more conservative driving), but that's not what's being
optimized for right now. **races=40
(`2026-09-01_more-training2-seed110`) remains the better checkpoint for
"fast but safe."** More training on the unchanged reward has now
demonstrably hit its ceiling for the *speed* goal specifically -- it
converges toward the cap, not past it, and further training would likely
continue that convergence rather than reverse it.

This also sharpens the previous entry's failed speedcap20 experiment: the
fix isn't "cap=10 works, just train more" (this run shows more training
alone won't increase speed) and it isn't "raise the cap a lot" (that
regressed badly). The most promising untried option is a **smaller** cap
increase than the 2x jump tried before, giving the reward room to credit
higher speed without doubling the relative weight of speed vs. control.

## Next steps (proposed)

1. **(recommended)** Try a smaller speed-cap increase (e.g. 10.0 -> 12.0
   or 13.0, close to races=40's own organically-reached ~15-16 m/s rather
   than the 20.0 that regressed) and retrain, comparing against the
   races=40 reference directly.
2. Consider training from the races=40 *checkpoint* (fine-tuning) rather
   than from scratch each time, if the training script is extended to
   support resuming from a saved checkpoint -- not currently supported by
   `scripts/train_sac.py`, which always initializes a fresh `SACAgent`.
3. Revisit the standing "reduce hesitation" refinement-plan item as an
   alternative, non-reward-magnitude lever for lap time.
