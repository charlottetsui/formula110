# Cap the reward's speed term (2026-09-01)

Follow-up to `2026-09-01_stochastic-vs-deterministic-diagnosis`, which
rejected the eval-mode-artifact hypothesis and proposed: cap the speed
term in `forward_progress_m` so reward stops scaling with speed past some
point, removing the structural incentive to push speed indefinitely.

## What changed

`src/training/reward.py`: added `MAX_REWARDED_SPEED_MPS = 10.0`. The speed
used in `forward_progress_m` is now `copysign(min(abs(speed_mps), 10.0),
speed_mps)` instead of the raw (unbounded) `speed_mps`. Chosen from the
observed speed range of the only zero-elimination checkpoint so far (seed
110, avg max speed ~6.9 m/s) and the competent heuristic baseline
(~5 m/s): above both, so genuinely fast safe driving is still fully
rewarded, well below the 15-40+ m/s regime seen in every crashing
checkpoint. Everything else held fixed vs.
`2026-09-01_longer-training-round-seed909` (seed 909, races=10,
round_seconds=120, eval_round_seconds=120).

## Result: another clean null result on the metric that matters

| | pre-cap (longer-training-round) | post-cap (this run) |
| --- | --- | --- |
| avg scored distance vs crash_fast | 91.3 m | 84.5 m |
| eliminated | 10/10 | **10/10 (unchanged)** |
| avg max speed | 40.5 m/s | **38.8 m/s (essentially unchanged)** |
| avg off-track time | 1.0s (0.9%) | 0.2s (0.2%) |
| avg wall contact time | 0.6s | 0.1s |
| avg low-progress time | 0.8s (0.7%) | 1.1s (1.0%) |

The cap had essentially no effect on the outcome that matters. Max speed
barely moved and elimination rate is identical.

## Why: the terminal penalty is simply too small relative to a short burst

`WEIGHT_PROGRESS * MAX_REWARDED_SPEED_MPS * dt_s` = `1.0 * 10.0 / 60` =
**0.1667 reward per tick**, sustained at the (now-capped) maximum
rewarded speed. That means:

| sustained duration | cumulative progress reward | vs. `WEIGHT_TERMINAL_PENALTY = 10.0` |
| --- | --- | --- |
| 1.0s (60 ticks) | 10.0 | breaks even |
| 2.0s | 20.0 | **2x the penalty** |
| 3.0s | 30.0 | **3x the penalty** |
| 5.0s | 50.0 | **5x the penalty** |

Given observed scored distance (~85-92m) at the observed speed (~38 m/s),
the car covers that ground in roughly 2-3 seconds -- meaning the
cumulative progress reward from that burst alone likely already exceeds
the one-time terminal penalty, net positive for the whole race even after
dying. The speed cap changed the *marginal* reward for exceeding 10 m/s
(now zero), but did nothing to change this calculus, since the car isn't
being rewarded for exceeding the cap -- it's rewarded plenty just by
reaching and holding the cap for a couple of seconds, which is already
enough to make dying profitable.

## Decision and rationale

The speed cap alone doesn't fix this; keeping it anyway since it's not
harmful and closes off unbounded-speed reward-hacking in principle. The
real lever is the size of `WEIGHT_TERMINAL_PENALTY` relative to
achievable per-episode cumulative reward, not the shape of the progress
term. Not adopting this checkpoint (100% elimination, unchanged).

## Next steps (proposed, not yet run)

1. **(quantitatively motivated)** Raise `WEIGHT_TERMINAL_PENALTY`
   substantially -- the arithmetic above suggests it needs to be at least
   in the tens (50-100+) to reliably dominate a multi-second high-speed
   burst, not comparable to a single tick's reward.
2. Switch from "no additional credit above the cap" to an *active*
   penalty for exceeding it (e.g. `max(0, speed - cap)` subtracted with
   its own weight), giving a genuine disincentive rather than a neutral
   zone.
3. Add a hard action/speed cap at the controller level as a safety
   backstop independent of reward shaping -- guaranteed to work
   regardless of what reward tuning converges to, at the cost of no longer
   being a purely learned behavior.
