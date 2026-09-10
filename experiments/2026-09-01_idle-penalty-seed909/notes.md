# Causal test: idle penalty on the seed that had been freezing (2026-09-01)

Same configuration as `experiments/2026-09-01_scaled-training-budget-seed909`
(races=10, round_seconds=60, buffer_capacity=150000, eval_round_seconds=120,
seed=909 held fixed), with one change: `WEIGHT_IDLE = 0.2` added to
`src/training/reward.py` (penalizes `abs(speed_mps) < 0.5`).

## Result: the freeze is fixed, but it overcorrected into reckless driving

| | before (frozen) | after (idle penalty) |
| --- | --- | --- |
| avg scored distance vs crash_fast | 2.2 m | 31.8 m |
| avg damage | 0.000 (every race) | **1.000 (every race)** |
| avg off-track time | 0.0s (every race) | 5.6s (4.6%) |
| avg wall contact time | 0.0s (every race) | 4.2s |
| avg low-progress time | 32.2s (26.8%) | 5.8s (4.8%) |
| max speed | ~4-5 m/s | **11.2-19.0 m/s** |
| race wins vs crash_fast | 5/10 | 10/10 |

The causal test succeeded at its narrow goal: the idle penalty clearly
broke the "do nothing" local optimum (wins 5/10 -> 10/10, distance 2.2m ->
31.8m, low-progress time way down). But `damage == 1.0` (full elimination)
in **every single one of the 10 evaluation races**, and max speed jumped
2-4x. This seed's policy now sprints (up to 19 m/s) and dies every race,
instead of freezing every race.

## What we observed

Off-track/wall-contact/low-progress times are all *lower* than seed 110's
"good" reference run, which looks good in isolation but is misleading: an
eliminated car stops being tracked for the rest of the round (the
simulator stops calling its controller), so these trackers simply had less
time to accumulate before the car died. The real story is in
`raw_distances_m` (9-82m, well short of the ~140-175m seed-110 achieved)
and `damages` (1.000 every time).

## Hypothesis for why this happened (not yet tested)

The simulator stops calling an eliminated car's controller. Combined with
a per-tick idle penalty, that creates an asymmetry: dying early *ends*
`WEIGHT_IDLE`'s accrual for the rest of the round, while surviving-but-
cautious keeps paying it every tick. For a long enough round, "sprint and
crash early" can look cheaper in cumulative reward than "survive idly."
`WEIGHT_DAMAGE` (scaled by the *delta* in one tick) doesn't clearly
dominate that calculus on its own.

## Decision and rationale

Do not adopt this checkpoint (100% elimination rate is unacceptable).
Confirmed the idle-penalty mechanism works as intended (fixes freezing)
but is not sufficient alone -- the reward now needs a way to make death
itself unambiguously costly, not just neutral-by-comparison to a long run
of idle penalties. Proposed next: add a one-time terminal penalty on top
of the existing delta-based damage penalty. See
`experiments/2026-09-01_terminal-penalty-seed909/notes.md` for that test
and an important, unexpected finding about *why* it had zero measurable
effect.
