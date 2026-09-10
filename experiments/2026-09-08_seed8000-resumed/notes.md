# Fine-tuning seed 8000: +40 races resumed -- overshoots the sweet spot (2026-09-08)

Companion experiment to `2026-09-08_seed8000-resumed-short/notes.md`
(+10 races) -- same `--resume-from
experiments/2026-09-08_seed-sweep-v2-8000/checkpoints/policy_final.pt
--warmup-steps 0`, same seed (8000), round_seconds=120,
buffer_capacity=800000, n_step=3, only `--races 10 -> 40`.

## Result: worse than the +10 resume on both safety and pace

| | original (races=40) | +10 resumed | +40 resumed |
| --- | --- | --- | --- |
| avg best lap time | 14.96s | 15.54s | 16.36s |
| avg laps | 7.30 | 6.75 | 7.05 |
| avg damage | 0.0298 | 0.0004 | 0.0138 |
| max damage (worst race) | 0.5942 | 0.0047 | 0.2764 |
| avg max speed | 26.7 m/s | 30.3 m/s | 34.4 m/s |

Still an improvement over the un-resumed original on max damage (0.2764
vs. 0.5942), but a real regression relative to the +10 resume on every
metric that matters -- the near-miss partially reappeared and lap time
got slower.

## Read

More training on this checkpoint is not monotonically better: +10 races
finds a clear sweet spot, +40 races overshoots it. Consistent with
non-monotonic training dynamics seen elsewhere this session.

## Decision and rationale

Not adopted -- `2026-09-08_seed8000-resumed-short` (+10 races) is
strictly better on every safety and pace metric that matters. Kept as
evidence of the non-monotonicity, not as a candidate checkpoint.
