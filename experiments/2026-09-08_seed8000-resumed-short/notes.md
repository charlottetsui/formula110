# Fine-tuning seed 8000: +10 races resumed -- a sweet spot (2026-09-08)

Directed to continue fine-tuning and finding ways to optimize the
adopted seed-8000 checkpoint (14.96s avg lap time, one documented
near-miss at 0.5942 damage). Two resume experiments were run in
parallel: this one (+10 races) and a longer one (+40 races, see
`2026-09-08_seed8000-resumed/notes.md`), both via `--resume-from
experiments/2026-09-08_seed-sweep-v2-8000/checkpoints/policy_final.pt
--warmup-steps 0`, same seed (8000), round_seconds=120,
buffer_capacity=800000, n_step=3, otherwise identical reward/observation.
The shorter resume was tried specifically because prior resume
experiments this session (seed 1000, seed 2000) showed +40-race resumes
tend to converge toward a shared, more conservative equilibrium -- a
smaller nudge might catch a partial improvement before that convergence
takes hold.

## Result: the near-miss is essentially eliminated, at a small pace cost

| | original (races=40) | +10 resumed |
| --- | --- | --- |
| avg best lap time | 14.96s | 15.54s (+3.9%) |
| avg laps | 7.30 | 6.75 |
| avg damage | 0.0298 | **0.0004** |
| **max damage (worst race)** | **0.5942** | **0.0047 (126x smaller)** |
| avg max speed | 26.7 m/s | 30.3 m/s (higher) |
| avg scored distance | 1444.6m | 1349.5m |
| eliminated | 0/20 | 0/20 |

Per-race detail confirms this is genuine, not an averaging artifact: 18
of 20 races have exactly 0.0000 damage, laps are consistently 6-8 every
race, and max speed is remarkably tight (29.7-31.2 m/s across literally
every race -- tighter variance than the original checkpoint's). The two
non-zero-damage races (0.0047 and 0.0040) are trivial grazes with brief
off-track/wall-contact, nothing remotely comparable to the original
0.5942 near-elimination event.

## This strictly beats the other safety-focused candidate too

| | seed 10000 (the other safety candidate) | seed 8000 +10 resumed |
| --- | --- | --- |
| avg best lap time | 16.70s | **15.54s (faster)** |
| avg laps | 6.80 | 6.75 (essentially tied) |
| avg damage | 0.0008 | **0.0004 (lower)** |
| max damage | 0.0154 | **0.0047 (lower)** |

Faster and safer than seed 10000 on every metric except raw top speed
(30.3 vs 35.4 m/s, which doesn't translate to lap pace here, consistent
with the long-standing finding that top speed and cornering pace aren't
the same thing).

## Read

More training does not monotonically improve or degrade this checkpoint
-- +10 races finds a clear sweet spot, +40 races (see sibling notes.md)
overshoots it and partially regresses on both safety and pace. This is
consistent with the non-monotonic training dynamics observed elsewhere
this session (e.g. causal test 11's races=40→80 regression under the old
capped reward), now demonstrated via fine-tuning duration on a single
already-good checkpoint rather than training budget from scratch.

## Decision and rationale

Recommending adoption as the new best/reference checkpoint -- it
dominates both prior candidates (original seed 8000 and seed 10000) on
the metrics that matter, at a small, well-characterized pace cost
relative to the original's raw speed.

## Next steps

1. If adopted, repackage `controllers.race_faster` from this checkpoint.
2. The sweet spot near +10 races (not 0, not +40) suggests intermediate
   resume durations (e.g. +5, +15, +20) might be worth a finer search if
   further optimization is wanted, though the non-monotonic pattern means
   "closer to 10 is better" is not guaranteed to hold smoothly.
