# Residual RL seed sweep, seed=12000: dramatically safer, first-ever win over the expert, but not faster (2026-09-12)

Fourth draw in the seed sweep at the locked-in v2 config
(residual_base=True, residual_action_scale=0.3, recovery-passthrough
fix), fresh seed=12000. Goal remains closing the pace gap to Lucy's raw
expert (8.94s solo).

## Results

**Standard baselines (20 races):**

| | v2 (seed=8000) | seed=12000 |
| --- | --- | --- |
| avg damage | 0.0254 | **0.0020 (12.7x lower)** |
| avg off-track | 0.325s | **0.076s (4.3x lower)** |
| avg wall-contact | 0.140s | **0.007s (20x lower)** |
| avg car-contact | 1.468s | 1.118s (lower) |
| avg laps | 10.15 | 9.95 (essentially tied) |
| avg best lap time | 11.37s | 11.57s (essentially tied) |
| **fastest individual lap** | 10.33s | **10.23s (new record)** |
| eliminated | 0/20 | 0/20 |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races):**

| | v2 (seed=8000) | seed=12000 |
| --- | --- | --- |
| eliminated | 0/10 | 0/10 |
| avg damage | 0.2204 | **0.0190 (11.6x lower)** |
| avg laps | 9.80 | **10.00** |
| avg best lap time | 11.78s | 13.25s (slower) |
| avg car-contact | 4.623s | 5.465s (higher) |
| avg off-track | 1.320s | 0.473s (lower) |
| avg wall-contact | 0.595s | 0.207s (lower) |
| **wins vs. the expert directly** | 0/10 | **1/10 (first ever)** |

`metrics.csv` showed stable training (critic loss bounded, entropy
settling smoothly).

## Read

A genuinely different kind of result from every prior seed: this is by
far the safest checkpoint found on this track for the combined approach
-- an order of magnitude lower damage in both test protocols, and it's
the first checkpoint (of any kind: plain SAC, hybrid shield, or any
residual variant) to actually **win a race outright against
`leaderboard_expert`** (seed 8675309, race 2 -- team-sum scored distance
exceeded the expert's).

But it does not close the pace gap -- if anything it moves slightly the
wrong way against the expert specifically (13.25s avg vs. v2's 11.78s),
while roughly tying v2 on the standard baselines (11.57s vs. 11.37s,
though it does set the best individual-lap record seen anywhere on this
track, 10.23s). The safety gain and the pace goal appear to be in
tension for this seed: it seems to have converged to a more
damage-averse driving style that trades a bit of average pace (against a
fast, unpredictable opponent) for dramatically fewer risky moments.

## Decision and rationale

Not unilaterally adopted, but flagged as a genuinely strong alternative
candidate to v2 -- arguably a *better* checkpoint overall if safety is
weighted heavily (order-of-magnitude lower damage, comparable pace on the
easier matchup, and the only checkpoint to ever beat the expert outright),
even though it does not advance the specific pace-vs-Lucy's-expert goal
this sweep was aimed at.

## Consolidated seed-sweep read (5 seeds: 8000/v2, 9000, 10000, 11000, 12000)

No seed has come close to Lucy's raw solo pace (8.94s avg, measured vs.
crash_fast alone) -- every sampled seed's average best lap time stays in
the 11.4-13.3s range on both protocols, roughly 25-45% slower. The single
fastest individual lap found across all 5 seeds and both protocols is
seed=12000's 10.23s (standard baselines) -- still ~14% slower than
Lucy's average, let alone her own best case. This is a consistent
pattern across genuinely different network initializations, which is
evidence (not proof) that the gap is not primarily an initialization-luck
problem -- it likely reflects the reward function's structural balance of
speed against safety, which every sampled seed converges toward in some
form, rather than a specific seed simply not having found the right
policy yet.

## Next steps

1. Awaiting direction: continue sampling (a larger batch, one at a time
   per direction already given), adopt seed=12000 as a safety-focused
   alternative to v2, or conclude that closing the pace gap via seed
   sampling alone is unlikely to work and a different lever (e.g.
   reward-weight tuning specific to residual-mode training) would be
   needed to make further progress on pace specifically.
2. Still open: the `controllers.minimum_viable` module gap.
