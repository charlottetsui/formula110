# Push training further again (2026-09-01)

Continues the training-budget trajectory: same seed (110), same reward,
same 120s round length as the last two entries. Only variable changed:
`--races 20 -> 40` (buffer capacity raised 400000 -> 800000 to match).

## Result: still improving, now essentially perfect on safety metrics

| training budget | avg damage | avg off-track | avg wall contact | avg max speed | avg laps | eliminated |
| --- | --- | --- | --- | --- | --- | --- |
| races=10 (~29k updates) | 0.746 | 1.88s | 0.92s | 30.2 m/s | 4.1 | 6/10 |
| races=20 (~60k updates) | 0.062 | 0.84s | 0.12s | 18.4 m/s | 4.0 | 0/10 |
| races=40 (~132k updates) | **0.000** | **0.00s** | **0.00s** | 15.5 m/s | 4.1 | 0/20 |

At races=40: **zero damage, zero off-track time, and zero wall contact in
every single one of the 20 evaluation races** (5 seeds x 2 races x 2
baselines) -- not just avoiding elimination, but not touching a wall or
leaving the track surface at all, in every evaluated race including all 4
held-out seeds. Max speed continued dropping (30.2 -> 18.4 -> 15.5 m/s)
and converged very tightly (15.4-16.1 m/s across all 20 races, vs. a much
wider spread before). Best lap times 23.6-27.0s (~183m track, ~6.8-7.7 m/s
average lap pace). Most races complete 4 laps in the 120s round; two
(seed 2024, both baselines) complete 5. 20/20 race wins against both
baselines, same as the races=20 run, but now with a materially larger and
more consistent margin (avg scored distance ~1740-1820m vs. ~1550-1650m).

Training took 413.2s (530,262 transitions, 132,316 gradient updates) --
roughly 2.2x the races=20 run's wall-clock for 2x the races, consistent
with earlier timing (physics/self-play simulation, not gradient
computation, dominates wall-clock).

## What this means

Performance has **not plateaued** -- doubling training budget a second
time produced further, still-meaningful gains, this time concentrated on
eliminating the last remaining safety incidents (small amounts of damage
and off-track time that races=20 still had) rather than on raw speed
(which continued to *decrease* slightly and tighten, suggesting the
policy is converging toward a stable, repeatable racing line rather than
searching for more speed). This is a genuine learning curve, not noise:
every tracked metric (damage, off-track time, wall contact, speed
variance) moved monotonically in the same direction across all three
training-budget levels tested today.

## Decision and rationale

This is the new best checkpoint, superseding races=20's. Given the clean
monotonic trend, it's plausible further training would continue to help,
but the marginal wall-clock cost is also growing (7:45 total for this run
vs. ~3:30 for races=20) and returns may be diminishing (races=40 already
achieved literally zero safety-relevant incidents across all seeds, so
there is limited further headroom on those specific metrics -- more
training from here would likely show up as speed/lap-count gains instead,
if any). Reasonable to pause the training-budget scaling here and consider
other directions (packaging, broader seed testing, or continuing further)
rather than assuming bigger is automatically better indefinitely.

## Next steps (proposed, not yet run)

1. Push training further still (e.g. races=80) to see whether the trend
   continues, now specifically watching lap count/speed since safety
   metrics are already at floor -- optional, given diminishing-returns
   consideration above.
2. Test on more seeds beyond the fixed 5 to build broader confidence.
3. Package this checkpoint as a self-contained, submission-ready
   controller (no `training/` dependency) -- this is now a strong enough
   result to be worth doing.
4. Watch it live for a qualitative check -- all evidence so far is
   headless stats.
