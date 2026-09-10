# Full current reward, seed 110 from scratch (2026-09-01)

Proposed at the end of the seed-909 causal-test chain: train seed 110 (the
only checkpoint so far with zero eliminations) from scratch with the full
current reward -- idle penalty, `WEIGHT_TERMINAL_PENALTY = 100.0`, speed
cap (`MAX_REWARDED_SPEED_MPS = 10.0`), and the wall-avoidance changes
(`WEIGHT_WALL_PROXIMITY = 1.0`, `WALL_WARNING_DISTANCE_M = 6.0`) -- to see
whether today's reward changes help, hurt, or don't matter on a seed that
wasn't already broken. Same config as the seed-909 tests: races=10,
round_seconds=120, buffer_capacity=200000, eval_round_seconds=120.

## Result: a genuinely different picture -- real speed, still unreliable

Not a repeat of seed 909's near-total failure, and not a clean repeat of
the old seed-110 success either. Per-race breakdown (vs. `crash_fast`):

| seed | race | laps | damage | best lap time | max speed |
| --- | --- | --- | --- | --- | --- |
| 110 | 1 | 8 | 0.385 | 17.9s | 26.4 m/s |
| 110 | 2 | 8 | 0.319 | 20.3s | 29.9 m/s |
| 42 | 1 | 3 | **1.000** | 16.7s | 34.8 m/s |
| 42 | 2 | 1 | **1.000** | 22.3s | 33.7 m/s |
| 7 | 1 | 0 | **1.000** | -- | 34.3 m/s |
| 7 | 2 | 9 | 0.329 | 15.8s | 25.4 m/s |
| 2024 | 1 | 8 | 0.422 | 20.2s | 25.6 m/s |
| 2024 | 2 | 4 | **1.000** | 19.8s | 25.7 m/s |
| 8675309 | 1 | 0 | **1.000** | -- | 30.9 m/s |
| 8675309 | 2 | 0 | **1.000** | -- | 35.3 m/s |

Aggregate vs. `crash_fast`: avg scored distance 835.1m (max 1695m), 6/10
eliminated, avg max speed 30.2 m/s, 10/10 wins. Aggregate vs.
`default_student_controller` (the strong heuristic, ~5 m/s sustained):
avg scored distance 841.8m, 5/10 eliminated, **6/10 race wins** -- the
**first time any SAC checkpoint has beaten this baseline at all**, let
alone in a majority of races.

## What this means

Two things are true at once:

1. **Real competitive speed emerged.** Best lap times of 15.8-22.3s for
   the 183m track are ~9-11.6 m/s average pace -- right around
   `MAX_REWARDED_SPEED_MPS`, exactly what the reward is supposed to
   encourage. In races it doesn't crash, it completes 8-9 laps in 120s and
   comfortably beats `default_student_controller`. Max speed (25-35 m/s)
   is also meaningfully lower than every seed-909 checkpoint tested today
   (38-47 m/s), suggesting the wall-avoidance/terminal-penalty changes
   *did* have a real, positive effect here.
2. **Reliability is still the open problem.** 6/10 races end in full
   elimination -- some after productive laps (3-4 laps, then a crash),
   some almost immediately (seed 7 race 1, seed 8675309 both races: near-
   zero off-track/wall-contact time, consistent with a fast, direct crash
   very early). This is a genuine ~40% success rate, not "basically
   fixed."

This is a materially different outcome from every seed-909 test today
(which stayed at ~90-100% elimination regardless of the same reward
changes) -- supporting the hypothesis from the previous entry that seed
909 was stuck in a resistant local optimum specific to its own training
trajectory, rather than the reward being broadly ineffective. The same
reward produces a much better outcome starting from seed 110.

## Decision and rationale

This is the new most interesting checkpoint, but not an unambiguous
"best" one: it's much faster and can beat the strong baseline, but is
less *reliable* than the old zero-elimination seed-110 checkpoint
(0/10 eliminated at ~6.9 m/s). Which is "better" depends on what's being
optimized -- average distance over many races favors this one heavily
(835m vs. 100.7m even counting the crashes); guaranteed survival favors
the old one. Given the course rubric's emphasis on reliability "across
random starting-point seeds," 40% elimination is not yet acceptable, but
this is real progress worth building on rather than a failure to
document and move past.

## Next steps (proposed, not yet run)

1. The problem has shifted from "no speed/no wall-avoidance" to
   "inconsistent -- crashes in ~40% of races." Investigate what
   distinguishes the crash races (7-race-1, 8675309-both) from the
   success races -- e.g. do the crash races share a spawn-point or early-
   track feature that's harder to handle.
2. More training (more races/updates) on this same reward+seed
   combination, to see if reliability improves with more gradient steps
   now that the reward is pointed in a productive direction.
3. Run the same full reward on 2-3 more seeds to see if this
   speed-with-partial-reliability pattern is general or specific to seed
   110 responding well.
4. Revisit whether `WEIGHT_TERMINAL_PENALTY = 100.0` and the wall-
   avoidance changes should be tuned further now that they're clearly
   having *some* effect (unlike on seed 909), rather than assuming they
   need another large jump.
