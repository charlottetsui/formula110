# More training on seed 110 + the full reward (2026-09-01)

Follow-up to `2026-09-01_full-reward-seed110` (races=10, ~40% elimination
rate, real but unreliable speed): same seed (110), same reward (idle
penalty, `WEIGHT_TERMINAL_PENALTY = 100.0`, speed cap, wall-avoidance
changes), same round length (120s). Only variable changed:
`--races 10 -> 20` (and `--buffer-capacity 200000 -> 400000` to
accommodate the larger run without excess eviction).

## Result: reliability problem resolved, not just improved

**Zero eliminations across all 20 evaluation races** (5 seeds x 2 races x
2 baselines). Every single race completed **exactly 4 laps**. Full
per-race detail:

| | vs crash_fast (10 races) | vs default_student_controller (10 races) |
| --- | --- | --- |
| laps | 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 | 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 |
| damage range | 0.037 - 0.076 | 0.037 - 0.680 (one harder hit, still survived) |
| max speed range | 17.8 - 21.2 m/s | 17.9 - 21.2 m/s |
| off-track time | 0.8 - 1.0s (< 1%) | 0.8 - 1.6s (< 1.4%) |
| wall contact | 0.1 - 0.2s | 0.1 - 1.5s |
| best lap time | 21.2 - 27.9s | 21.2 - 27.5s |
| race wins | **10/10** | **10/10** |

Won every single evaluated race against both baselines -- 20/20 total.
Max speed settled to a controlled, consistent ~18-21 m/s (down from the
25-47 m/s range seen at races=10), and best lap times (21-28s, ~183m
track = ~6.5-8.6 m/s average lap pace) indicate genuinely competent,
repeatable cornering, not just a fast straightaway sprint.

**Checked the training/eval seed overlap** (training used seed 110, which
is also one of the 5 fixed evaluation seeds) -- the 4 genuinely held-out
seeds (42, 7, 2024, 8675309, never used for training spawns) show the
identical pattern: 4 laps and low damage in all 8 of their races. The
result is not an artifact of testing on the training seed.

## What changed vs. the races=10 run

Only the amount of training: 242,262 transitions and 60,316 gradient
updates (vs. 117,542 / 29,136 before -- roughly 2x, matching the 2x race
count), in 184.4s wall-clock. More experience with the same
already-productive reward converted a ~40%-elimination, high-variance
policy into a 0%-elimination, low-variance one. This directly supports the
"seed 909 was stuck, the reward works" reading from the previous entry,
and now additionally shows that *more of the same training* (not a
different seed, not a different reward) was enough to close the
reliability gap for seed 110.

## Decision and rationale

This is the new best checkpoint by every metric that matters: reliability
(0/20 eliminated, previously as bad as 100% on other seeds today),
consistency (4 laps in literally every race, both training and held-out
seeds), and competitiveness (beats `default_student_controller` in every
evaluated race, the first checkpoint today to beat it at all was 6/10 --
this one is 10/10). `controllers.sac_candidate` auto-selects it as the
newest checkpoint.

## Next steps (proposed, not yet run)

1. Investigate why exactly 4 laps every time -- is the round length (120s)
   now the binding constraint (i.e. would a longer round show 5+ laps
   consistently, or does something else cap progress around there)?
2. Try an even longer training run (more races) to see if performance
   keeps improving or has plateaued at this level.
3. Test on more seeds beyond the fixed 5 to build more confidence before
   treating this as leaderboard-ready.
4. Consider packaging this checkpoint as a proper submission-ready
   controller (self-contained, no `training/` dependency) per the
   refinement plan's packaging step, given it's now a genuinely strong
   candidate rather than a diagnostic checkpoint.
5. Watch it live (`controllers.sac_candidate`) for a qualitative check --
   all evidence so far is from headless stats.
