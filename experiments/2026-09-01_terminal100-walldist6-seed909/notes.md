# Raise terminal penalty 10x + strengthen wall avoidance (2026-09-01)

Two bundled, thematically-related changes to `src/training/reward.py`,
both aimed at making the car handle wall proximity and elimination risk
better, vs. `2026-09-01_speed-cap-seed909` (seed 909 held fixed, races=10,
round_seconds=120, eval_round_seconds=120):

1. `WEIGHT_TERMINAL_PENALTY`: 10.0 -> 100.0. Motivated by the previous
   entry's arithmetic: at 10.0, as little as ~2s of driving at the capped
   speed already outweighs the penalty. 100.0 requires ~10s to break even.
2. `WEIGHT_WALL_PROXIMITY`: 0.5 -> 1.0, `WALL_WARNING_DISTANCE_M`: 3.0 ->
   6.0. Every crashing checkpoint so far drives at 15-40+ m/s; at those
   speeds a 3.0m warning distance is covered in a fraction of a second --
   nowhere near enough lead time to react. 6.0m gives ~0.6s of lead time
   even at the (capped) 10 m/s reward-speed cruising pace, and doubling
   the weight makes avoiding a wall compete with, rather than being
   dominated by, going forward.

## Result: essentially no change

| | pre-change (speed-cap) | post-change (this run) |
| --- | --- | --- |
| avg scored distance vs crash_fast | 84.5 m | 80.5 m |
| eliminated | 10/10 | **9/10** (marginal) |
| avg max speed | 38.8 m/s | **39.0 m/s (unchanged)** |
| avg off-track time | 0.2s (0.2%) | 2.7s (2.2%) |
| avg wall contact time | 0.1s | 1.8s |
| avg low-progress time | 1.1s (1.0%) | 12.7s (10.5%) |
| wins vs crash_fast | 10/10 | 9/10 |

The one race that avoided full elimination (seed 110, race 1) isn't a
genuine "successfully avoided the wall" story: damage reached 0.469 (a
serious partial hit) and then the car spent 107.1 of 120 seconds (89%)
in a low-progress/stuck state -- it looks like a heavy but non-fatal
impact left it effectively disabled for the rest of the race, not like it
detected and steered around a wall in time. Every other race (9/10) still
reached full elimination, at essentially the same extreme speeds (29-47
m/s) as before every reward change tested today.

## What this means

Three reward-tuning attempts in a row on seed 909 (speed cap, then this
combined terminal-penalty-and-wall-avoidance change) have left average max
speed pinned in the same 38-40 m/s range regardless of what the reward
does. That consistency across genuinely different reward shapes is itself
informative: it suggests seed 909's policy may be stuck in a strong,
resistant local optimum (a simple "floor it straight" behavior is easy to
represent and may have gotten reinforced early, before any of today's
fixes existed) that isn't responding to reward-shape changes alone within
this training budget (~17-21k gradient updates each time). This is
different from the earlier finding that a fix was untested (the terminal
penalty never firing at 60s) -- these three fixes clearly *are* active
during training now, and still haven't moved the needle much.

## Decision and rationale

Keep both changes (not harmful, and directionally correct even if not
sufficient alone). Do not adopt this checkpoint (9/10 elimination is still
unacceptable). Given the pattern across three consecutive attempts,
continuing to iterate on reward shape using only seed 909 risks over-
fitting conclusions to one seed's particular stuck optimum rather than
learning something general about the reward. The most informative next
step is likely to test the *current* (now substantially revised) reward
on a seed that hasn't shown this pathology -- seed 110, the only checkpoint
so far with zero eliminations -- to find out whether today's reward
changes help there, hurt there, or make no difference, rather than
continuing to chase seed 909 specifically.

## Next steps (proposed, not yet run)

1. **(recommended)** Train seed 110 from scratch with the full current
   reward (idle penalty, terminal penalty at 100.0, speed cap, wall-
   avoidance changes) and compare against its own prior result (0/10
   eliminations, ~6.9 m/s, 10/10 wins) to see whether today's changes
   preserve or degrade what was already working, and whether they still
   don't matter there either (which would point at something more
   fundamental than reward shape, e.g. entropy/exploration collapse).
2. Run a small seed sweep (3-5 seeds) at the current reward for a real
   sense of the outcome distribution rather than one more single-seed
   causal test.
3. Add a hard action/speed cap at the controller level as a safety
   backstop -- increasingly the most reliable lever given reward shaping
   alone has now made four attempts without controlling top speed.
