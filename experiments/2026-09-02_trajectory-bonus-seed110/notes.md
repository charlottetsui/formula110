# Best-trajectory reward bonus (2026-09-02) -- severe regression, design bug found

New mechanism: `src/training/trajectory.py`'s `BestTrajectoryTracker`,
rewarding distance gained per tick relative to the best-known distance
ever reached at that same tick, across the whole training run (see the
module docstring for the full rationale -- a location/time-specific
"beat your own record" signal, motivated by four straight failures of
uniform, global reward-constant tuning on 2026-09-01/02). Wired in via
`TrainingState.trajectory` and a new `--trajectory-bonus` opt-in flag on
`scripts/train_sac.py` (default off, so existing behavior is unaffected
unless explicitly requested). Same seed (110), races=40, round_seconds=120,
buffer_capacity=800000 as the reference, `--trajectory-bonus` added.

## Result: the worst outcome of the day, a different failure shape

- Self-play's own training distance: **3.0m / 0.0m over 40 whole races**
  (every prior run's self-play total was in the hundreds to tens of
  thousands of meters).
- Evaluation: **0 laps completed in every one of 20 races.** Marshal
  recoveries as high as **56 in a single race** (previous worst across
  every experiment today was 21). Low-progress (stuck) time as high as
  111.5 of 120 seconds (93% of the race). Damage and max speed both
  swing wildly race to race (0.000-1.000 damage, 10.2-18.0 m/s) with no
  consistent pattern. Lost most races against `crash_fast` (previously
  10/10 or 20/20 in every single prior checkpoint) and all races against
  `default_student_controller`.

This is not the same failure shape as any prior regression today (which
were consistently "uniformly faster and crashier" or "uniformly slower
and more cautious," both coherent single strategies). This looks like
genuine training instability -- an incoherent, unstable policy that can't
settle into any strategy.

## Root cause: a same-tick, same-race cross-copy reward-corruption bug

Read `_run_headless_student_runtime_step` in
`src/racing/race/head_to_head.py`: within one physics tick, both cars in a
race are controlled sequentially in the same loop
(`for entry, controller, runtime in zip(entries, controllers, runtimes)`),
and self-play's two `TrainableController` copies share one
`TrainingState` -- including, now, one `BestTrajectoryTracker`. Whichever
car is processed first in a given tick calls
`BestTrajectoryTracker.update()`, writing its current distance as the new
"record" for that tick, *before* the second car's `bonus_m()` call reads
that same tick's record. The second car is therefore compared against a
"best" that was just set by its own rival, racing simultaneously in the
*same episode* -- not a genuinely separate, earlier best run. Since one
starting grid position is ahead of the other (README: "the car in the
outside lane starts ahead of the car in the inside lane"), this creates a
systematic, adversarial corruption: whichever copy tends to be ahead
keeps setting the bar, and the other is punished for "losing" a race that
was never a fair comparison to begin with -- self-play's two copies end
up fighting the shared tracker instead of cooperatively improving one
policy's own pace over time.

This is a genuine design flaw in the tracker's integration
(`TrainableController.__call__`), not a bad hyperparameter choice --
`WEIGHT_TRAJECTORY_BONUS`'s value was never the issue.

## Decision and rationale

`--trajectory-bonus` defaults to off, so this doesn't affect any existing
default behavior -- but do not use it as currently implemented. Not
adopting this checkpoint (worst of the day by every metric). The
underlying idea (reward relative to your own best-known pace) is still
well-motivated; the bug is specifically in *when* the shared record gets
written relative to when concurrent copies read it within a race, not in
the concept itself.

## Next steps (proposed, not yet run)

1. **Fix the tracker's update timing** before trying this again. Options:
   - Only update the record from a *fully completed* episode (after a car
     is eliminated or the round ends), using that episode's whole
     tick-by-tick trajectory, rather than updating continuously while
     other copies are still racing and reading it mid-episode.
   - Give each self-play *role* (challenger vs. incumbent) or each grid
     position its own tracker, so same-race cross-copy comparison can't
     happen -- though this weakens the "shared curriculum" benefit across
     copies.
   - Snapshot the record at the start of each race and only read from
     that snapshot for the whole race, applying updates afterward --
     decouples what's read during a race from what's written during it.
2. Also worth checking separately: whether warmup's random actions (the
   first `--warmup-steps` transitions, pure noise) can plant spurious
   early-tick records that later, better-controlled episodes then have to
   fight against -- not confirmed as a contributing cause here, but a
   plausible secondary issue worth checking once the primary bug above is
   fixed.
3. If fixing this well takes real design work, deprioritize it below the
   `--resume-from` checkpoint-continuation idea or simply treating
   races=40 as the practical best result -- this feature needs another
   design pass before it's worth a second training run.
