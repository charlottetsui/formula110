# n-step (n=3) returns vs. the 1-step reference (2026-09-07)

Directed to implement n-step returns in the SAC critic target and test
`n=3` as the first causal test, per the discussion of why 1-step TD
targets propagate delayed crash penalties slowly (`WEIGHT_TERMINAL_PENALTY`
only reaches earlier states via many sequential Bellman backups, which
several 2026-09-01/09-02 causal tests worked around by raising the
penalty's magnitude instead of shortening the backup path).

Implementation (`src/training/replay_buffer.py`, `src/training/sac.py`,
`src/training/controller.py`, `scripts/train_sac.py`): each
`TrainableController` now holds a small per-car sliding window of raw
1-tick transitions; once the window reaches `n_step` ticks (or the
episode terminates early), it emits one transition to the shared replay
buffer using the discounted sum of the window's rewards and a per-transition
bootstrap `discount = gamma**actual_n` (shorter than `gamma**n` when
termination truncates the window). `ReplayBuffer`/`ReplayBatch` gained a
`discounts` field so the critic target can use this per-sample value
instead of a shared scalar `gamma`. `n_step=1` is verified (unit test) to
reproduce the exact prior 1-step behavior; default CLI value is 1, so no
existing run's behavior changes unless `--n-step` is passed explicitly.
165 tests pass (11 new: replay buffer discount storage, critic-target
wiring, and four controller-level n-step windowing tests), `ruff`/`pyright`
clean, and a tiny real headless race (`--races 1 --round-seconds 5
--n-step 3`) confirmed the plumbing runs against the actual simulator
before committing to a full run.

Ran with `--n-step 3`, otherwise identical to the current reference
(`2026-09-02_uncapped-speed-scaled-risk-seed110`): seed 110, races=40,
round_seconds=120, buffer_capacity=800000, eval_round_seconds=120,
unchanged reward. Took 448.7s training (570,114 transitions, 142,318
gradient updates -- more transitions than the reference's 498,864 despite
identical race/round settings, consistent with this checkpoint driving
more competently and therefore surviving longer per race on average).

## Result: better than the reference on safety AND lap time, at a lower top speed

| | reference (1-step, uncapped speed) | n-step=3 |
| --- | --- | --- |
| avg damage | 0.0028 | **0.0000** |
| avg off-track | 0.082s | **0.000s** |
| avg wall contact | 0.076s | **0.000s** |
| avg max speed | 17.13 m/s | 14.73 m/s (-14%) |
| avg laps | 3.90 | **4.00** |
| avg best lap time | 28.46s | **25.06s (-12%, faster)** |
| avg marshal/race | 0.25 | 0.25 (unchanged) |
| avg scored distance | 768.1m | **869.8m** |
| eliminated | 0/20 | 0/20 (unchanged) |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

Every safety metric reached exact zero across all 20 evaluation races (5
seeds x 2 baselines, including the 4 held-out seeds) -- matching the best
safety result seen anywhere in this project (the original races=40
checkpoint before the speed cap was removed). Unlike every prior
speed-focused experiment, this wasn't a trade-off: lower top speed
*and* faster average lap pace *and* better safety, all at once, on the
same seed and reward as the reference.

The likely mechanism, consistent with the n-step motivation: a lower top
speed with a *faster* lap time implies less time lost elsewhere -- less
hesitation, fewer near-miss recoveries, or a more direct racing line
through corners -- rather than a pure straight-line speed gain. This is
exactly the kind of anticipatory, consequence-aware behavior n-step
returns are meant to encourage (the critic sees the wall-proximity cost
of an aggressive entry speed several ticks sooner, without needing
`WEIGHT_WALL_PROXIMITY` or `WEIGHT_TERMINAL_PENALTY` raised further).
Cannot confirm this mechanistically from headless stats alone -- would
need a live watch or per-tick sensor logging to see whether cornering
technique actually changed, as opposed to e.g. less time spent
oscillating on straights.

## Decision and rationale

Adopting `2026-09-07_nstep3-seed110` as the new best/reference
checkpoint -- it dominates the prior reference on every tracked metric
except top speed, where it's lower but the net effect (lap time) is still
better. This is the first change all week to *not* trade one goal off
against another (previously: every speed gain cost either safety or lap
consistency, and vice versa).

## Next steps

1. Watch the checkpoint live (`controllers.sac_candidate`, once pointed
   at this checkpoint) to check qualitatively whether cornering technique
   changed, per the open mechanism question above.
2. Try `n=5` or `n=10` as the next point on this axis now that `n=3` has
   shown a real, positive effect -- per the earlier discussion, larger n
   trades lower bias for higher variance and more off-policy staleness in
   the replay-buffer rewards, so this isn't guaranteed to keep improving
   monotonically the way training budget did.
3. Repackage `controllers.race_faster` from this checkpoint -- it is now
   the best available checkpoint by a clear margin, and the packaged
   module still ships the much older races=20 checkpoint from
   2026-09-01 (open since that entry).
4. Still open: a genuine multi-training-seed sweep at this config, to
   check whether n-step's benefit is specific to seed 110 or general
   (the same caveat that applies to every SAC-track result so far).
