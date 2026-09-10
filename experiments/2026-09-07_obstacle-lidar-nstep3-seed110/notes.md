# Add obstacle LiDAR to the observation (2026-09-07)

Directed to prioritize reducing hesitation. n=5 (see the sibling
`2026-09-07_nstep5-seed110/` entry) regressed rather than helping, so
before guessing at another reward-side fix, ran a per-tick diagnostic
(`sensor_sample_callback` logging `contact.robot`, speed, position across
a real evaluation race with the `2026-09-07_nstep3-seed110` checkpoint)
instead. Found `wall_contact` was exactly 0.00s across every evaluated
race, while `contact.robot` repeated in windows spaced ~180-190m apart --
matching the track's ~183m lap length -- meaning the policy was colliding
with the opponent car (stationary, for the `crash_fast` baseline) at
roughly the same point on the track once per lap, every lap.

Root cause: `training/observation.py`'s 17-dim vector has zero information
about other cars. `camera.competitors` and `sensors.lidar` (the only
public fields that detect other robots) were both deferred in the
original design ("opponent-aware features only matter once a policy can
already hold the track solo") -- solo driving reached that bar back on
2026-09-01 (zero damage/off-track/wall-contact ever since) but the
deferred feature was never revisited. The policy has been driving
completely blind to other cars this whole time.

A steering-direction-reversal reward term (`WEIGHT_STEERING_REVERSAL`) was
implemented earlier in this session under the (wrong, pre-diagnosis)
hypothesis that hesitation was steering wobble -- disabled (weight 0.0,
mechanism kept in `src/training/reward.py`) once this diagnosis landed, so
it wouldn't confound this test.

## Change

Added `sensors.lidar` (7 beams: -90/-45/-20/0/20/45/90 degrees, same
encoding as the existing `wall_lidar`) to the observation vector.
`OBSERVATION_DIM` 17 -> 24. `sensors.lidar` detects "nearby barriers,
robots, and blockers" per its docstring, so it's the minimal way to give
the policy opponent-visibility without a bigger redesign (no change to
`camera.competitors` handling, which would need variable-length-tuple
encoding). Same seed (110), races=40, round_seconds=120,
buffer_capacity=800000, reward, n_step=3 as the `2026-09-07_nstep3-
seed110` reference -- the observation change is the only new variable.
Verified with 168 passing tests (11 new for the n-step work, 2 new for
this observation change) and a real, tiny smoke race before committing to
the full run (necessary since this changes the network's input dimension,
not just a reward constant).

## Result: real improvement on 19/20 races, one flagged outlier

| | n-step=3 (no obstacle lidar) | + obstacle lidar |
| --- | --- | --- |
| avg car-contact | 1.595s | **0.952s (-40%)** |
| avg marshal/race | 0.25 | **0.10** |
| avg max speed | 14.73 m/s | 23.24 m/s |
| avg best lap time | 25.06s | 24.79s |
| avg low-progress (all 20 races) | 3.18s | 3.61s (worse, headline) |
| avg low-progress (19/20, outlier excluded) | 3.18s | **2.63s (better)** |
| avg damage | 0.0000 | 0.0027 |
| avg off-track | 0.000s | 0.027s |
| avg wall-contact | 0.000s | 0.009s |
| eliminated | 0/20 | 0/20 |

Per-race car-contact/low-progress breakdown showed a clean, consistent
improvement across 19 of the 20 evaluated races (low-progress in the
1.65-4.40s range throughout, vs. the reference's more scattered 1.35-5.97s
range) with one severe outlier: `seed=2024 vs default_student_controller
race=2` recorded low-progress=22.17s (5-10x every other race in this run),
the only nonzero wall-contact (0.18s) and real damage (0.054) anywhere in
the run.

Diagnosed the outlier directly (same per-tick logging approach, this
checkpoint, that seed/baseline/race) rather than discarding it as noise:
~54 of 120 seconds spent below 1.2 m/s. A contact window around tick
6007-6221 shows *negative* speed (reversing, -3.33 to -0.04 m/s) while
off-center (-0.5 to -0.67m) -- consistent with the car getting physically
wedged against the opponent near the track edge, not a brief bump-and-
recover like every other contact window in this run and the reference.
One marshal recovery fired during the race, but only after most of the
120s was already spent stuck.

## Read

The fix works as intended for the dominant case: car-contact time dropped
substantially and consistently, marshal interventions dropped by more
than half, and lap time held or slightly improved -- all while max speed
rose sharply (23-26 m/s vs. 14.7 m/s), plausibly because the policy no
longer needs to drive as cautiously around a risk it previously couldn't
perceive at all. But it introduces or exposes a rarer, more severe
failure mode: getting stuck against a *moving* opponent is harder to
recover from than a stationary wall, and a single marshal reset doesn't
reliably resolve it quickly. This is a genuine, not-yet-solved trade-off --
reporting both sides rather than only the flattering 19-race subset.

## Decision and rationale

Adopting `2026-09-07_obstacle-lidar-nstep3-seed110` as the new best/
reference checkpoint. The aggregate improvement is real and consistent
across seeds and baselines; the outlier is high-cost but rare (1/20) and
non-fatal (no elimination, no marshal-limit failure). Flagging the
stuck-against-opponent failure mode as an open item, not a solved one.

## Next steps

1. A small seed sweep at this checkpoint would help tell whether the
   stuck-against-opponent case is specific to one spawn/track geometry or
   a general risk -- no checkpoint on this track has had a proper
   multi-seed sweep yet.
2. Consider a reward-side complement now that the observation supports
   it: a proximity-based penalty for closing distance on a competitor
   (mirroring `WEIGHT_WALL_PROXIMITY`'s speed-scaled design), teaching
   the policy to avoid getting close in the first place rather than only
   giving it the sensory input to react once close.
3. Re-test the disabled `WEIGHT_STEERING_REVERSAL` mechanism on its own
   now that the dominant hesitation cause has a different fix -- any
   residual steering wobble is now measurable in isolation from the
   collision effect.
4. Still open: a genuine multi-training-seed sweep, and repackaging
   `controllers.race_faster` (still ships the races=20 checkpoint from
   2026-09-01).
