# Hybrid (SAC + expert shield) controller: a mixed result, not a clean win (2026-09-11)

Real-race evaluation of `src/controllers/hybrid_controller.py` (combined-
approach option B: SAC drives normally, `leaderboard_expert.Controller`
takes over via a hard switch during wall/robot-proximity hazards). Built,
unit-tested (8 new tests, all passing), and verified to run in a real
headless race before this evaluation. This file records the *outcome* of
testing it properly across the fixed seed set, which turned out to be more
nuanced than the initial single-race smoke test suggested.

## Method

Ran the hybrid against all three available opponents (`crash_fast`,
`default_student_controller`, `leaderboard_expert`) across the fixed
5-seed set, 2 races each (`race_count=2`, `round_seconds=120`) — the same
protocol `training.evaluation.evaluate_against_baselines` uses. Also ran
the *plain* SAC controller (`controllers.race_faster`, no shield) against
`leaderboard_expert` under the identical protocol, since that specific
matchup (pure SAC vs. the expert as a live opponent) had never actually
been tested before -- every prior "reference checkpoint" number only came
from racing against `crash_fast`/`default_student_controller`, both much
slower and less aggressive than `leaderboard_expert`.

## Results

**vs. crash_fast + default_student_controller (20 races, directly
comparable to the pure-SAC reference `2026-09-08_seed8000-resumed-short`):**

| | pure SAC (reference) | hybrid |
| --- | --- | --- |
| avg damage | 0.0004 | 0.0500 |
| avg car-contact | 1.702s | 2.652s |
| avg off-track | 0.155s | 0.090s |
| avg wall-contact | 0.152s | 0.033s |
| avg laps | 6.75 | 7.25 |
| avg best lap time | 15.54s | 15.08s |
| eliminated | **0/20** | **1/20** |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races each; this matchup
had no prior baseline, so both sides were tested here for the first time):**

| | pure SAC (no shield) | hybrid |
| --- | --- | --- |
| avg damage | 0.1471 | 0.6055 |
| avg car-contact | 5.465s | 4.035s |
| avg off-track | 2.637s | 1.988s |
| avg wall-contact | 1.103s | 0.950s |
| avg laps | 6.50 | 4.70 |
| avg best lap time | 16.51s | 14.99s |
| eliminated | **0/10** | **6/10** |

Raw per-race data: `hybrid_eval.json` (hybrid vs. all three opponents),
`pure_sac_vs_expert.json` (plain `race_faster`, no shield, vs. the expert).

## Read

**Not a clean win.** In both matchups, the hybrid *reduces average
time* spent in wall/car contact and off-track -- exactly what the shield
is meant to do -- but *increases the elimination rate*, in one case
dramatically (0/10 -> 6/10 against the expert). This is a specific,
recognizable pattern: fewer, shorter close calls on average, but the
close calls that do happen are more likely to be catastrophic.

**Likely mechanism:** the hard switch hands full control to a completely
different control law (a hand-tuned rule-based controller, never trained
alongside or with any awareness of SAC's own control style) the instant a
hazard is detected, and hands it back the instant it clears -- with no
continuity in the commanded throttle/steer across that boundary. If SAC
and the expert would have commanded meaningfully different actions at the
exact tick the switch fires (very plausible, since they're independently
tuned controllers with no shared training), that's a sudden discontinuity
in the car's control input at precisely the moment it's already close to
a wall or another car -- the worst possible moment for an abrupt command
change. This has not been directly confirmed with a per-tick trace (that
would be the natural next diagnostic step), but it's consistent with
every symptom observed: contact-time *down* (the expert's more decisive
avoidance response does work, on average) while elimination *up* (some
fraction of those handoffs themselves destabilize the car).

This is also consistent with a pattern that has shown up repeatedly
elsewhere on the SAC track (docs/rl_design.md section 6): safety-motivated
mechanisms here tend to produce all-or-nothing outcomes rather than smooth
trade-offs (e.g. `WEIGHT_ROBOT_PROXIMITY`'s two failed variants, the
`WALL_WARNING_DISTANCE_M` recalibration). A hard switch between two
unrelated control laws is a particularly literal way to introduce exactly
that kind of discontinuity.

## Decision and rationale

**Not adopted as a final controller as-is.** The mechanism works
correctly (routes as designed, passes every unit test, races successfully
end to end), but the net effect on real races is a genuine trade-off, not
an improvement -- and the specific failure mode (rarer but worse crashes)
is a worse shape of risk than the status quo, not a better one, even
though the "average case" metrics look encouraging.

## Next steps

1. The most likely fix is smoothing the handoff -- blend toward the
   expert's action proportionally to hazard severity (a "dimmer switch")
   instead of a hard cut, or ease the transition over a few ticks in
   either direction, rather than switching the full commanded action in
   one tick. This was flagged as an alternative when the shield was first
   proposed and not yet built.
2. A per-tick trace of one of the 6 leaderboard_expert eliminations would
   confirm or rule out the discontinuous-handoff hypothesis directly,
   the same way every other outlier on this project has been diagnosed.
3. `controllers.hybrid_controller` is left in the codebase as a real,
   working, tested artifact -- just not one to submit or race with yet.
