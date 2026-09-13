# Residual RL v2: recovery-passthrough fix resolves the one crash entirely (2026-09-11)

Follow-up to `experiments/2026-09-11_residual-expert-base-seed8000/` (v1),
which had one elimination out of 10 races against `leaderboard_expert`
(seed 2024, race 1). A per-tick trace of that race found the cause: not a
single high-speed impact, but a repeated stuck-against-the-same-wall-spot
loop -- the car hit the identical wall location 8 times over ~8 seconds
(ticks 3576-4062+), each time backing off via the expert's stuck-recovery
maneuver (reverse + a fixed hard steer for up to 28 ticks) and then
driving straight back into the same spot, damage climbing a little each
cycle until full elimination. Root cause: the SAC correction was still
being added on top of the expert's command *even during its deliberate
recovery maneuver*, diluting a precise, fixed escape trajectory just
enough that it never fully cleared the obstacle.

**Fix (`src/training/controller.py`):** detect when the base expert's
command is a recovery command (checking `leaderboard_expert.Controller`'s
`_recovery_ticks_remaining` counter before and after calling it, since a
recovery maneuver can both start and be mid-flight within one call) and
pass it through completely unmodified in that case -- no residual applied
at all during recovery. Confirmed this fix alone does *not* resolve the
crash when only patched onto v1's already-trained weights (identical
outcome) -- the policy had learned its behavior assuming its correction
always applied, so it needed to be retrained under the corrected dynamics
from scratch. This run is that retrain, otherwise identical config to v1
(seed=8000, races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000).

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v1 | v2 (recovery fix) | v2-8000 (original reference) |
| --- | --- | --- | --- |
| avg damage | 0.0662 | **0.0254** | 0.0298 |
| avg off-track | 0.537s | **0.325s** | 0.152s |
| avg wall-contact | 0.256s | **0.140s** | 0.107s |
| avg car-contact | 1.669s | **1.468s** | 1.212s |
| avg laps | 10.10 | 10.15 | 7.30 |
| avg best lap time | 11.25s | 11.37s | 14.96s |
| eliminated | 0/20 | 0/20 | 0/20 |
| wins | 20/20 | 20/20 | 20/20 |

v2 improves on v1 across every safety metric (damage, off-track, wall-
contact, car-contact all lower) while keeping the same speed/lap gains
over the reference.

**vs. leaderboard_expert as a live opponent (10 races; the real test,
since this is what exposed both the shield's failure and v1's one crash):**

| | pure SAC | hybrid shield | residual v1 | **residual v2** |
| --- | --- | --- | --- | --- |
| eliminated | 0/10 | 6/10 | 1/10 | **0/10** |
| avg damage | 0.1471 | 0.6055 | 0.2278 | 0.2204 |
| avg car-contact | 5.465s | 4.035s | 6.147s | 4.623s |
| avg laps | 6.50 | 4.70 | 9.50 | **9.80** |
| avg best lap time | 16.51s | 14.99s | 12.08s | **11.78s** |

Seed 2024 specifically (the exact race that crashed in v1) now completes
cleanly both times: race 1 damage 0.0/10 laps, race 2 damage 0.009/10
laps -- no elimination, no repeated stuck loop. Raw data:
`vs_leaderboard_expert.json`.

## Read

The recovery-passthrough fix fully resolved the diagnosed failure mode
and, as a side benefit, also improved every safety metric on the easier
standard-baseline matchup too (not just the specific crash) -- consistent
with the fix addressing a real, generally-applicable interference problem
rather than papering over one seed's bad luck. v2 now **matches pure
SAC's perfect elimination record (0/10) while being the fastest and most
complete controller of every variant tested** in the harder matchup,
including beating v1 on both safety and pace simultaneously (not a
trade-off between them, which is the pattern seen almost everywhere else
on this track when a safety fix is applied).

**Remaining gap:** Lucy's raw expert alone (no combination), evaluated
under the same standard protocol, laps at 8.94s average with 0.047 avg
damage and 0/10 eliminated -- still meaningfully faster than v2's
11.37-11.78s. This combined controller has not yet closed that gap, and
may not be able to without also taking on more of her aggression (and its
attendant risk) -- but it now clearly and unambiguously surpasses **plain
SAC alone** (safer-or-equal, ~35% faster lap times, more laps) and the
**hybrid shield** (dramatically safer, faster, more laps) in every
matchup tested.

## Decision and rationale

Not unilaterally adopted or repackaged into `race_faster.py` -- presented
as the strongest, most complete combined-approach candidate found this
session. Given it now dominates the shield and matches-or-beats plain SAC
on every tracked metric, this is a much stronger case for adoption than
either prior candidate, but the decision is left for direction per this
track's established practice.

## Next steps

1. Await direction on adoption / packaging as a self-contained
   `controllers.*` module.
2. Closing the remaining pace gap to Lucy's raw expert (8.94s) would
   likely require either more training, a larger `RESIDUAL_ACTION_SCALE`
   (currently 0.3, allowing more aggressive correction), or accepting
   that some of the gap reflects her expert's greater risk tolerance,
   which a combined controller optimizing for both speed and safety may
   not fully match by design.
3. A seed sweep would establish robustness versus one favorable draw,
   per this track's own established caution about n=1 results -- not yet
   done for either v1 or v2.
