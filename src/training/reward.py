"""Proxy reward and episode-boundary detection for the SAC controller.

See ``docs/rl_design.md`` section 2.3-2.4. Official lap progress is private
to the simulator, so this reward is a proxy built entirely from the public
``RobotSensors`` fields visible to a controller. Every evaluation run should
also report the simulator's own public race stats (`HeadToHeadResult`) to
catch proxy/objective divergence -- this module only produces the training
signal, not a claim about real race performance.
"""

from __future__ import annotations

import math

from racing.student.api import CameraCompetitorReading, CameraSensors, LidarSensors, RobotSensors

WEIGHT_PROGRESS = 1.0
# Raised 0.05 -> 0.3 (2026-09-01, single-variable experiment) after the
# 2026-08-31 minimum experiment's trained controller spent 7.5-27% of each
# evaluation race off-track: at 0.05, a corner cut that stays roughly
# aligned with the track heading earns more cumulative `forward_progress`
# reward than it loses to `center_offset_m` penalty, so the policy had
# little incentive to stay on the drivable surface (TRACK_WIDTH/2, ~3.3m)
# rather than the wider marshal-reset radius (~4.7m). See
# docs/lab_notebook.md's 2026-09-01 entry for the before/after comparison
# this change produced.
#
# Tried lowering 0.3 -> 0.15 (2026-09-03) as a single-variable follow-up
# to removing MAX_REWARDED_SPEED_MPS (2026-09-02): that change increased
# top speed (+10%) without regressing safety, but lap time/count got
# slightly worse, not better -- hypothesis was that a uniform centerline
# penalty was suppressing a real racing line (wider entry/exit through
# corners, let alone drift-style sliding) even where it would be
# genuinely faster. Reverted: the result was a clear regression, not a
# trade-off -- avg max speed nearly halved (17.13 -> 8.80 m/s), avg laps
# dropped from 3.90 to 1.00, avg best lap time nearly tripled (28.46s ->
# 78.80s), and wins against `default_student_controller` dropped from
# 10/10 to 3/10, while safety stayed flat (still ~0 damage/off-track/
# wall-contact, 0/20 eliminated) -- so this wasn't unlocking a faster
# line, it was weakening the one signal keeping cornering deliberate.
# See docs/lab_notebook.md's 2026-09-03 entry and
# experiments/2026-09-03_center-offset-half-seed110/notes.md.
WEIGHT_CENTER_OFFSET = 0.3

# Added 2026-09-12 as a residual-mode-specific attempt at closing the pace gap to
# controllers.leaderboard_expert's raw speed, after the residual-scale sweep, the
# network-initialization seed sweep, and a flat progress-weight reweight all failed
# cleanly (see docs/rl_design.md section 6, causal test 37 and its follow-ups) --
# each treated speed as a single global dial. This instead targets cornering
# technique specifically: WEIGHT_CENTER_OFFSET today penalizes drifting off the
# centerline identically everywhere on the track, but leaderboard_expert.py itself
# explicitly does the opposite -- it computes how sharp the upcoming bend is (its own
# `bend_score`, from lookahead offsets) and only backs off for real corners, using the
# full track width on straights. The learned correction has never been given that
# distinction; every prior attempt to loosen this penalty *uniformly* (2026-09-03,
# WEIGHT_CENTER_OFFSET halved) was a clean regression, but that changed the penalty
# everywhere, including corners, which is exactly where it's load-bearing. This scales
# the penalty by curvature instead: full strength approaching a real bend, reduced (not
# zero) on straights. `curvature_aware_center_offset=False` (the default) leaves
# existing behavior identical -- this is a training.controller.TrainableController /
# scripts/train_sac.py opt-in flag, not a change to the constant itself.
CENTER_OFFSET_CURVATURE_REFERENCE_M = 3.0
MIN_CENTER_OFFSET_SCALE = 0.3

WEIGHT_CONTACT = 0.2
# Held at 5.0 since this reward's inception (2026-08-31) -- never itself the variable in any
# prior causal test. Added an opt-in `damage_weight` override to `step_reward` (2026-09-12) as
# a residual-mode-specific "accept more risk on purpose" lever, alongside
# `wall_proximity_speed_scale_mps` below: the base action already comes from a competent (if
# reckless) expert in that mode, so a lower damage cost may let the learned correction chase
# more of the expert's own speed instead of being dominated by caution tuned for a from-scratch
# policy with no such floor. Plain self-play is unaffected unless passed explicitly.
WEIGHT_DAMAGE = 5.0
WEIGHT_REVERSE = 0.1
# Added 2026-09-01 after a repeated-seed check on the scaled-training-budget
# experiment: one training seed converged to a "do nothing" policy (zero
# damage, zero off-track time, zero wall contact across every evaluation
# race, but ~27% of each race spent essentially stationary). Standing still
# was a free local optimum -- it never touches WEIGHT_DAMAGE, WEIGHT_CONTACT,
# or (near spawn) WEIGHT_CENTER_OFFSET, and nothing previously penalized
# near-zero forward speed (WEIGHT_REVERSE only fires on *negative* speed).
# This closes that loophole directly rather than reweighting existing
# terms, so it doesn't touch the safety-side incentives (WEIGHT_DAMAGE,
# WEIGHT_WALL_PROXIMITY, WEIGHT_CONTACT stay exactly as they were) while
# making idling itself costly. See docs/lab_notebook.md's 2026-09-01 entry
# and experiments/2026-09-01_scaled-training-budget-seed909/notes.md.
WEIGHT_IDLE = 0.2
IDLE_SPEED_MPS = 0.5
# Added 2026-09-01 as a follow-up to WEIGHT_IDLE, same day: testing the idle
# penalty on the seed that had been freezing fixed the freeze (10/10 race
# wins, up from 5/10), but the same seed then reached full elimination
# (damage == 1.0) in 10/10 evaluation races, driving at up to ~19 m/s
# (previously ~4-8.5 m/s). Likely mechanism: the simulator stops calling an
# eliminated car's controller, so dying early *ends* WEIGHT_IDLE's per-tick
# accrual for the rest of the round, while surviving-but-cautious keeps
# paying it every tick -- for a long enough round, "sprint and crash early"
# can look cheaper than "survive idly." WEIGHT_DAMAGE alone (scaled by the
# *delta* in one tick) doesn't clearly dominate that calculus. This adds a
# fixed, one-time cost specifically for the terminal transition, on top of
# the existing delta-based WEIGHT_DAMAGE penalty, so death itself is
# unambiguously bad regardless of how much idle-penalty it would otherwise
# have avoided. See docs/lab_notebook.md's 2026-09-01 entry and
# experiments/2026-09-01_idle-penalty-seed909/notes.md.
#
# Raised 10.0 -> 100.0 (2026-09-01) after two more experiments showed 10.0
# was not nearly enough: at the (now-capped) MAX_REWARDED_SPEED_MPS, one
# tick of progress reward is ~0.167, so as little as ~2 seconds of driving
# already accumulates more than a 10.0 penalty -- a policy dying after a
# short high-speed burst (observed: ~85-92m covered in ~2-3s before
# crashing) was still net-positive for the whole episode even with the
# penalty in place. 100.0 requires roughly 10 seconds of at-cap driving to
# break even, comfortably longer than the ~2-3s bursts observed so far,
# while staying well under a full episode's achievable reward (~1200 over
# a 120s round at the speed cap) so it shouldn't by itself reintroduce the
# "do nothing" freeze from before (WEIGHT_IDLE already guards against that
# separately). See docs/lab_notebook.md's 2026-09-01 entry and
# experiments/2026-09-01_speed-cap-seed909/notes.md for the arithmetic.
WEIGHT_TERMINAL_PENALTY = 100.0
# Added 2026-09-01 after the idle+terminal-penalty reward still produced a
# checkpoint (seed 909, either 60s or 120s training rounds) that reached
# 100% elimination while averaging 40 m/s -- confirmed by direct A/B test
# (deterministic vs. stochastic evaluation of the same weights, essentially
# identical) that this was substantively learned, not an eval-mode
# artifact. `forward_progress_m` previously scaled linearly with
# `speed_mps` with no ceiling, so a policy that discovers "more speed =
# more reward, monotonically" had no structural reason to stop -- a
# one-time WEIGHT_TERMINAL_PENALTY can be outweighed by a large enough
# high-speed burst beforehand. This caps the *speed used for the progress
# reward* (not the action itself, and not the other terms) so reward stops
# increasing past this point. Chosen from the observed speed range of the
# only checkpoint so far with zero eliminations (seed 110, avg max speed
# ~6.9 m/s, individual races 4.9-8.5 m/s) and the competent heuristic
# baseline (`default_student_controller`, ~5 m/s sustained): set above
# both so genuinely fast, safe driving is still fully rewarded, but well
# below the 15-40+ m/s regime seen in every crash-every-race checkpoint.
# See docs/lab_notebook.md's 2026-09-01 entry and
# experiments/2026-09-01_stochastic-vs-deterministic-diagnosis/notes.md.
#
# Tried raising 10.0 -> 20.0 (2026-09-01) to credit the
# `2026-09-01_more-training2-seed110` checkpoint's already-observed
# 15.4-16.1 m/s max speed (which exceeded the old cap even though average
# lap pace, ~6.8-7.7 m/s, stayed well under it). Reverted: trained from
# scratch at cap=20.0 with everything else unchanged, and got a clear
# regression, not a faster-and-still-safe controller -- max speed roughly
# doubled (34-38 m/s) but laps completed dropped (4.1 avg -> 0-2), lap
# times got *slower* despite the higher top speed (21-28s -> 36-96s),
# damage came back (0.000 -> 0.14-0.66), and marshal recoveries spiked to
# as high as 21/race (from ~0). Doubling the cap shifted the reward's
# relative balance too far toward raw speed at the expense of cornering
# control -- the cap was never blocking top speed, so raising it just
# reopened the same speed-vs-control trade-off seen in the seed-909 causal
# chain earlier today, from a different starting point. See
# docs/lab_notebook.md's 2026-09-01 entry and
# experiments/2026-09-01_speedcap20-seed110/notes.md.
#
# A separate experiment training further at cap=10.0 (races=40 -> 80, same
# day) showed the reward has no incentive to exceed the cap at all -- more
# training converged speed *down* toward ~13-15 m/s (from races=40's
# ~15-16 m/s), not up. Tried a small step, 10.0 -> 12.0 (2026-09-02, 20%
# increase, still below races=40's own observed peak ~16 m/s): reverted
# again -- essentially a tie with the cap=10.0 reference on safety/laps,
# but average best-lap time was *slower* (24.76s -> 28.20s), not faster.
# Three points on this axis (10.0, 12.0, 20.0) have now been tried from
# this seed/config; none beat the original 10.0. Treating
# MAX_REWARDED_SPEED_MPS tuning as exhausted for the speed goal rather
# than continuing to search this axis. See docs/lab_notebook.md's
# 2026-09-02 entry and
# experiments/2026-09-02_speedcap12-seed110/notes.md.
#
# REMOVED entirely (2026-09-02) after seven consecutive experiments
# (this cap at three values, more training via two different paths,
# steering smoothness, a best-trajectory bonus) all failed to beat the
# races=40 reference on speed -- the mechanism was understood by then:
# a flat cap is context-blind, so any speed above it is reward-neutral
# regardless of whether the car has 10m of open track ahead or is
# skimming a wall. Replaced with WALL_PROXIMITY_SPEED_SCALE_MPS below:
# forward_progress_m is now uncapped, and the *wall-proximity* penalty
# scales with current speed instead, so risk is priced by how dangerous
# the current situation actually is (fast + close to a wall) rather than
# by a single global speed number. See docs/lab_notebook.md's 2026-09-02
# entry.

# WALL_WARNING_DISTANCE_M raised 3.0 -> 6.0 and WEIGHT_WALL_PROXIMITY raised
# 0.5 -> 1.0 (2026-09-01) as a direct wall-avoidance strengthening pass,
# alongside the WEIGHT_TERMINAL_PENALTY raise above: every crashing
# checkpoint so far had driven at 15-40+ m/s, and at those speeds a 3.0m
# warning distance gives almost no reaction time (covered in well under a
# tenth of a second at 38 m/s) -- the proximity penalty was only ever
# ramping up once a collision was already essentially unavoidable. 6.0m
# gives real lead time even at a moderate cruising speed (~0.6s at
# 10 m/s), and doubling the weight makes the signal comparable in scale
# to WEIGHT_PROGRESS so avoiding a wall competes with, rather than being
# dominated by, going forward. See docs/lab_notebook.md's 2026-09-01
# entry.
#
# Tried 6.0 -> 12.0 (2026-09-08) after diagnosing a near-miss (0.5942
# damage, no elimination) in `2026-09-08_seed-sweep-v2-8000` (packaged in
# `race_faster.py`): the car accelerated through a tightening corner
# (8.8 -> 14.1 m/s) while wall clearance shrank from 3.90m to 1.30m in
# roughly 0.4s, taking a hard hit despite the wall-proximity mechanism
# nominally being active throughout. The 6.0m distance was explicitly
# calibrated for "~0.6s at 10 m/s" reaction time back when checkpoints
# cruised near that speed -- this checkpoint reaches ~27 m/s, where 6.0m
# gives only ~0.22s. Trained fresh on seed 8000 (the exact seed that
# produced the near-miss) with the doubled distance: the near-miss was
# essentially eliminated (avg damage 0.0298 -> 0.0000, off-track/
# wall-contact both near zero), **but at a steep pace cost that erases
# most of why this checkpoint was adopted** -- avg laps 7.30 -> 3.35, avg
# best lap time 14.96s -> 31.46s, avg max speed 26.65 -> 18.74 m/s. Same
# "safety fix overcorrects into a much slower policy" pattern seen
# repeatedly this session with proximity-based reward changes (the two
# `WEIGHT_ROBOT_PROXIMITY` attempts). Reverted to 6.0. See
# docs/lab_notebook.md's 2026-09-08 entry and
# experiments/2026-09-08_wallwarn12-seed8000/notes.md.
WEIGHT_WALL_PROXIMITY = 1.0
WALL_WARNING_DISTANCE_M = 6.0
WALL_WARNING_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-20.0, 0.0, 20.0)
# Added 2026-09-02 alongside removing MAX_REWARDED_SPEED_MPS (above): the
# same wall-proximity reading now costs more the faster the car is going,
# so "close to a wall at 5 m/s" (recoverable, common while lining up a
# corner) and "close to a wall at 35 m/s" (usually fatal) are no longer
# priced the same. At this speed, proximity penalty is 2x its 0 m/s value;
# chosen to match the old MAX_REWARDED_SPEED_MPS cruising target so the
# risk curve's shape is familiar even though the hard ceiling is gone.
# Intent: let the policy go as fast as a given moment's margin allows --
# including carrying speed through a corner with some slide, if that's
# actually faster in this physics model -- rather than a flat,
# situation-blind limit. See docs/lab_notebook.md's 2026-09-02 entry.
#
# Tried 10.0 -> 15.0 (2026-09-08) after observing what looked like a
# shared training-dynamics attractor (~25-26s lap time) that both the
# fastest and slowest seeds in a 5-seed sweep converged toward with more
# resumed training, regardless of starting point -- this constant was the
# most directly implicated lever since it was chosen "to match the old
# MAX_REWARDED_SPEED_MPS cruising target." Result: worse, not better --
# trained fresh on seed 1000 (races=40, matching the config that produced
# the reference 17.19s/31.02 m/s result at scale=10.0) and got 30.31s avg
# lap time, 20.39 m/s avg max speed -- both worse than the scale=10.0
# reference, and landing in the same ~25-30s range as the "converged"
# resumed checkpoints rather than a faster regime. Reverted to 10.0.
#
# This result was confounded, though, by a real bug found the same day:
# `scripts/train_sac.py` never passed `seed=args.seed` to `SACAgent(...)`,
# so *every* training run on this track so far -- across all "training
# seed" experiments, this one included -- started from bit-for-bit
# identical network initialization (`torch.manual_seed(0)`). The apparent
# shared attractor may be partly or entirely an artifact of that shared
# starting point rather than a property of this reward constant. Fixed
# the same day (see `scripts/train_sac.py`); this comparison should be
# re-run with genuine network-initialization diversity before drawing a
# firm conclusion about `WALL_PROXIMITY_SPEED_SCALE_MPS` specifically. See
# docs/lab_notebook.md's 2026-09-08 entry.
#
# Added an opt-in `wall_proximity_speed_scale_mps` override to `step_reward` (2026-09-12) as a
# residual-mode-specific pace lever, after four other levers (residual scale in both directions,
# a 5-seed sweep, a flat progress-weight reweight, curvature-aware center-offset) all failed to
# close the pace gap to controllers.leaderboard_expert's raw speed -- see docs/rl_design.md
# section 6, causal test 37 and its follow-ups. This constant is the closest thing left to a
# "speed ceiling" since `MAX_REWARDED_SPEED_MPS` was removed structurally (2026-09-02, above):
# raising it makes the wall-proximity penalty grow more slowly with speed, i.e. tolerates more
# speed before pricing it as risky. Untested at any value in residual mode specifically (the one
# prior test of this constant, 10.0 -> 15.0 on 2026-09-08, was on plain self-play and confounded
# by the shared-seed bug described above). Plain self-play is unaffected unless passed explicitly.
WALL_PROXIMITY_SPEED_SCALE_MPS = 10.0

# Added 2026-09-07 as the proposed complement to adding `sensors.lidar` to
# the observation (`training.observation`), then disabled the same day after
# a clear regression. Direct structural analog of
# `WEIGHT_WALL_PROXIMITY`/`WALL_WARNING_DISTANCE_M` above, but for the
# nearest competitor from `camera.competitors` instead of a wall beam --
# intended to teach proactive avoidance rather than only reacting via
# `WEIGHT_CONTACT` once contact already happened. Trained at weight=1.0 with
# no angle restriction on the nearest-competitor reading (unlike the wall-
# proximity beams, which are front-only): avg car-contact improved only
# slightly (0.952s -> 0.825s) while avg laps dropped 4.00 -> 2.90, avg best
# lap time rose 24.79s -> 35.31s (+42%), avg marshal/race rose 5.5x
# (0.10 -> 0.55), and max speed converged to a suspiciously uniform
# ~17.3-17.7 m/s across nearly every race (previously 14.7-23+ m/s,
# situation-dependent) -- a broad, uniform slowdown, not a targeted fix.
# Worse, the specific problem this was meant to solve got *worse*, not
# better: `seed=110 vs crash_fast` (both races, reproducibly) rose to
# 6.3-6.8s of car-contact, the highest seen in any configuration tested
# (previous worst: 4.12s), with 2 marshal recoveries and only 2 laps
# completed. Likely mechanism: with no angle restriction, the penalty fires
# for *any* nearby competitor regardless of whether it's actually in the
# way (beside, behind, or on a different part of a switchback), teaching
# generalized caution around any competitor rather than specifically
# avoiding collisions -- for a stationary blocker mid-track (`crash_fast`),
# that caution plausibly makes it harder to commit to a clean pass, not
# easier. Disabled (weight 0.0, mechanism kept) rather than deleted -- an
# angle-restricted or closing-speed-scaled variant (both noted as
# candidates when this was first written) remains untested and may not
# have the same failure mode. See docs/lab_notebook.md's 2026-09-07/09-08
# entry and
# experiments/2026-09-07_robot-proximity-obstacle-lidar-nstep3-seed110/notes.md.
#
# Re-enabled 2026-09-08 with the angle restriction proposed above: only a
# competitor within ROBOT_WARNING_ANGLE_DEGREES of straight ahead counts,
# mirroring how WALL_WARNING_BEAM_ANGLES_DEGREES is front-only rather than
# all-around. Intent: stop penalizing a competitor that's beside or behind
# (not actually in the way -- e.g. mid-pass, once already alongside a
# blocker) while still penalizing one closing from ahead. **Result: much
# worse, not better** -- 17 of 20 evaluation races completed exactly zero
# laps (vs. 4.00 avg before either proximity attempt), avg low-progress
# time exploded to 46.2s/race (38.5% of the round, one race hit 119.58s --
# essentially the entire round), and this was uniform across every seed,
# not one outlier. Mechanism: unlike avoiding a wall (which requires active
# steering, not stopping), avoiding an *ahead* competitor is trivially
# satisfied by never closing distance at all -- crawling at 6-10 m/s
# (comfortably above WEIGHT_IDLE's 0.5 m/s threshold, so that penalty
# doesn't fire either) permanently avoids the front-cone penalty with no
# need to ever commit to a pass. The unrestricted version (broad,
# situation-blind caution) and this restricted version (a clean escape
# hatch: just never approach) are different failure modes, not points on a
# spectrum from bad to good -- disabled again (weight 0.0, both mechanism
# and angle restriction kept in code) rather than tuning the angle/distance
# further without a different underlying idea. See
# docs/lab_notebook.md's 2026-09-08 entry and
# experiments/2026-09-08_robot-proximity-angle-restricted-seed110/notes.md.
WEIGHT_ROBOT_PROXIMITY = 0.0
ROBOT_WARNING_DISTANCE_M = 8.0
ROBOT_WARNING_ANGLE_DEGREES = 45.0

# Added 2026-09-10 as a new combined-approach direction with the separate
# imitation-learning track (`controllers.leaderboard_expert`, a hand-written
# rule-based controller developed independently on `lucy-il`): reward the
# policy for choosing an action close to what that expert would have chosen,
# but *only* in situations already judged hazardous by the wall/robot
# proximity checks above -- ordinary cornering/racing behavior is left
# untouched. Distinct from every other training-time combination attempted
# on this track (docs/rl_design.md section 6, causal test 35 and its
# follow-up): those changed self-play's *opponent* to the expert itself and
# caused severe, fast training collapse at every dose and resume strategy
# tried, because a fixed, unfamiliar, already-competent opponent shifts the
# entire training distribution the policy learns from. This mechanism
# changes nothing about self-play or its opponent -- the expert is consulted
# only as a labeling function for what a known-competent driver would do at
# the exact tick the policy's own action is being scored, so it cannot
# destabilize training the same way. Weight chosen to be comparable in scale
# to WEIGHT_WALL_PROXIMITY's typical single-hazard contribution (~0.3-1.0)
# without dominating WEIGHT_PROGRESS -- not yet tuned by a dedicated
# experiment.
WEIGHT_EXPERT_MATCH = 0.5

# Tried 2026-09-02 after three attempts to raise speed via
# MAX_REWARDED_SPEED_MPS (10.0 -> 12.0, -> 20.0) all failed to beat the
# 10.0 reference on lap time -- tried a different lever: penalize
# oscillating/hesitant steering instead of pushing the speed incentive
# further. `step_reward` only sees sensor transitions (no direct access to
# the steer action -- see `training.controller.TrainableController`, which
# computes reward before choosing the next action), so this used
# tick-to-tick change in `imu.yaw_rate_degrees_per_s` as a proxy for
# jerky steering. Reverted to 0.0 (mechanism kept, disabled by weight):
# trained at weight=0.1 and got the clearest regression of any experiment
# so far -- laps completed halved (4.10 avg -> exactly 2.00, every race),
# best lap time nearly doubled (24.76s -> 45.48s), max speed dropped 31%
# (15.53 -> 10.75 m/s), with safety unaffected (still perfect). Raw
# yaw-rate *change* can't distinguish wasteful oscillation from a
# legitimate, necessary steering input for cornering -- both involve the
# yaw rate changing quickly, so the penalty suppressed real cornering, not
# just hesitation. See docs/lab_notebook.md's 2026-09-02 entry and
# experiments/2026-09-02_steering-smoothness-seed110/notes.md.
WEIGHT_STEERING_SMOOTHNESS = 0.0
YAW_RATE_CHANGE_SCALE_DEGREES_PER_S = 200.0

# Added 2026-09-07 as a more targeted retry of the same "reduce hesitation"
# goal `WEIGHT_STEERING_SMOOTHNESS` was meant for. That mechanism penalized
# raw yaw-rate *magnitude* of change, which can't tell a deliberate cornering
# turn (yaw rate changing quickly, but consistently in one direction) from
# actual hesitation (steering wobbling back and forth) -- both look the same
# under a magnitude-only proxy, which is why it suppressed real cornering
# instead of just hesitation. This penalizes yaw-rate *sign reversals*
# instead: a held turn keeps a consistent sign even as its magnitude
# changes, so only an actual direction flip (steering left, then right, in
# consecutive ticks) counts. The threshold excludes near-zero yaw rate
# (car going essentially straight) from counting as a "reversal" -- noise
# around zero shouldn't be penalized the same as a real correction.
# n-step returns (n=3, kept fixed for the observation-change test below)
# already improved lap time via off-track/wall-avoidance
# (docs/lab_notebook.md's 2026-09-07 entry) but left low-progress/car-contact
# time (this project's headless proxies for hesitation) flat or slightly
# worse. Implemented this term expecting steering wobble to be the cause --
# before testing it, a direct per-tick diagnostic (logging contact.robot
# across an evaluation race) found the real cause instead: the policy has
# zero observation of other cars at all (`camera.competitors`/`sensors.lidar`
# were both deferred by the original design, see `training.observation`) and
# was colliding blind with a stationary opponent once per lap, at the same
# track position each time -- not steering indecision. Disabled here
# (weight 0.0, mechanism kept and tested) so it doesn't confound the
# opponent-observation test, which is now the better-supported hypothesis;
# revisit as its own single-variable test once that's evaluated. See
# docs/lab_notebook.md's 2026-09-07 entry.
WEIGHT_STEERING_REVERSAL = 0.0
YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S = 5.0

# Real, exact elimination (`damage == 1.0`) is never observed in-band: the
# simulator stops calling a controller once its car is marked eliminated,
# and that flag is set from the damage applied *after* the tick whose
# sensors the controller last saw (see docs/lab_notebook.md, 2026-08-31
# entry, "what we observed"). A near-elimination threshold approximates the
# terminal transition instead of requiring the unobservable exact value.
NEAR_ELIMINATION_DAMAGE = 0.9


def step_reward(
    previous: RobotSensors,
    current: RobotSensors,
    *,
    previous_action: tuple[float, float] | None = None,
    expert_action: tuple[float, float] | None = None,
    progress_weight: float = WEIGHT_PROGRESS,
    curvature_aware_center_offset: bool = False,
    wall_proximity_speed_scale_mps: float = WALL_PROXIMITY_SPEED_SCALE_MPS,
    damage_weight: float = WEIGHT_DAMAGE,
) -> float:
    """Return the proxy reward for the transition from `previous` to `current`.

    `previous_action` (the actual `(throttle, steer)` chosen at `previous`) and
    `expert_action` (what `controllers.leaderboard_expert.Controller` would
    have chosen there) are both optional and default to `None`, in which case
    the expert-match term below contributes nothing -- callers that don't run
    a shadow expert controller (e.g. every test in this file, and
    `TrainableController` with `expert_match=False`) are unaffected.

    `progress_weight` overrides `WEIGHT_PROGRESS` for this call and defaults to
    it, so every existing caller is unaffected. Added 2026-09-12 to test
    whether residual-mode training (`TrainableController`'s `residual_base`)
    -- where the *base* action already comes from a competent, if reckless,
    expert -- benefits from weighting speed more heavily relative to the
    caution terms below, which were tuned entirely in the context of a
    from-scratch policy that has to supply 100% of its own collision
    avoidance. Plain self-play is unaffected unless this is passed explicitly.

    `curvature_aware_center_offset` (default `False`, so existing behavior is
    unchanged) scales the `WEIGHT_CENTER_OFFSET` penalty by how sharp the
    upcoming bend is instead of applying it uniformly -- see
    `CENTER_OFFSET_CURVATURE_REFERENCE_M`'s docstring above for the rationale.

    `wall_proximity_speed_scale_mps` and `damage_weight` override
    `WALL_PROXIMITY_SPEED_SCALE_MPS`/`WEIGHT_DAMAGE` respectively and both default
    to the module constants, so every existing caller is unaffected. Added
    2026-09-12, alongside `progress_weight`, as further residual-mode-specific
    "accept more risk on purpose" levers for closing the pace gap to
    `controllers.leaderboard_expert`'s raw speed -- see those constants'
    docstrings above for the rationale.
    """
    forward_progress_m = current.odometry.speed_mps * math.cos(math.radians(current.camera.heading_error_degrees))
    forward_progress_m *= current.dt_s
    damage_delta = max(0.0, current.contact.damage - previous.contact.damage)
    reverse_penalty = max(0.0, -current.odometry.speed_mps)
    in_contact = current.contact.wall > 0.0 or current.contact.robot > 0.0
    is_idle = abs(current.odometry.speed_mps) < IDLE_SPEED_MPS
    yaw_rate_change = abs(current.imu.yaw_rate_degrees_per_s - previous.imu.yaw_rate_degrees_per_s)
    steering_smoothness_penalty = min(1.0, yaw_rate_change / YAW_RATE_CHANGE_SCALE_DEGREES_PER_S)
    is_steering_reversal = _is_steering_reversal(
        previous.imu.yaw_rate_degrees_per_s, current.imu.yaw_rate_degrees_per_s
    )
    speed_risk_multiplier = 1.0 + abs(current.odometry.speed_mps) / wall_proximity_speed_scale_mps
    wall_proximity_penalty = _wall_proximity_penalty(current.wall_lidar) * speed_risk_multiplier
    robot_proximity_penalty = _robot_proximity_penalty(current.camera.competitors) * speed_risk_multiplier
    expert_match_cost = _expert_match_penalty(previous, previous_action, expert_action)
    center_offset_scale = _center_offset_curvature_scale(current.camera) if curvature_aware_center_offset else 1.0

    return (
        progress_weight * forward_progress_m
        - WEIGHT_CENTER_OFFSET * center_offset_scale * abs(current.camera.center_offset_m)
        - WEIGHT_WALL_PROXIMITY * wall_proximity_penalty
        - WEIGHT_ROBOT_PROXIMITY * robot_proximity_penalty
        - WEIGHT_CONTACT * (1.0 if in_contact else 0.0)
        - damage_weight * damage_delta
        - WEIGHT_REVERSE * reverse_penalty
        - WEIGHT_IDLE * (1.0 if is_idle else 0.0)
        - WEIGHT_TERMINAL_PENALTY * (1.0 if is_terminal(current) else 0.0)
        - WEIGHT_STEERING_SMOOTHNESS * steering_smoothness_penalty
        - WEIGHT_STEERING_REVERSAL * (1.0 if is_steering_reversal else 0.0)
        - WEIGHT_EXPERT_MATCH * expert_match_cost
    )


def is_terminal(sensors: RobotSensors) -> bool:
    """Return whether this snapshot should close out the current episode."""
    return sensors.contact.damage >= NEAR_ELIMINATION_DAMAGE


def is_new_episode(sensors: RobotSensors) -> bool:
    """Return whether `sensors` marks the first tick of a fresh controller run.

    Reserved for training loops that reuse one controller instance across
    more than one car life (a persistent single-car loop, for example).
    Self-play via ``run_headless_head_to_head`` gives every car in every
    race a fresh controller instance already (via ``copy_for_car``), so this
    normally never fires there -- see docs/rl_design.md section 3.
    """
    return sensors.tick == 0


def _is_steering_reversal(previous_yaw_rate_degrees_per_s: float, current_yaw_rate_degrees_per_s: float) -> bool:
    """Return whether steering direction flipped, ignoring near-zero (essentially straight) yaw rate."""
    if abs(previous_yaw_rate_degrees_per_s) < YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S:
        return False
    if abs(current_yaw_rate_degrees_per_s) < YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S:
        return False
    return math.copysign(1.0, previous_yaw_rate_degrees_per_s) != math.copysign(1.0, current_yaw_rate_degrees_per_s)


def _bend_score(camera: CameraSensors) -> float:
    """Return how sharply the track ahead curves, from lookahead offsets.

    Mirrors `controllers.leaderboard_expert`'s own `bend_score` exactly (near/
    middle/far lookahead offsets, same weighting) so "curvy" means the same
    thing here as it does to the expert whose command this is meant to
    complement, not a new, uncalibrated notion of curvature.
    """
    offsets = camera.lookahead_offsets_m
    near = offsets[0] if offsets else camera.center_offset_m
    middle = offsets[len(offsets) // 2] if offsets else near
    far = offsets[-1] if offsets else middle
    return abs(far - near) + 0.45 * abs(middle - near)


def _center_offset_curvature_scale(camera: CameraSensors) -> float:
    """Return a `[MIN_CENTER_OFFSET_SCALE, 1.0]` multiplier for the center-offset penalty.

    Full strength (`1.0`) at or above `CENTER_OFFSET_CURVATURE_REFERENCE_M` of
    bend (a real corner); ramps down to `MIN_CENTER_OFFSET_SCALE` (never all
    the way to zero -- some centering signal stays even on a straight) as the
    track ahead flattens out.
    """
    ratio = min(1.0, _bend_score(camera) / CENTER_OFFSET_CURVATURE_REFERENCE_M)
    return MIN_CENTER_OFFSET_SCALE + (1.0 - MIN_CENTER_OFFSET_SCALE) * ratio


def _wall_proximity_penalty(wall_lidar: LidarSensors) -> float:
    warnings = tuple(
        _proximity_ratio(
            wall_lidar.distance_at_angle_degrees(angle_degrees), warning_distance_m=WALL_WARNING_DISTANCE_M
        )
        for angle_degrees in WALL_WARNING_BEAM_ANGLES_DEGREES
    )
    return sum(warnings) / len(warnings)


def _robot_proximity_penalty(competitors: tuple[CameraCompetitorReading, ...]) -> float:
    """Return a proximity penalty for the nearest roughly-ahead competitor, or 0.0 if none qualify."""
    ahead = [competitor for competitor in competitors if abs(competitor.angle_degrees) <= ROBOT_WARNING_ANGLE_DEGREES]
    if not ahead:
        return 0.0
    nearest_distance_m = min(competitor.distance_m for competitor in ahead)
    return _proximity_ratio(nearest_distance_m, warning_distance_m=ROBOT_WARNING_DISTANCE_M)


def _proximity_ratio(distance_m: float, *, warning_distance_m: float) -> float:
    if not math.isfinite(distance_m) or distance_m >= warning_distance_m:
        return 0.0
    return (warning_distance_m - max(0.0, distance_m)) / warning_distance_m


def _in_hazard(sensors: RobotSensors) -> bool:
    """Return whether `sensors` describes a wall- or competitor-proximity hazard.

    Reuses the exact proximity checks `_wall_proximity_penalty`/
    `_robot_proximity_penalty` are built from (regardless of whether those
    mechanisms' own weights are currently enabled), so "hazard" here means
    precisely the situations this file already has a notion of being risky.
    """
    wall_hazard = _wall_proximity_penalty(sensors.wall_lidar) > 0.0
    robot_hazard = _robot_proximity_penalty(sensors.camera.competitors) > 0.0
    return wall_hazard or robot_hazard


def _expert_match_penalty(
    previous: RobotSensors,
    previous_action: tuple[float, float] | None,
    expert_action: tuple[float, float] | None,
) -> float:
    """Return how far `previous_action` was from `expert_action`, but only in a hazard state.

    Zero whenever either action is unavailable (the mechanism is disabled, or
    this is the controller's first tick, before any action has been taken) or
    `previous` wasn't judged hazardous -- ordinary racing behavior is never
    nudged toward the expert's, only behavior at the exact moments already
    flagged as risky by this file's own proximity checks.
    """
    if previous_action is None or expert_action is None or not _in_hazard(previous):
        return 0.0
    return (abs(previous_action[0] - expert_action[0]) + abs(previous_action[1] - expert_action[1])) / 2.0
