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

from racing.student.api import LidarSensors, RobotSensors

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
WEIGHT_CENTER_OFFSET = 0.3
WEIGHT_CONTACT = 0.2
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
MAX_REWARDED_SPEED_MPS = 10.0

# WALL_WARNING_DISTANCE_M raised 3.0 -> 6.0 and WEIGHT_WALL_PROXIMITY raised
# 0.5 -> 1.0 (2026-09-01) as a direct wall-avoidance strengthening pass,
# alongside the WEIGHT_TERMINAL_PENALTY raise above: every crashing
# checkpoint so far has driven at 15-40+ m/s, and at those speeds a 3.0m
# warning distance gives almost no reaction time (covered in well under a
# tenth of a second at 38 m/s) -- the proximity penalty was only ever
# ramping up once a collision was already essentially unavoidable. 6.0m
# gives real lead time even at the MAX_REWARDED_SPEED_MPS cruising speed
# (~0.6s at 10 m/s), and doubling the weight makes the signal comparable
# in scale to WEIGHT_PROGRESS so avoiding a wall competes with, rather
# than being dominated by, going forward. See docs/lab_notebook.md's
# 2026-09-01 entry.
WEIGHT_WALL_PROXIMITY = 1.0
WALL_WARNING_DISTANCE_M = 6.0
WALL_WARNING_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-20.0, 0.0, 20.0)

# Real, exact elimination (`damage == 1.0`) is never observed in-band: the
# simulator stops calling a controller once its car is marked eliminated,
# and that flag is set from the damage applied *after* the tick whose
# sensors the controller last saw (see docs/lab_notebook.md, 2026-08-31
# entry, "what we observed"). A near-elimination threshold approximates the
# terminal transition instead of requiring the unobservable exact value.
NEAR_ELIMINATION_DAMAGE = 0.9


def step_reward(previous: RobotSensors, current: RobotSensors) -> float:
    """Return the proxy reward for the transition from `previous` to `current`."""
    rewarded_speed_mps = math.copysign(
        min(abs(current.odometry.speed_mps), MAX_REWARDED_SPEED_MPS), current.odometry.speed_mps
    )
    forward_progress_m = rewarded_speed_mps * math.cos(math.radians(current.camera.heading_error_degrees))
    forward_progress_m *= current.dt_s
    damage_delta = max(0.0, current.contact.damage - previous.contact.damage)
    reverse_penalty = max(0.0, -current.odometry.speed_mps)
    in_contact = current.contact.wall > 0.0 or current.contact.robot > 0.0
    is_idle = abs(current.odometry.speed_mps) < IDLE_SPEED_MPS

    return (
        WEIGHT_PROGRESS * forward_progress_m
        - WEIGHT_CENTER_OFFSET * abs(current.camera.center_offset_m)
        - WEIGHT_WALL_PROXIMITY * _wall_proximity_penalty(current.wall_lidar)
        - WEIGHT_CONTACT * (1.0 if in_contact else 0.0)
        - WEIGHT_DAMAGE * damage_delta
        - WEIGHT_REVERSE * reverse_penalty
        - WEIGHT_IDLE * (1.0 if is_idle else 0.0)
        - WEIGHT_TERMINAL_PENALTY * (1.0 if is_terminal(current) else 0.0)
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


def _wall_proximity_penalty(wall_lidar: LidarSensors) -> float:
    warnings = tuple(
        _proximity_ratio(wall_lidar.distance_at_angle_degrees(angle_degrees))
        for angle_degrees in WALL_WARNING_BEAM_ANGLES_DEGREES
    )
    return sum(warnings) / len(warnings)


def _proximity_ratio(distance_m: float) -> float:
    if not math.isfinite(distance_m) or distance_m >= WALL_WARNING_DISTANCE_M:
        return 0.0
    return (WALL_WARNING_DISTANCE_M - max(0.0, distance_m)) / WALL_WARNING_DISTANCE_M
