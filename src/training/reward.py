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
WEIGHT_WALL_PROXIMITY = 0.5
WEIGHT_CONTACT = 0.2
WEIGHT_DAMAGE = 5.0
WEIGHT_REVERSE = 0.1

WALL_WARNING_DISTANCE_M = 3.0
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
    forward_progress_m = current.odometry.speed_mps * math.cos(math.radians(current.camera.heading_error_degrees))
    forward_progress_m *= current.dt_s
    damage_delta = max(0.0, current.contact.damage - previous.contact.damage)
    reverse_penalty = max(0.0, -current.odometry.speed_mps)
    in_contact = current.contact.wall > 0.0 or current.contact.robot > 0.0

    return (
        WEIGHT_PROGRESS * forward_progress_m
        - WEIGHT_CENTER_OFFSET * abs(current.camera.center_offset_m)
        - WEIGHT_WALL_PROXIMITY * _wall_proximity_penalty(current.wall_lidar)
        - WEIGHT_CONTACT * (1.0 if in_contact else 0.0)
        - WEIGHT_DAMAGE * damage_delta
        - WEIGHT_REVERSE * reverse_penalty
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
