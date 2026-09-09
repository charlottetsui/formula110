"""Sensor snapshot -> SAC observation vector encoding.

Implements the observation vector from ``docs/rl_design.md`` section 2.1:
a fixed-size, roughly ``[-1, 1]``-scaled feature vector built only from
public ``RobotSensors`` fields.
"""

from __future__ import annotations

import math

import numpy as np

from racing.student.api import RobotSensors

MAX_SPEED_MPS = 20.0
WALL_LIDAR_CAP_M = 20.0
CENTER_OFFSET_CAP_M = 5.0
LOOKAHEAD_OFFSET_CAP_M = 5.0
YAW_RATE_CAP_DEGREES_PER_S = 180.0

WALL_LIDAR_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
LOOKAHEAD_COUNT = 3

OBSERVATION_DIM = (
    1  # signed speed
    + 1  # heading error
    + 1  # center offset
    + LOOKAHEAD_COUNT  # lookahead offsets
    + len(WALL_LIDAR_BEAM_ANGLES_DEGREES)  # wall lidar beams
    + 1  # yaw rate
    + 1  # wall contact flag
    + 1  # robot contact flag
    + 1  # damage
)


def encode_observation(sensors: RobotSensors) -> np.ndarray:
    """Return the fixed-size, roughly ``[-1, 1]``-scaled observation vector."""
    wall_beams = tuple(
        _scale_distance(sensors.wall_lidar.distance_at_angle_degrees(angle_degrees), cap=WALL_LIDAR_CAP_M)
        for angle_degrees in WALL_LIDAR_BEAM_ANGLES_DEGREES
    )
    lookahead = _padded_lookahead(sensors.camera.lookahead_offsets_m)
    features = (
        _clip_ratio(sensors.odometry.speed_mps, MAX_SPEED_MPS),
        _clip_ratio(sensors.camera.heading_error_degrees, 180.0),
        _clip_ratio(sensors.camera.center_offset_m, CENTER_OFFSET_CAP_M),
        *(_clip_ratio(offset_m, LOOKAHEAD_OFFSET_CAP_M) for offset_m in lookahead),
        *wall_beams,
        _clip_ratio(sensors.imu.yaw_rate_degrees_per_s, YAW_RATE_CAP_DEGREES_PER_S),
        1.0 if sensors.contact.wall > 0.0 else 0.0,
        1.0 if sensors.contact.robot > 0.0 else 0.0,
        sensors.contact.damage,
    )
    observation = np.asarray(features, dtype=np.float32)
    if observation.shape != (OBSERVATION_DIM,):
        raise ValueError(f"encoded observation has shape {observation.shape}, expected ({OBSERVATION_DIM},)")
    return observation


def _padded_lookahead(offsets_m: tuple[float, ...]) -> tuple[float, ...]:
    if len(offsets_m) >= LOOKAHEAD_COUNT:
        return offsets_m[:LOOKAHEAD_COUNT]
    return offsets_m + (0.0,) * (LOOKAHEAD_COUNT - len(offsets_m))


def _clip_ratio(value: float, cap: float) -> float:
    if not math.isfinite(value):
        value = math.copysign(cap, value) if value != 0.0 else 0.0
    return max(-1.0, min(1.0, value / cap))


def _scale_distance(distance_m: float, *, cap: float) -> float:
    """Map a (possibly infinite) LiDAR distance to ``[0, 1]``, capped at `cap`."""
    if not math.isfinite(distance_m):
        return 1.0
    return max(0.0, min(1.0, distance_m / cap))
