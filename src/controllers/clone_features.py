"""Versioned public-sensor representation shared by cloning and inference."""

from __future__ import annotations

from collections import deque
from math import isfinite, log1p

import numpy as np
from numpy.typing import NDArray

from racing import RobotCommand, RobotSensors

FEATURE_VERSION = 1
HISTORY_LAGS = (0, 1, 2, 4, 8, 16, 24, 32)
BEAM_ANGLES = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
FRAME_DIM = 42
OBSERVATION_DIM = FRAME_DIM * len(HISTORY_LAGS)


def encode_frame(sensors: RobotSensors, previous: RobotCommand) -> NDArray[np.float32]:
    """Keep high-speed information, visibility and no-hit indicators explicit."""
    offsets = sensors.camera.lookahead_offsets_m
    near = offsets[0] if offsets else sensors.camera.center_offset_m
    middle = offsets[len(offsets) // 2] if offsets else near
    far = offsets[-1] if offsets else middle
    features = [
        sensors.odometry.speed_mps / 50.0,
        sensors.camera.heading_error_degrees / 180.0,
        sensors.camera.center_offset_m / 20.0,
        near / 20.0,
        middle / 20.0,
        far / 20.0,
        sensors.imu.yaw_rate_degrees_per_s / 360.0,
        float(sensors.camera.visible),
        sensors.dt_s * 60.0,
        float(sensors.contact.wall > 0),
        float(sensors.contact.robot > 0),
        sensors.contact.damage,
    ]
    for lidar in (sensors.wall_lidar, sensors.lidar):
        distances = [lidar.distance_at_angle_degrees(angle) for angle in BEAM_ANGLES]
        features.extend(log1p(max(0.0, min(d, 80.0))) / log1p(80.0) if isfinite(d) else 1.0 for d in distances)
        features.extend(float(isfinite(d) and d < lidar.max_distance_m) for d in distances)
    features.extend((previous.throttle, previous.steer))
    frame = np.asarray(features, dtype=np.float32)
    if frame.shape != (FRAME_DIM,) or not np.isfinite(frame).all():
        raise ValueError("Invalid cloning observation")
    return frame


class ObservationHistory:
    """Eight causal samples spanning 32 ticks; fresh instances for every car."""

    def __init__(self) -> None:
        self.frames: deque[NDArray[np.float32]] = deque(maxlen=max(HISTORY_LAGS) + 1)

    def append(self, sensors: RobotSensors, previous: RobotCommand) -> NDArray[np.float32]:
        if sensors.tick == 0:
            self.frames.clear()
        self.frames.append(encode_frame(sensors, previous))
        return np.concatenate([self.frames[max(0, len(self.frames) - 1 - lag)] for lag in HISTORY_LAGS])


def release_brake(requested: RobotCommand, previous: RobotCommand) -> RobotCommand:
    """Exactly neutral for one tick when switching from braking to forward drive."""
    if previous.throttle < 0.0 and requested.throttle > 0.0:
        return RobotCommand(throttle=0.0, steer=requested.steer)
    return requested
