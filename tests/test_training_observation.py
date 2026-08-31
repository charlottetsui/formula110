from __future__ import annotations

import math

import numpy as np

from racing.student.api import (
    CameraSensors,
    ContactSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotSensors,
)
from training.observation import OBSERVATION_DIM, encode_observation


def test_encode_observation_returns_fixed_shape_float32_vector() -> None:
    observation = encode_observation(RobotSensors())

    assert observation.shape == (OBSERVATION_DIM,)
    assert observation.dtype == np.float32


def test_encode_observation_scales_finite_values_into_unit_range() -> None:
    sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=1000.0),
        camera=CameraSensors(center_offset_m=1000.0, heading_error_degrees=90.0),
        imu=ImuSensors(yaw_rate_degrees_per_s=-1000.0),
    )

    observation = encode_observation(sensors)

    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_encode_observation_maps_infinite_wall_lidar_to_one() -> None:
    sensors = RobotSensors(wall_lidar=LidarSensors())  # default beams are all math.inf

    observation = encode_observation(sensors)

    # Wall LiDAR beams sit after speed(1) + heading(1) + center(1) + lookahead(3).
    wall_beam_slice = observation[6:13]
    assert np.allclose(wall_beam_slice, 1.0)


def test_encode_observation_flags_contact_and_reports_damage() -> None:
    sensors = RobotSensors(contact=ContactSensors(wall=0.5, damage=0.4))

    observation = encode_observation(sensors)

    assert observation[-1] == 0.4  # damage is the last feature
    assert observation[-3] == 1.0  # wall contact flag
    assert observation[-2] == 0.0  # robot contact flag


def test_encode_observation_pads_short_lookahead_tuples() -> None:
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(1.0,), lookahead_distances_m=(4.0,)))

    observation = encode_observation(sensors)

    assert not math.isnan(observation[3])
    assert observation.shape == (OBSERVATION_DIM,)
