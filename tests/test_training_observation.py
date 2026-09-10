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


def test_encode_observation_maps_infinite_obstacle_lidar_to_one() -> None:
    sensors = RobotSensors(lidar=LidarSensors())  # default beams are all math.inf

    observation = encode_observation(sensors)

    # Obstacle lidar beams sit after speed(1)+heading(1)+center(1)+lookahead(3)+wall_lidar(7).
    obstacle_beam_slice = observation[13:20]
    assert np.allclose(obstacle_beam_slice, 1.0)


def test_encode_observation_detects_a_nearby_car_even_with_no_wall_nearby() -> None:
    # `sensors.lidar` (unlike `wall_lidar`) detects other robots -- this is the
    # feature added 2026-09-07 after diagnosing blind collisions with a stationary
    # opponent car (docs/lab_notebook.md's 2026-09-07 entry).
    open_wall_lidar = LidarSensors()  # no wall nearby
    nearby_car = LidarSensors(distances_m=tuple(2.0 for _ in range(7)))
    clear = RobotSensors(wall_lidar=open_wall_lidar, lidar=LidarSensors())
    blocked_by_car = RobotSensors(wall_lidar=open_wall_lidar, lidar=nearby_car)

    clear_observation = encode_observation(clear)
    blocked_observation = encode_observation(blocked_by_car)

    # Wall-lidar-derived features (indices 6:13) are identical -- only the new
    # obstacle-lidar slice (13:20) should differ.
    assert np.allclose(clear_observation[6:13], blocked_observation[6:13])
    assert not np.allclose(clear_observation[13:20], blocked_observation[13:20])


def test_encode_observation_pads_short_lookahead_tuples() -> None:
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(1.0,), lookahead_distances_m=(4.0,)))

    observation = encode_observation(sensors)

    assert not math.isnan(observation[3])
    assert observation.shape == (OBSERVATION_DIM,)
