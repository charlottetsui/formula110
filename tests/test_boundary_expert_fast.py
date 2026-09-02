from __future__ import annotations

from controllers.boundary_expert_fast import Controller, create_controller
from racing import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors


def test_boundary_expert_ignores_small_center_offset_on_straight() -> None:
    centered = RobotSensors(camera=CameraSensors(center_offset_m=0.0))
    displaced = RobotSensors(camera=CameraSensors(center_offset_m=1.0))

    centered_command = create_controller()(centered)
    displaced_command = create_controller()(displaced)

    assert displaced_command == centered_command
    assert displaced_command.throttle == 1.0


def test_boundary_expert_steers_away_from_close_left_wall() -> None:
    sensors = RobotSensors(
        wall_lidar=LidarSensors(distances_m=(0.5, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0))
    )

    assert create_controller()(sensors).steer > 0.0


def test_boundary_expert_brakes_for_short_front_boundary() -> None:
    sensors = RobotSensors(
        wall_lidar=LidarSensors(distances_m=(5.0, 4.0, 2.0, 1.5, 2.0, 4.0, 5.0)),
        odometry=OdometrySensors(speed_mps=9.0),
    )

    assert create_controller()(sensors).throttle < 0.0


def test_boundary_expert_retains_recovery_after_contact() -> None:
    controller = Controller()
    contact = RobotSensors(contact=ContactSensors(wall=0.1))

    assert controller(contact).throttle < 0.0
    assert controller(RobotSensors()).throttle < 0.0


def test_boundary_expert_commands_stay_in_documented_ranges() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(
            center_offset_m=100.0,
            heading_error_degrees=179.0,
            lookahead_offsets_m=(100.0, 100.0, 100.0),
        ),
        odometry=OdometrySensors(speed_mps=-20.0),
    )

    command = create_controller()(sensors)

    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0
