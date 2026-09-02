from __future__ import annotations

from controllers.geometry_expert import create_controller as create_reliable_controller
from controllers.geometry_expert_fast import Controller, create_controller
from racing import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors


def test_fast_expert_accelerates_harder_on_clear_straight() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.0, 0.0, 0.0)),
        odometry=OdometrySensors(speed_mps=4.0),
    )

    fast = create_controller()(sensors)
    reliable = create_reliable_controller()(sensors)

    assert fast.throttle > reliable.throttle


def test_fast_expert_uses_full_throttle_from_rest_on_clear_straight() -> None:
    sensors = RobotSensors(camera=CameraSensors(lookahead_offsets_m=(0.0, 0.0, 0.0)))

    assert create_controller()(sensors).throttle == 1.0


def test_fast_expert_brakes_for_large_upcoming_bend() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.0, 3.0, 7.0)),
        odometry=OdometrySensors(speed_mps=5.0),
    )

    assert create_controller()(sensors).throttle < 0.0


def test_fast_expert_retains_stateful_wall_recovery() -> None:
    controller = Controller()
    contact = RobotSensors(
        contact=ContactSensors(wall=0.1),
        wall_lidar=LidarSensors(distances_m=(4.0, 4.0, 2.0, 1.0, 1.0, 1.0, 0.5)),
    )

    assert controller(contact).throttle < 0.0
    assert controller(RobotSensors()).throttle < 0.0


def test_fast_expert_commands_stay_in_documented_ranges() -> None:
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
