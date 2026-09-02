from __future__ import annotations

from controllers.leaderboard_expert import Controller, create_controller
from racing import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors


def test_leaderboard_expert_uses_full_throttle_on_clear_straight() -> None:
    assert create_controller()(RobotSensors()).throttle == 1.0


def test_leaderboard_expert_brakes_hard_when_fast_before_severe_bend() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(heading_error_degrees=35.0, lookahead_offsets_m=(0.0, 4.0, 9.0)),
        odometry=OdometrySensors(speed_mps=30.0),
    )

    assert create_controller()(sensors).throttle == -1.0


def test_leaderboard_expert_retains_bounded_recovery() -> None:
    controller = Controller()

    assert controller(RobotSensors(contact=ContactSensors(wall=0.1))).throttle == -1.0
    assert controller(RobotSensors()).throttle == -1.0


def test_leaderboard_expert_steers_away_from_close_left_wall() -> None:
    sensors = RobotSensors(
        wall_lidar=LidarSensors(distances_m=(0.5, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0))
    )

    assert create_controller()(sensors).steer > 0.0


def test_leaderboard_expert_commands_stay_in_documented_ranges() -> None:
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
