from __future__ import annotations

from controllers.geometry_expert import Controller, create_controller
from racing import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors


def test_expert_steers_toward_right_hand_geometry() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(
            center_offset_m=0.4,
            heading_error_degrees=12.0,
            lookahead_offsets_m=(0.5, 0.8, 1.2),
        )
    )

    assert create_controller()(sensors).steer > 0.0


def test_expert_slows_for_a_sharp_bend() -> None:
    straight = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.0, 0.0, 0.0)),
        odometry=OdometrySensors(speed_mps=2.0),
    )
    bend = RobotSensors(
        camera=CameraSensors(lookahead_offsets_m=(0.0, 2.0, 5.0)),
        odometry=OdometrySensors(speed_mps=2.0),
    )

    straight_command = create_controller()(straight)
    bend_command = create_controller()(bend)

    assert bend_command.throttle < straight_command.throttle


def test_expert_keeps_reversing_after_wall_contact_ends() -> None:
    controller = Controller()
    contact = RobotSensors(
        contact=ContactSensors(wall=0.1),
        wall_lidar=LidarSensors(distances_m=(4.0, 4.0, 2.0, 1.0, 1.0, 1.0, 0.5)),
    )

    first_command = controller(contact)
    next_command = controller(RobotSensors())

    assert first_command.throttle < 0.0
    assert next_command.throttle < 0.0


def test_expert_commands_stay_in_documented_ranges() -> None:
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
