from __future__ import annotations

from controllers.leaderboard_expert import Controller, create_controller
from racing import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors
from racing.physics import resolve_vehicle_actuator_command


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
    sensors = RobotSensors(wall_lidar=LidarSensors(distances_m=(0.5, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0)))

    assert create_controller()(sensors).steer > 0.0


def test_moving_wall_contact_does_not_latch_reverse_recovery() -> None:
    controller = Controller()
    controller(
        RobotSensors(
            contact=ContactSensors(wall=0.1),
            odometry=OdometrySensors(speed_mps=20.0),
        )
    )

    assert controller(RobotSensors()).throttle == 1.0


def test_blind_corner_releases_brakes_before_stopping() -> None:
    controller = Controller()
    assert (
        controller(
            RobotSensors(
                camera=CameraSensors(visible=False),
                odometry=OdometrySensors(speed_mps=20.0),
            )
        ).throttle
        < 0.0
    )
    assert (
        controller(
            RobotSensors(
                camera=CameraSensors(visible=False),
                odometry=OdometrySensors(speed_mps=4.0),
            )
        ).throttle
        == 0.0
    )
    assert (
        controller(RobotSensors(camera=CameraSensors(visible=False), odometry=OdometrySensors(speed_mps=4.0))).throttle
        > 0.0
    )


def test_close_front_wall_keeps_rolling_when_not_stuck() -> None:
    sensors = RobotSensors(
        wall_lidar=LidarSensors(distances_m=(5.0, 4.0, 2.0, 0.3, 2.0, 4.0, 5.0)),
        odometry=OdometrySensors(speed_mps=4.0),
    )

    assert create_controller()(sensors).throttle > 0.0


def test_persistent_contact_does_not_extend_recovery_forever() -> None:
    controller = Controller()
    sensors = RobotSensors(contact=ContactSensors(wall=0.1))
    for _ in range(28):
        assert controller(sensors).throttle == -1.0

    assert controller(RobotSensors()).throttle == 0.0
    assert controller(RobotSensors()).throttle == 1.0


def test_corner_brakes_release_in_physics_while_car_is_still_moving() -> None:
    controller = Controller()
    bend = CameraSensors(heading_error_degrees=35.0, lookahead_offsets_m=(0.0, 4.0, 9.0))
    braking = resolve_vehicle_actuator_command(
        command=controller(RobotSensors(camera=bend, odometry=OdometrySensors(speed_mps=30.0))),
        current_speed_kmh=108.0,
    )
    assert braking.brake_force > 0.0
    assert braking.next_pending_drive_direction == -1

    # The corner target is 23 m/s: release at 21 m/s, well before a stop.
    rolling = RobotSensors(camera=bend, odometry=OdometrySensors(speed_mps=21.0))
    release_command = controller(rolling)
    released = resolve_vehicle_actuator_command(
        command=release_command,
        current_speed_kmh=75.6,
        pending_drive_direction=braking.next_pending_drive_direction,
    )
    assert release_command.steer > 0.0
    assert released.brake_force == 0.0
    assert released.next_pending_drive_direction == 0

    accelerating = resolve_vehicle_actuator_command(
        command=controller(rolling),
        current_speed_kmh=75.6,
        pending_drive_direction=released.next_pending_drive_direction,
    )
    assert accelerating.brake_force == 0.0
    assert accelerating.engine_force > 0.0


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
