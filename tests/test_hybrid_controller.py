from __future__ import annotations

from controllers import leaderboard_expert, race_faster
from controllers.hybrid_controller import Controller
from racing import (
    CameraCompetitorReading,
    CameraSensors,
    ContactSensors,
    LidarSensors,
    OdometrySensors,
    RobotSensors,
)

# Matches WALL_LIDAR_BEAM_ANGLES_DEGREES order: (-90, -45, -20, 0, 20, 45, 90).
_CLOSE_FRONT_WALL = LidarSensors(distances_m=(5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0))


def test_controller_hands_off_for_a_close_ahead_competitor_too() -> None:
    close_ahead = CameraCompetitorReading(distance_m=2.0, angle_degrees=0.0)
    hazard = RobotSensors(camera=CameraSensors(competitors=(close_ahead,)))
    reference_expert = leaderboard_expert.create_controller()

    assert Controller()(hazard) == reference_expert(hazard)


def test_controller_ignores_a_distant_competitor() -> None:
    far_ahead = CameraCompetitorReading(distance_m=20.0, angle_degrees=0.0)
    not_a_hazard = RobotSensors(camera=CameraSensors(competitors=(far_ahead,)))
    reference_sac = race_faster.create_controller()

    assert Controller()(not_a_hazard) == reference_sac(not_a_hazard)


def test_controller_ignores_a_near_but_beside_competitor() -> None:
    beside = CameraCompetitorReading(distance_m=1.0, angle_degrees=90.0)
    not_a_hazard = RobotSensors(camera=CameraSensors(competitors=(beside,)))
    reference_sac = race_faster.create_controller()

    assert Controller()(not_a_hazard) == reference_sac(not_a_hazard)


def test_controller_defers_to_sac_when_no_hazard_is_present() -> None:
    open_track = RobotSensors()
    reference_sac = race_faster.create_controller()

    assert Controller()(open_track) == reference_sac(open_track)


def test_controller_hands_off_to_the_expert_during_a_hazard() -> None:
    hazard = RobotSensors(wall_lidar=_CLOSE_FRONT_WALL)
    reference_expert = leaderboard_expert.create_controller()

    assert Controller()(hazard) == reference_expert(hazard)


def test_controller_returns_commands_in_documented_ranges() -> None:
    tricky = RobotSensors(
        wall_lidar=_CLOSE_FRONT_WALL,
        camera=CameraSensors(center_offset_m=3.0, heading_error_degrees=-45.0),
        odometry=OdometrySensors(speed_mps=25.0),
    )

    command = Controller()(tricky)

    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0


def test_copy_for_car_gives_the_expert_independent_recovery_state() -> None:
    original = Controller()
    stuck_in_a_hazard = RobotSensors(
        contact=ContactSensors(wall=0.1), odometry=OdometrySensors(speed_mps=0.0), wall_lidar=_CLOSE_FRONT_WALL
    )
    original(stuck_in_a_hazard)  # triggers the original's own expert into a 28-tick recovery

    copy = original.copy_for_car()
    still_hazardous_but_not_stuck = RobotSensors(wall_lidar=_CLOSE_FRONT_WALL)

    # A fresh copy's expert must not be mid-recovery just because the original's is.
    assert copy(still_hazardous_but_not_stuck).throttle != -1.0
