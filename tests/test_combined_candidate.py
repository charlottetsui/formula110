from __future__ import annotations

from controllers import leaderboard_expert
from controllers.combined_candidate import Controller
from racing import CameraSensors, ContactSensors, OdometrySensors, RobotSensors


def test_controller_returns_commands_in_documented_ranges() -> None:
    tricky = RobotSensors(
        camera=CameraSensors(center_offset_m=3.0, heading_error_degrees=-45.0),
        odometry=OdometrySensors(speed_mps=25.0),
    )

    command = Controller()(tricky)

    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0


def test_controller_beats_crash_fast_on_the_open_track() -> None:
    # A weak sanity check that doesn't require the real simulator: on a clear track
    # the combined controller should commit to forward progress, not stall or reverse.
    command = Controller()(RobotSensors())

    assert command.throttle > 0.0


def test_controller_passes_through_the_expert_unmodified_during_recovery() -> None:
    stuck = RobotSensors(contact=ContactSensors(wall=0.1), odometry=OdometrySensors(speed_mps=0.0))
    reference_expert = leaderboard_expert.create_controller()

    # leaderboard_expert.Controller's recovery command is always exactly (-1.0, +-0.9);
    # the combined controller must reproduce it exactly, not blend a correction into it.
    assert Controller()(stuck) == reference_expert(stuck)


def test_copy_for_car_gives_the_expert_independent_recovery_state() -> None:
    original = Controller()
    stuck = RobotSensors(contact=ContactSensors(wall=0.1), odometry=OdometrySensors(speed_mps=0.0))
    original(stuck)  # triggers the original's own expert into a 28-tick recovery

    copy = original.copy_for_car()

    # A fresh copy's expert must not be mid-recovery just because the original's is.
    assert copy(RobotSensors()).throttle != -1.0
