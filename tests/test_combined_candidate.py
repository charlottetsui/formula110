from __future__ import annotations

from controllers import imitation
from controllers.combined_candidate import Controller
from racing import CameraSensors, LidarSensors, OdometrySensors, RobotSensors


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


def test_controller_passes_through_the_clone_unmodified_outside_a_hazard() -> None:
    open_track = RobotSensors(
        camera=CameraSensors(center_offset_m=3.0, heading_error_degrees=-45.0),
        odometry=OdometrySensors(speed_mps=25.0),
    )
    reference_clone = imitation.create_controller()

    # No wall/competitor proximity hazard here -- the correction must be gated off entirely,
    # so the combined controller's command must exactly equal the clone's own raw command.
    assert Controller()(open_track) == reference_clone(open_track)


def test_controller_applies_a_residual_correction_during_a_hazard() -> None:
    close_wall = LidarSensors(distances_m=tuple(1.0 for _ in range(7)))
    hazard = RobotSensors(wall_lidar=close_wall)
    reference_clone = imitation.create_controller()

    # Neither half alone produces the output on a hazard tick: the combined controller's
    # command must differ from the clone's own raw command, since a bounded SAC correction
    # is added on top of it.
    assert Controller()(hazard) != reference_clone(hazard)


def test_copy_for_car_gives_the_clone_independent_history_state() -> None:
    original = Controller()
    original(RobotSensors(tick=0, odometry=OdometrySensors(speed_mps=10.0)))
    original(RobotSensors(tick=1, odometry=OdometrySensors(speed_mps=20.0)))  # builds up clone history

    copy = original.copy_for_car()
    probe = RobotSensors(tick=5, camera=CameraSensors(heading_error_degrees=12.0))

    # A copy fed a single tick from a clean start must behave exactly like a brand-new
    # Controller fed that same tick -- if copy_for_car shared the clone's history deque
    # instead of giving the copy its own, the original's two prior ticks would leak in.
    assert copy(probe) == Controller()(probe)
