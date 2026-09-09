"""Aggressive phase-based controller for the 30-second leaderboard trial.

The controller uses only documented public sensors. Unlike the reliability
experts, it treats processed centerline geometry as turn information rather
than a path that must be followed exactly, and it deliberately accepts more
risk to maximize forward progress.
"""

from __future__ import annotations

from math import isfinite

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Leaderboard Expert"
RACING_COLOR: str = "#FF1744"

# Keep normal cornering moving; reversing is reserved for getting unstuck.
MIN_ROLLING_SPEED_MPS = 6.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _finite_distance(distance_m: float, cap_m: float = 80.0) -> float:
    return min(distance_m, cap_m) if isfinite(distance_m) else cap_m


class Controller:
    """Accelerate hard, brake decisively, and carry speed between boundaries."""

    def __init__(self) -> None:
        self._previous_steer = 0.0
        self._recovery_ticks_remaining = 0
        self._recovery_steer = 0.0
        self._previous_throttle = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._race_command(sensors)
        # Negative throttle arms the simulator's brake-before-reverse state.
        # A direct switch to positive throttle keeps braking until near zero;
        # one neutral tick clears that state so acceleration can resume rolling.
        if self._previous_throttle < 0.0 and command.throttle > 0.0:
            command = RobotCommand(throttle=0.0, steer=command.steer)
        self._previous_throttle = command.throttle
        return command

    def _race_command(self, sensors: RobotSensors) -> RobotCommand:
        walls = sensors.wall_lidar
        front_wall_m = _finite_distance(walls.front_m)

        speed_mps = sensors.odometry.speed_mps
        if (
            self._recovery_ticks_remaining == 0
            and abs(speed_mps) < 2.0
            and (sensors.contact.wall > 0.0 or front_wall_m < 0.35)
        ):
            self._recovery_ticks_remaining = 28
            left_space = _finite_distance(walls.left_m) + _finite_distance(walls.front_left_m)
            right_space = _finite_distance(walls.right_m) + _finite_distance(walls.front_right_m)
            self._recovery_steer = -0.9 if left_space > right_space else 0.9

        if self._recovery_ticks_remaining > 0:
            self._recovery_ticks_remaining -= 1
            self._previous_steer = self._recovery_steer
            return RobotCommand(throttle=-1.0, steer=self._recovery_steer)

        if not sensors.camera.visible:
            open_side = -0.4 if walls.left_m > walls.right_m else 0.4
            self._previous_steer = open_side
            throttle = 1.0 if speed_mps < MIN_ROLLING_SPEED_MPS else -0.2
            return RobotCommand(throttle=throttle, steer=open_side)

        camera = sensors.camera
        offsets = camera.lookahead_offsets_m
        near_offset = offsets[0] if offsets else camera.center_offset_m
        middle_offset = offsets[len(offsets) // 2] if offsets else near_offset
        far_offset = offsets[-1] if offsets else middle_offset

        left_m = _finite_distance(walls.left_m)
        right_m = _finite_distance(walls.right_m)
        front_left_m = _finite_distance(walls.front_left_m)
        front_right_m = _finite_distance(walls.front_right_m)

        # Lookahead and heading provide turn direction. Center correction is a
        # low-gain fallback that starts only when displacement is already large.
        heading_term = camera.heading_error_degrees / 42.0
        lookahead_term = 0.065 * near_offset + 0.057 * middle_offset + 0.05 * far_offset
        excess_offset_m = max(0.0, abs(camera.center_offset_m) - 2.0)
        center_term = 0.08 * excess_offset_m * (1.0 if camera.center_offset_m > 0.0 else -1.0)

        # High-speed yaw damping limits fishtailing without reducing the initial
        # steering request that begins a corner.
        yaw_damping = -sensors.imu.yaw_rate_degrees_per_s / 500.0

        wall_term = 0.0
        if left_m < 1.15:
            wall_term += 0.75 * (1.15 - left_m) / 1.15
        if right_m < 1.15:
            wall_term -= 0.75 * (1.15 - right_m) / 1.15
        if front_left_m < 2.4:
            wall_term += 0.65 * (2.4 - front_left_m) / 2.4
        if front_right_m < 2.4:
            wall_term -= 0.65 * (2.4 - front_right_m) / 2.4

        raw_steer = _clamp(
            heading_term + lookahead_term + center_term + yaw_damping + wall_term,
            -1.0,
            1.0,
        )
        # Bound the per-tick change to prevent rapid left/right corrections at
        # high speed, while still allowing a full transition in about 0.2 s.
        steer = _clamp(raw_steer, self._previous_steer - 0.14, self._previous_steer + 0.14)
        self._previous_steer = steer

        bend_score = abs(far_offset - near_offset) + 0.45 * abs(middle_offset - near_offset)
        heading_error = abs(camera.heading_error_degrees)

        # Explicit phases make acceleration and braking decisive. The thresholds
        # intentionally keep much more speed than the reliability controllers.
        if bend_score < 1.2 and heading_error < 9.0:
            target_speed_mps = 38.0
        elif bend_score < 3.2 and heading_error < 18.0:
            target_speed_mps = 36.0
        elif bend_score < 6.5 and heading_error < 32.0:
            target_speed_mps = 31.0
        else:
            target_speed_mps = 23.0

        # A wall appearing rapidly in the forward beam overrides the geometry
        # phase. The horizon grows with speed to leave room for hard braking.
        braking_horizon_m = max(3.5, sensors.odometry.speed_mps * 0.42)
        if front_wall_m < braking_horizon_m:
            target_speed_mps = min(
                target_speed_mps,
                _clamp(2.5 * (front_wall_m - 0.35), MIN_ROLLING_SPEED_MPS, 18.0),
            )

        obstacle_front_m = _finite_distance(sensors.lidar.front_m)
        if obstacle_front_m + 0.25 < front_wall_m and obstacle_front_m < 5.0:
            target_speed_mps = min(target_speed_mps, 12.0)

        speed_error = target_speed_mps - sensors.odometry.speed_mps
        if speed_error > 1.0:
            throttle = 1.0
        elif speed_error < -1.0:
            throttle = -1.0
        else:
            throttle = _clamp(speed_error * 0.45, -0.45, 0.65)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> Controller:
    return Controller()
