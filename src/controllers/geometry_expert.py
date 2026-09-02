"""Rule-based expert that follows the processed track geometry.

This controller deliberately uses only the public ``RobotSensors`` contract so
that its decisions can later serve as learnable imitation-learning labels.
"""

from __future__ import annotations

from math import isfinite

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Geometry Expert"
RACING_COLOR: str = "#2BA84A"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _finite_distance(distance_m: float, cap_m: float = 30.0) -> float:
    """Turn the documented infinite LiDAR no-hit value into a finite value."""
    return min(distance_m, cap_m) if isfinite(distance_m) else cap_m


class Controller:
    """Follow the centerline and briefly reverse when the car becomes trapped."""

    def __init__(self) -> None:
        self._recovery_ticks_remaining = 0
        self._recovery_steer = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        walls = sensors.wall_lidar
        front_m = _finite_distance(walls.front_m)

        # A short stateful recovery is more reliable than changing direction for
        # only the single tick on which contact is first observed.
        if sensors.contact.wall > 0.0 or front_m < 0.65:
            self._recovery_ticks_remaining = 42
            left_space = _finite_distance(walls.left_m) + _finite_distance(walls.front_left_m)
            right_space = _finite_distance(walls.right_m) + _finite_distance(walls.front_right_m)
            self._recovery_steer = -0.75 if left_space > right_space else 0.75

        if self._recovery_ticks_remaining > 0:
            self._recovery_ticks_remaining -= 1
            return RobotCommand(throttle=-0.55, steer=self._recovery_steer)

        if not sensors.camera.visible:
            # The camera is currently always visible, but the sensor contract
            # asks controllers to handle a missing reading.
            cautious_steer = -0.25 if walls.left_m > walls.right_m else 0.25
            return RobotCommand(throttle=0.12, steer=cautious_steer)

        camera = sensors.camera
        lookahead = camera.lookahead_offsets_m
        near_offset = lookahead[0] if lookahead else camera.center_offset_m
        middle_offset = lookahead[len(lookahead) // 2] if lookahead else near_offset
        far_offset = lookahead[-1] if lookahead else middle_offset

        # Positive camera values mean the desired path is to the right, which
        # matches positive/right steering in RobotCommand.
        center_term = 0.20 * camera.center_offset_m
        heading_term = camera.heading_error_degrees / 70.0
        lookahead_term = 0.055 * near_offset + 0.035 * middle_offset + 0.02 * far_offset

        # Side-wall pressure gently moves the car away from a close boundary.
        left_m = _finite_distance(walls.left_m)
        right_m = _finite_distance(walls.right_m)
        safe_side_distance_m = 1.6
        wall_term = 0.0
        if left_m < safe_side_distance_m:
            wall_term += 0.35 * (safe_side_distance_m - left_m) / safe_side_distance_m
        if right_m < safe_side_distance_m:
            wall_term -= 0.35 * (safe_side_distance_m - right_m) / safe_side_distance_m

        steer = _clamp(center_term + heading_term + lookahead_term + wall_term, -0.9, 0.9)

        # Slow down for large pose errors and for bends indicated by changing
        # lookahead offsets. This favors reliable demonstrations over lap speed.
        bend_score = abs(far_offset - near_offset) + 0.35 * abs(middle_offset - near_offset)
        pose_score = abs(camera.heading_error_degrees) / 35.0 + abs(camera.center_offset_m) / 2.5
        caution = max(bend_score / 5.0, pose_score)
        target_speed_mps = _clamp(4.4 - 1.8 * caution, 1.2, 4.4)

        if front_m < 3.5:
            target_speed_mps = min(target_speed_mps, _clamp(front_m - 0.5, 0.8, 2.2))

        speed_error = target_speed_mps - sensors.odometry.speed_mps
        throttle = _clamp(0.24 * speed_error, -0.45, 0.55)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> Controller:
    """Return independent recovery state for each car and race."""
    return Controller()
