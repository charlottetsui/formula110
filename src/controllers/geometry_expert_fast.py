"""Leaderboard-oriented rule-based expert with anticipatory speed control.

The conservative geometry expert remains unchanged as a reliability baseline.
This variant uses the same public sensors, but prioritizes progress and lap
speed. It accelerates at the physical command limit on clear straights and
uses upcoming geometry to decide when that speed is no longer safe.
"""

from __future__ import annotations

from math import isfinite

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Geometry Expert Fast"
RACING_COLOR: str = "#7B2CFF"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _finite_distance(distance_m: float, cap_m: float = 30.0) -> float:
    return min(distance_m, cap_m) if isfinite(distance_m) else cap_m


class Controller:
    """Prioritize race speed while retaining bounded wall recovery."""

    def __init__(self) -> None:
        self._recovery_ticks_remaining = 0
        self._recovery_steer = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        walls = sensors.wall_lidar
        front_m = _finite_distance(walls.front_m)

        if sensors.contact.wall > 0.0 or front_m < 0.65:
            self._recovery_ticks_remaining = 42
            left_space = _finite_distance(walls.left_m) + _finite_distance(walls.front_left_m)
            right_space = _finite_distance(walls.right_m) + _finite_distance(walls.front_right_m)
            self._recovery_steer = -0.78 if left_space > right_space else 0.78

        if self._recovery_ticks_remaining > 0:
            self._recovery_ticks_remaining -= 1
            return RobotCommand(throttle=-0.58, steer=self._recovery_steer)

        if not sensors.camera.visible:
            cautious_steer = -0.25 if walls.left_m > walls.right_m else 0.25
            return RobotCommand(throttle=0.12, steer=cautious_steer)

        camera = sensors.camera
        lookahead = camera.lookahead_offsets_m
        near_offset = lookahead[0] if lookahead else camera.center_offset_m
        middle_offset = lookahead[len(lookahead) // 2] if lookahead else near_offset
        far_offset = lookahead[-1] if lookahead else middle_offset

        # Strong lookahead response positions the car early enough to carry
        # speed through a bend instead of making a late, speed-killing turn.
        center_term = 0.17 * camera.center_offset_m
        heading_term = camera.heading_error_degrees / 58.0
        lookahead_term = 0.07 * near_offset + 0.045 * middle_offset + 0.028 * far_offset

        left_m = _finite_distance(walls.left_m)
        right_m = _finite_distance(walls.right_m)
        safe_side_distance_m = 1.7
        wall_term = 0.0
        if left_m < safe_side_distance_m:
            wall_term += 0.42 * (safe_side_distance_m - left_m) / safe_side_distance_m
        if right_m < safe_side_distance_m:
            wall_term -= 0.42 * (safe_side_distance_m - right_m) / safe_side_distance_m

        steer = _clamp(center_term + heading_term + lookahead_term + wall_term, -1.0, 1.0)

        # Center offset alone is not a reason to crawl on a straight. Only the
        # portion beyond 1.15 m reduces speed, while heading and upcoming bend
        # geometry retain authority to request early braking.
        bend_score = abs(far_offset - near_offset) + 0.4 * abs(middle_offset - near_offset)
        bend_caution = bend_score / 5.5
        heading_caution = abs(camera.heading_error_degrees) / 48.0
        center_caution = max(0.0, abs(camera.center_offset_m) - 1.15) / 2.5
        steering_caution = max(0.0, abs(steer) - 0.48) / 0.52
        caution = max(bend_caution, heading_caution, center_caution, steering_caution)

        target_speed_mps = _clamp(9.6 - 4.0 * caution, 2.0, 9.6)

        # At higher speeds the front beam needs a longer braking horizon.
        if front_m < 5.5:
            front_limited_speed = _clamp(1.0 + 1.05 * (front_m - 0.65), 1.0, 5.8)
            target_speed_mps = min(target_speed_mps, front_limited_speed)

        speed_error = target_speed_mps - sensors.odometry.speed_mps
        throttle = _clamp(0.36 * speed_error, -0.85, 1.0)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> Controller:
    """Return independent recovery state for each car and race."""
    return Controller()
