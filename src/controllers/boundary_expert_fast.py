"""High-risk expert that treats track boundaries, not the centerline, as limits.

This controller targets leaderboard progress. Processed track geometry supplies
turn direction, while LiDAR supplies the hard constraint: do not hit a wall or
another blocker. Small centerline errors are intentionally ignored.
"""

from __future__ import annotations

from math import isfinite

from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Boundary Expert Fast"
RACING_COLOR: str = "#E6007A"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _finite_distance(distance_m: float, cap_m: float = 30.0) -> float:
    return min(distance_m, cap_m) if isfinite(distance_m) else cap_m


class Controller:
    """Carry high speed while steering away from actual nearby boundaries."""

    def __init__(self) -> None:
        self._recovery_ticks_remaining = 0
        self._recovery_steer = 0.0

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        walls = sensors.wall_lidar
        front_wall_m = _finite_distance(walls.front_m)

        if sensors.contact.wall > 0.0 or front_wall_m < 0.45:
            self._recovery_ticks_remaining = 34
            left_space = _finite_distance(walls.left_m) + _finite_distance(walls.front_left_m)
            right_space = _finite_distance(walls.right_m) + _finite_distance(walls.front_right_m)
            self._recovery_steer = -0.82 if left_space > right_space else 0.82

        if self._recovery_ticks_remaining > 0:
            self._recovery_ticks_remaining -= 1
            return RobotCommand(throttle=-0.7, steer=self._recovery_steer)

        if not sensors.camera.visible:
            open_side = -0.35 if walls.left_m > walls.right_m else 0.35
            return RobotCommand(throttle=0.2, steer=open_side)

        camera = sensors.camera
        lookahead = camera.lookahead_offsets_m
        near_offset = lookahead[0] if lookahead else camera.center_offset_m
        middle_offset = lookahead[len(lookahead) // 2] if lookahead else near_offset
        far_offset = lookahead[-1] if lookahead else middle_offset

        # Heading and lookahead describe where the track goes. Center position
        # has no influence until the car is substantially displaced.
        heading_term = camera.heading_error_degrees / 54.0
        lookahead_term = 0.07 * near_offset + 0.052 * middle_offset + 0.034 * far_offset
        excess_offset_m = max(0.0, abs(camera.center_offset_m) - 1.7)
        center_term = 0.10 * excess_offset_m * (1.0 if camera.center_offset_m > 0.0 else -1.0)

        left_m = _finite_distance(walls.left_m)
        right_m = _finite_distance(walls.right_m)
        front_left_m = _finite_distance(walls.front_left_m)
        front_right_m = _finite_distance(walls.front_right_m)

        # Boundary pressure is dormant in open track and rises steeply only
        # near the wall. Diagonal beams react early enough for high-speed turns.
        wall_term = 0.0
        side_clearance_m = 1.25
        diagonal_clearance_m = 2.15
        if left_m < side_clearance_m:
            wall_term += 0.62 * (side_clearance_m - left_m) / side_clearance_m
        if right_m < side_clearance_m:
            wall_term -= 0.62 * (side_clearance_m - right_m) / side_clearance_m
        if front_left_m < diagonal_clearance_m:
            wall_term += 0.48 * (diagonal_clearance_m - front_left_m) / diagonal_clearance_m
        if front_right_m < diagonal_clearance_m:
            wall_term -= 0.48 * (diagonal_clearance_m - front_right_m) / diagonal_clearance_m

        # Full LiDAR sees cars and blockers. If its front beam is much shorter
        # than wall-only LiDAR, bias toward the side with more obstacle space.
        obstacle_term = 0.0
        obstacle_front_m = _finite_distance(sensors.lidar.front_m)
        if obstacle_front_m < 4.0 and obstacle_front_m + 0.25 < front_wall_m:
            obstacle_left_m = _finite_distance(sensors.lidar.front_left_m)
            obstacle_right_m = _finite_distance(sensors.lidar.front_right_m)
            obstacle_term = -0.48 if obstacle_left_m > obstacle_right_m else 0.48

        steer = _clamp(heading_term + lookahead_term + center_term + wall_term + obstacle_term, -1.0, 1.0)

        # Stay near the vehicle's demonstrated top speed. Only severe upcoming
        # geometry or a genuinely short front beam causes meaningful braking.
        bend_score = abs(far_offset - near_offset) + 0.4 * abs(middle_offset - near_offset)
        bend_caution = _clamp(bend_score / 6.0, 0.0, 2.0)
        heading_caution = _clamp(abs(camera.heading_error_degrees) / 55.0, 0.0, 2.0)
        target_speed_mps = _clamp(10.7 - 1.75 * max(bend_caution, heading_caution), 6.6, 10.7)

        if front_wall_m < 3.0:
            front_limited_speed = _clamp(2.0 + 2.2 * (front_wall_m - 0.45), 2.0, 7.5)
            target_speed_mps = min(target_speed_mps, front_limited_speed)
        if obstacle_front_m < 1.6 and obstacle_front_m + 0.25 < front_wall_m:
            target_speed_mps = min(target_speed_mps, 4.5)

        speed_error = target_speed_mps - sensors.odometry.speed_mps
        throttle = _clamp(0.42 * speed_error, -1.0, 1.0)
        return RobotCommand(throttle=throttle, steer=steer)


def create_controller() -> Controller:
    return Controller()
