from __future__ import annotations

import pytest

from racing.student.api import (
    CameraCompetitorReading,
    CameraSensors,
    ContactSensors,
    ImuSensors,
    LidarSensors,
    OdometrySensors,
    RobotSensors,
)
from training.reward import (
    IDLE_SPEED_MPS,
    NEAR_ELIMINATION_DAMAGE,
    ROBOT_WARNING_ANGLE_DEGREES,
    ROBOT_WARNING_DISTANCE_M,
    WALL_PROXIMITY_SPEED_SCALE_MPS,
    YAW_RATE_CHANGE_SCALE_DEGREES_PER_S,
    YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S,
    is_new_episode,
    is_terminal,
    step_reward,
)


def test_step_reward_rewards_forward_progress_aligned_with_track_heading() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    moving_straight = RobotSensors(
        dt_s=1 / 60,
        odometry=OdometrySensors(speed_mps=5.0),
        camera=CameraSensors(heading_error_degrees=0.0),
    )

    reward = step_reward(previous, moving_straight)

    assert reward > 0.0


def test_step_reward_penalizes_reverse_driving() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    reversing = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=-5.0))

    reward = step_reward(previous, reversing)

    assert reward < 0.0


def test_step_reward_penalizes_new_damage_this_tick() -> None:
    previous = RobotSensors(contact=ContactSensors(damage=0.1))
    undamaged_next = RobotSensors(contact=ContactSensors(damage=0.1))
    damaged_next = RobotSensors(contact=ContactSensors(damage=0.5))

    assert step_reward(previous, damaged_next) < step_reward(previous, undamaged_next)


def test_step_reward_penalizes_center_offset() -> None:
    previous = RobotSensors()
    centered = RobotSensors(camera=CameraSensors(center_offset_m=0.0))
    off_center = RobotSensors(camera=CameraSensors(center_offset_m=4.0))

    assert step_reward(previous, off_center) < step_reward(previous, centered)


def test_step_reward_penalizes_close_walls() -> None:
    previous = RobotSensors()
    open_track = RobotSensors()  # default wall_lidar beams are all math.inf (no hit)
    wall_ahead = RobotSensors(wall_lidar=LidarSensors(distances_m=tuple(1.0 for _ in range(7))))

    assert step_reward(previous, wall_ahead) < step_reward(previous, open_track)


def test_step_reward_penalizes_standing_still() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    idle = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=0.0))
    slow_forward = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=IDLE_SPEED_MPS))

    assert step_reward(previous, idle) < 0.0
    assert step_reward(previous, idle) < step_reward(previous, slow_forward)


def test_step_reward_idle_penalty_only_applies_below_the_speed_threshold() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    just_below_threshold = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=IDLE_SPEED_MPS - 0.01))
    at_threshold = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=IDLE_SPEED_MPS))

    assert step_reward(previous, just_below_threshold) < step_reward(previous, at_threshold)


def test_step_reward_progress_keeps_scaling_with_speed_uncapped() -> None:
    # 2026-09-02: MAX_REWARDED_SPEED_MPS was removed entirely -- there is no
    # speed at which additional progress reward stops accruing anymore.
    previous = RobotSensors(dt_s=1 / 60)
    fast = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=10.0))
    much_faster = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=40.0))

    assert step_reward(previous, fast) < step_reward(previous, much_faster)


def test_step_reward_wall_proximity_penalty_grows_with_speed() -> None:
    # Isolates the proximity-vs-speed scaling from the (now uncapped) progress
    # term by comparing the *cost of an identical nearby wall* at two speeds --
    # the progress term is identical between the "open" and "near wall" cases
    # at a fixed speed, so it cancels out of the subtraction.
    previous = RobotSensors()
    close_wall = LidarSensors(distances_m=tuple(1.0 for _ in range(7)))
    slow_open = RobotSensors(odometry=OdometrySensors(speed_mps=1.0))
    slow_near_wall = RobotSensors(odometry=OdometrySensors(speed_mps=1.0), wall_lidar=close_wall)
    fast_open = RobotSensors(odometry=OdometrySensors(speed_mps=30.0))
    fast_near_wall = RobotSensors(odometry=OdometrySensors(speed_mps=30.0), wall_lidar=close_wall)

    slow_wall_cost = step_reward(previous, slow_open) - step_reward(previous, slow_near_wall)
    fast_wall_cost = step_reward(previous, fast_open) - step_reward(previous, fast_near_wall)

    assert fast_wall_cost > slow_wall_cost


def test_step_reward_wall_proximity_multiplier_doubles_at_the_speed_scale() -> None:
    previous = RobotSensors()
    close_wall = LidarSensors(distances_m=tuple(1.0 for _ in range(7)))
    stationary_open = RobotSensors(odometry=OdometrySensors(speed_mps=0.0))
    stationary_near_wall = RobotSensors(odometry=OdometrySensors(speed_mps=0.0), wall_lidar=close_wall)
    at_scale_open = RobotSensors(odometry=OdometrySensors(speed_mps=WALL_PROXIMITY_SPEED_SCALE_MPS))
    at_scale_near_wall = RobotSensors(
        odometry=OdometrySensors(speed_mps=WALL_PROXIMITY_SPEED_SCALE_MPS), wall_lidar=close_wall
    )

    stationary_cost = step_reward(previous, stationary_open) - step_reward(previous, stationary_near_wall)
    at_scale_cost = step_reward(previous, at_scale_open) - step_reward(previous, at_scale_near_wall)

    assert at_scale_cost == pytest.approx(stationary_cost * 2.0)


def test_step_reward_robot_proximity_penalty_is_currently_disabled() -> None:
    # WEIGHT_ROBOT_PROXIMITY was tried unrestricted (2026-09-07, broad overcaution
    # regression) and angle-restricted (2026-09-08, worse -- 17/20 evaluation races
    # completed zero laps, the policy learned to simply never approach a competitor
    # rather than pass cleanly). Disabled again after both failures -- verify a nearby,
    # roughly-ahead competitor currently has no effect on reward.
    previous = RobotSensors()
    no_competitors = RobotSensors(camera=CameraSensors(competitors=()))
    nearby_ahead = RobotSensors(
        camera=CameraSensors(competitors=(CameraCompetitorReading(distance_m=2.0, angle_degrees=0.0),))
    )

    assert step_reward(previous, nearby_ahead) == step_reward(previous, no_competitors)


def test_robot_proximity_angle_restriction_logic_is_still_correct_though_disabled() -> None:
    # The angle-restriction mechanism itself is mechanically correct (this is what was
    # tested in training) -- the *policy's response* to it is what failed, not a bug in
    # the reward code. Verify the mechanism's own logic directly via a manual weight
    # override, so a future re-enable starts from known-correct angle filtering.
    import training.reward as reward_module

    original_weight = reward_module.WEIGHT_ROBOT_PROXIMITY
    reward_module.WEIGHT_ROBOT_PROXIMITY = 1.0
    try:
        previous = RobotSensors()
        far_ahead_reading = CameraCompetitorReading(distance_m=7.0, angle_degrees=0.0)
        far_ahead_only = RobotSensors(camera=CameraSensors(competitors=(far_ahead_reading,)))
        beside_reading = CameraCompetitorReading(distance_m=1.0, angle_degrees=ROBOT_WARNING_ANGLE_DEGREES + 10.0)
        far_ahead_and_near_but_beside = RobotSensors(
            camera=CameraSensors(competitors=(far_ahead_reading, beside_reading))
        )
        ahead_reading = CameraCompetitorReading(distance_m=1.0, angle_degrees=-10.0)
        far_ahead_and_near_ahead = RobotSensors(camera=CameraSensors(competitors=(far_ahead_reading, ahead_reading)))

        # A near-but-beside competitor doesn't qualify (outside the warning angle).
        assert step_reward(previous, far_ahead_and_near_but_beside) == step_reward(previous, far_ahead_only)
        # A near-and-ahead competitor does qualify, and is nearer than the far one.
        assert step_reward(previous, far_ahead_and_near_ahead) < step_reward(previous, far_ahead_only)
        # A distant-even-if-ahead competitor doesn't qualify either.
        distant_reading = CameraCompetitorReading(distance_m=ROBOT_WARNING_DISTANCE_M, angle_degrees=0.0)
        distant_competitor = RobotSensors(camera=CameraSensors(competitors=(distant_reading,)))
        no_competitors = RobotSensors(camera=CameraSensors(competitors=()))
        assert step_reward(previous, distant_competitor) == step_reward(previous, no_competitors)
    finally:
        reward_module.WEIGHT_ROBOT_PROXIMITY = original_weight


def test_step_reward_applies_a_one_time_penalty_for_the_terminal_transition() -> None:
    previous = RobotSensors(contact=ContactSensors(damage=0.85))
    survives = RobotSensors(contact=ContactSensors(damage=0.88))  # still below NEAR_ELIMINATION_DAMAGE
    eliminated = RobotSensors(contact=ContactSensors(damage=0.95))  # crosses NEAR_ELIMINATION_DAMAGE

    # Both take on similar new damage this tick; only `eliminated` is terminal.
    assert step_reward(previous, eliminated) < step_reward(previous, survives)


def test_step_reward_steering_smoothness_penalty_is_currently_disabled() -> None:
    # WEIGHT_STEERING_SMOOTHNESS was tried at 0.1 (2026-09-02) and reverted to
    # 0.0 after a clear regression -- verify yaw-rate swings have no effect
    # on reward while the mechanism stays disabled by weight.
    previous = RobotSensors(imu=ImuSensors(yaw_rate_degrees_per_s=0.0))
    small_change = RobotSensors(imu=ImuSensors(yaw_rate_degrees_per_s=5.0))
    large_change = RobotSensors(imu=ImuSensors(yaw_rate_degrees_per_s=YAW_RATE_CHANGE_SCALE_DEGREES_PER_S * 4))

    assert step_reward(previous, large_change) == step_reward(previous, small_change)


def test_step_reward_steering_reversal_penalty_is_currently_disabled() -> None:
    # WEIGHT_STEERING_REVERSAL was implemented as a hesitation-reduction hypothesis
    # (2026-09-07) but disabled (weight 0.0) before testing once per-tick diagnosis
    # found the real cause of low-progress time was blind car-collision, not
    # steering direction reversal -- verify a reversal currently has no effect on
    # reward while the mechanism stays disabled by weight.
    turning_left = RobotSensors(imu=ImuSensors(yaw_rate_degrees_per_s=YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S + 10.0))
    turning_right = RobotSensors(
        imu=ImuSensors(yaw_rate_degrees_per_s=-(YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S + 10.0))
    )

    assert step_reward(turning_left, turning_right) == step_reward(turning_left, turning_left)


def test_is_terminal_true_at_and_above_near_elimination_threshold() -> None:
    assert is_terminal(RobotSensors(contact=ContactSensors(damage=NEAR_ELIMINATION_DAMAGE)))
    assert is_terminal(RobotSensors(contact=ContactSensors(damage=1.0)))
    assert not is_terminal(RobotSensors(contact=ContactSensors(damage=NEAR_ELIMINATION_DAMAGE - 0.01)))


def test_is_new_episode_true_only_on_tick_zero() -> None:
    assert is_new_episode(RobotSensors(tick=0))
    assert not is_new_episode(RobotSensors(tick=1))
