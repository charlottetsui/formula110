from __future__ import annotations

from racing.student.api import CameraSensors, ContactSensors, LidarSensors, OdometrySensors, RobotSensors
from training.reward import (
    IDLE_SPEED_MPS,
    MAX_REWARDED_SPEED_MPS,
    NEAR_ELIMINATION_DAMAGE,
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


def test_step_reward_progress_scales_with_speed_below_the_cap() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    slower = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=MAX_REWARDED_SPEED_MPS / 2))
    faster = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=MAX_REWARDED_SPEED_MPS))

    assert step_reward(previous, slower) < step_reward(previous, faster)


def test_step_reward_progress_saturates_above_the_speed_cap() -> None:
    previous = RobotSensors(dt_s=1 / 60)
    at_cap = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=MAX_REWARDED_SPEED_MPS))
    far_above_cap = RobotSensors(dt_s=1 / 60, odometry=OdometrySensors(speed_mps=MAX_REWARDED_SPEED_MPS * 4))

    assert step_reward(previous, at_cap) == step_reward(previous, far_above_cap)


def test_step_reward_applies_a_one_time_penalty_for_the_terminal_transition() -> None:
    previous = RobotSensors(contact=ContactSensors(damage=0.85))
    survives = RobotSensors(contact=ContactSensors(damage=0.88))  # still below NEAR_ELIMINATION_DAMAGE
    eliminated = RobotSensors(contact=ContactSensors(damage=0.95))  # crosses NEAR_ELIMINATION_DAMAGE

    # Both take on similar new damage this tick; only `eliminated` is terminal.
    assert step_reward(previous, eliminated) < step_reward(previous, survives)


def test_is_terminal_true_at_and_above_near_elimination_threshold() -> None:
    assert is_terminal(RobotSensors(contact=ContactSensors(damage=NEAR_ELIMINATION_DAMAGE)))
    assert is_terminal(RobotSensors(contact=ContactSensors(damage=1.0)))
    assert not is_terminal(RobotSensors(contact=ContactSensors(damage=NEAR_ELIMINATION_DAMAGE - 0.01)))


def test_is_new_episode_true_only_on_tick_zero() -> None:
    assert is_new_episode(RobotSensors(tick=0))
    assert not is_new_episode(RobotSensors(tick=1))
