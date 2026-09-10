from __future__ import annotations

import numpy as np
import pytest

from racing.student.api import CameraSensors, ContactSensors, OdometrySensors, RobotSensors
from training.controller import TrainableController, TrainingState
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent
from training.trajectory import BestTrajectoryTracker


def _training_state(*, warmup_steps: int = 5, update_every_n_steps: int = 2, batch_size: int = 4) -> TrainingState:
    return TrainingState(
        agent=SACAgent(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=(8, 8), seed=0),
        buffer=ReplayBuffer(capacity=1_000, observation_dim=OBSERVATION_DIM, action_dim=2),
        rng=np.random.default_rng(0),
        warmup_steps=warmup_steps,
        update_every_n_steps=update_every_n_steps,
        batch_size=batch_size,
    )


def _sensors(*, tick: int, damage: float = 0.0, distance_m: float = 0.0) -> RobotSensors:
    return RobotSensors(
        dt_s=1 / 60,
        tick=tick,
        odometry=OdometrySensors(speed_mps=2.0, distance_m=distance_m),
        camera=CameraSensors(center_offset_m=0.1, heading_error_degrees=1.0),
        contact=ContactSensors(damage=damage),
    )


def test_first_call_returns_a_valid_command_and_pushes_no_transition() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=True)

    command = controller(_sensors(tick=0))

    assert -1.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0
    assert len(state.buffer) == 0


def test_second_call_pushes_exactly_one_transition() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=True)
    controller(_sensors(tick=0))

    controller(_sensors(tick=1))

    assert len(state.buffer) == 1


def test_near_elimination_damage_marks_the_transition_done() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=True)
    controller(_sensors(tick=0))
    controller(_sensors(tick=1, damage=0.95))

    # The buffer holds exactly one transition, so sampling one element deterministically returns it.
    only_transition = state.buffer.sample(1, rng=np.random.default_rng(0))
    assert only_transition.dones[0] == 1.0


def test_evaluation_controller_never_writes_to_the_shared_buffer() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=False)

    for tick in range(10):
        controller(_sensors(tick=tick))

    assert len(state.buffer) == 0


def test_copy_for_car_shares_state_but_not_episode_history() -> None:
    state = _training_state()
    original = TrainableController(state=state, training=True)
    original(_sensors(tick=0))
    original(_sensors(tick=1))
    pushed_before_copy = len(state.buffer)

    copy = original.copy_for_car()
    copy(_sensors(tick=0))  # a fresh copy's first call has no previous observation of its own

    assert len(state.buffer) == pushed_before_copy

    copy(_sensors(tick=1))

    assert len(state.buffer) == pushed_before_copy + 1  # but shares the same learning state as `original`


def test_deterministic_action_is_repeatable_for_the_same_sensors() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=False, deterministic=True)
    same_sensors = _sensors(tick=5)

    first = controller(same_sensors)
    second = controller(same_sensors)

    assert (first.throttle, first.steer) == (second.throttle, second.steer)


def test_deterministic_false_overrides_training_flag_and_samples_stochastically() -> None:
    state = _training_state()
    controller = TrainableController(state=state, training=False, deterministic=False)
    same_sensors = _sensors(tick=5)

    first = controller(same_sensors)
    second = controller(same_sensors)

    assert (first.throttle, first.steer) != (second.throttle, second.steer)
    assert len(state.buffer) == 0  # still never writes to the buffer -- only action selection changed


def test_copy_for_car_preserves_the_deterministic_override() -> None:
    state = _training_state()
    original = TrainableController(state=state, training=False, deterministic=False)

    copy = original.copy_for_car()
    same_sensors = _sensors(tick=5)
    first = copy(same_sensors)
    second = copy(same_sensors)

    assert (first.throttle, first.steer) != (second.throttle, second.steer)


def test_trajectory_bonus_is_added_to_the_pushed_reward_when_present() -> None:
    without_tracker = _training_state()
    with_tracker = _training_state()
    with_tracker.trajectory = BestTrajectoryTracker(max_ticks=100)
    with_tracker.trajectory.update(tick=0, distance_m=0.0)
    with_tracker.trajectory.update(tick=1, distance_m=1.0)  # best-known pace: 1.0m gained by tick 1

    controller_without = TrainableController(state=without_tracker, training=True)
    controller_without(_sensors(tick=0, distance_m=0.0))
    controller_without(_sensors(tick=1, distance_m=2.0))  # gains 2.0m -- ahead of the 1.0m record

    controller_with = TrainableController(state=with_tracker, training=True)
    controller_with(_sensors(tick=0, distance_m=0.0))
    controller_with(_sensors(tick=1, distance_m=2.0))

    reward_without = without_tracker.buffer.sample(1, rng=np.random.default_rng(0)).rewards[0]
    reward_with = with_tracker.buffer.sample(1, rng=np.random.default_rng(0)).rewards[0]

    # gained 2.0m vs. a 1.0m record -> +1.0m bonus at WEIGHT_TRAJECTORY_BONUS=1.0
    assert reward_with == pytest.approx(reward_without + 1.0, abs=1e-4)


def test_trajectory_tracker_is_updated_from_controller_calls() -> None:
    state = _training_state()
    state.trajectory = BestTrajectoryTracker(max_ticks=100)
    controller = TrainableController(state=state, training=True)

    controller(_sensors(tick=0, distance_m=0.0))
    controller(_sensors(tick=1, distance_m=3.0))

    # a later, slower run at the same tick should now see a positive record to chase
    bonus = state.trajectory.bonus_m(previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.0)
    assert bonus == pytest.approx(1.0 - 3.0)


def test_n_step_one_matches_previous_single_step_behavior() -> None:
    state = _training_state()  # n_step defaults to 1
    controller = TrainableController(state=state, training=True)
    controller(_sensors(tick=0))

    controller(_sensors(tick=1))

    assert len(state.buffer) == 1  # one push per tick, same as before n-step existed


def test_n_step_three_holds_transitions_until_the_window_fills() -> None:
    state = _training_state()
    state.n_step = 3
    controller = TrainableController(state=state, training=True)
    controller(_sensors(tick=0))
    controller(_sensors(tick=1))  # 1st raw tick buffered, window not yet full
    assert len(state.buffer) == 0

    controller(_sensors(tick=2))  # 2nd raw tick buffered, window not yet full
    assert len(state.buffer) == 0

    controller(_sensors(tick=3))  # 3rd raw tick completes the window -- one push now

    assert len(state.buffer) == 1


def test_n_step_return_is_the_discounted_sum_of_the_window() -> None:
    state = _training_state()
    state.n_step = 3
    gamma = state.agent.gamma
    controller = TrainableController(state=state, training=True)

    controller(_sensors(tick=0, distance_m=0.0))
    controller(_sensors(tick=1, distance_m=1.0))
    controller(_sensors(tick=2, distance_m=1.0))  # no further progress -- isolates each tick's reward
    controller(_sensors(tick=3, distance_m=1.0))

    transition = state.buffer.sample(1, rng=np.random.default_rng(0))
    # step_reward is deterministic given fixed sensors, so the 3 per-tick rewards are equal;
    # the n-step return must be their discounted sum, and the discount must be gamma**3.
    single_step_reward = transition.rewards[0] / (1 + gamma + gamma**2)
    assert transition.discounts[0] == pytest.approx(gamma**3, abs=1e-5)
    assert single_step_reward > 0  # sanity check the reward isn't trivially zero


def test_n_step_flushes_every_partial_window_immediately_on_termination() -> None:
    state = _training_state()
    state.n_step = 5
    controller = TrainableController(state=state, training=True)
    controller(_sensors(tick=0))
    controller(_sensors(tick=1))  # buffers raw tick [0->1], window not yet full at n_step=5

    controller(_sensors(tick=2, damage=0.95))  # terminal -- flushes both pending windows now

    # two raw ticks were pending ([0->1], [1->2]); termination flushes one n-step
    # transition per remaining window start ([0->1,1->2] and [1->2]), not just one.
    assert len(state.buffer) == 2
    transitions = state.buffer.sample(2, rng=np.random.default_rng(0))
    assert (transitions.dones == 1.0).all()


def test_warmup_actions_are_random_until_buffer_reaches_warmup_steps() -> None:
    state = _training_state(warmup_steps=1_000)
    controller = TrainableController(state=state, training=True)

    for tick in range(5):
        controller(_sensors(tick=tick))

    assert len(state.buffer) < state.warmup_steps
