from __future__ import annotations

import numpy as np

from racing.student.api import CameraSensors, ContactSensors, OdometrySensors, RobotSensors
from training.controller import TrainableController, TrainingState
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent


def _training_state(*, warmup_steps: int = 5, update_every_n_steps: int = 2, batch_size: int = 4) -> TrainingState:
    return TrainingState(
        agent=SACAgent(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=(8, 8), seed=0),
        buffer=ReplayBuffer(capacity=1_000, observation_dim=OBSERVATION_DIM, action_dim=2),
        rng=np.random.default_rng(0),
        warmup_steps=warmup_steps,
        update_every_n_steps=update_every_n_steps,
        batch_size=batch_size,
    )


def _sensors(*, tick: int, damage: float = 0.0) -> RobotSensors:
    return RobotSensors(
        dt_s=1 / 60,
        tick=tick,
        odometry=OdometrySensors(speed_mps=2.0),
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


def test_warmup_actions_are_random_until_buffer_reaches_warmup_steps() -> None:
    state = _training_state(warmup_steps=1_000)
    controller = TrainableController(state=state, training=True)

    for tick in range(5):
        controller(_sensors(tick=tick))

    assert len(state.buffer) < state.warmup_steps
