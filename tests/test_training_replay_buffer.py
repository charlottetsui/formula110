from __future__ import annotations

import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer


def _transition(
    rng: np.random.Generator, *, observation_dim: int, action_dim: int
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    observation = rng.uniform(-1.0, 1.0, size=observation_dim).astype(np.float32)
    action = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
    next_observation = rng.uniform(-1.0, 1.0, size=observation_dim).astype(np.float32)
    return observation, action, float(rng.normal()), next_observation, bool(rng.random() < 0.5)


def _push(buffer: ReplayBuffer, transition: tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
    observation, action, reward, next_observation, done = transition
    buffer.push(observation, action, reward, next_observation, done, discount=0.99)


def test_replay_buffer_starts_empty() -> None:
    buffer = ReplayBuffer(capacity=10, observation_dim=4, action_dim=2)

    assert len(buffer) == 0
    assert buffer.capacity == 10


def test_replay_buffer_grows_until_capacity_then_wraps() -> None:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=5, observation_dim=3, action_dim=2)

    for _ in range(8):
        _push(buffer, _transition(rng, observation_dim=3, action_dim=2))

    assert len(buffer) == 5


def test_replay_buffer_sample_returns_requested_batch_size() -> None:
    rng = np.random.default_rng(1)
    buffer = ReplayBuffer(capacity=20, observation_dim=3, action_dim=2)
    for _ in range(20):
        _push(buffer, _transition(rng, observation_dim=3, action_dim=2))

    batch = buffer.sample(8, rng=rng)

    assert batch.observations.shape == (8, 3)
    assert batch.actions.shape == (8, 2)
    assert batch.rewards.shape == (8,)
    assert batch.next_observations.shape == (8, 3)
    assert batch.dones.shape == (8,)
    assert batch.discounts.shape == (8,)


def test_replay_buffer_sample_rejects_batch_larger_than_size() -> None:
    buffer = ReplayBuffer(capacity=10, observation_dim=3, action_dim=2)
    buffer.push(
        np.zeros(3, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        0.0,
        np.zeros(3, dtype=np.float32),
        False,
        discount=0.99,
    )

    with pytest.raises(ValueError, match="cannot sample"):
        buffer.sample(2, rng=np.random.default_rng(0))


def test_replay_buffer_stores_the_per_transition_discount() -> None:
    buffer = ReplayBuffer(capacity=4, observation_dim=2, action_dim=1)
    zeros_obs, zeros_action = np.zeros(2, dtype=np.float32), np.zeros(1, dtype=np.float32)
    buffer.push(zeros_obs, zeros_action, 1.0, zeros_obs, False, discount=0.5)
    buffer.push(zeros_obs, zeros_action, 2.0, zeros_obs, True, discount=0.9)

    batch = buffer.sample(2, rng=np.random.default_rng(0))

    assert all(value == pytest.approx(0.5) or value == pytest.approx(0.9) for value in batch.discounts.tolist())
