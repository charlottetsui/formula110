from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent


def _agent() -> SACAgent:
    return SACAgent(observation_dim=6, action_dim=2, hidden_sizes=(8, 8), seed=0)


def _filled_buffer(*, size: int = 32) -> ReplayBuffer:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=size, observation_dim=6, action_dim=2)
    for _ in range(size):
        buffer.push(
            rng.uniform(-1.0, 1.0, size=6).astype(np.float32),
            rng.uniform(-1.0, 1.0, size=2).astype(np.float32),
            float(rng.normal()),
            rng.uniform(-1.0, 1.0, size=6).astype(np.float32),
            bool(rng.random() < 0.2),
            discount=0.99,
        )
    return buffer


def test_act_returns_bounded_action_of_the_right_shape() -> None:
    agent = _agent()
    observation = np.zeros(6, dtype=np.float32)

    action = agent.act(observation, deterministic=False)

    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)


def test_deterministic_act_is_repeatable_for_the_same_observation() -> None:
    agent = _agent()
    observation = np.full(6, 0.3, dtype=np.float32)

    first = agent.act(observation, deterministic=True)
    second = agent.act(observation, deterministic=True)

    assert np.allclose(first, second)


def test_update_returns_finite_losses_and_changes_policy_weights() -> None:
    agent = _agent()
    buffer = _filled_buffer()
    rng = np.random.default_rng(1)
    before = next(agent.policy.parameters()).detach().clone()

    metrics = agent.update(buffer.sample(16, rng=rng))

    assert all(np.isfinite(value) for value in metrics.values())
    after = next(agent.policy.parameters()).detach().clone()
    assert not np.allclose(before.numpy(), after.numpy())


def test_critic_update_uses_the_per_transition_discount_not_a_shared_scalar() -> None:
    """Two agents with identical initial weights, updated on transitions that differ
    only in `discount`, must produce different critic losses -- if the critic target
    ignored `batch.discounts` and used a fixed scalar gamma instead (the pre-n-step
    behavior), these would be identical."""
    rng = np.random.default_rng(0)
    observation = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
    action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    next_observation = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)

    buffer_zero_discount = ReplayBuffer(capacity=1, observation_dim=6, action_dim=2)
    buffer_zero_discount.push(observation, action, 1.0, next_observation, False, discount=0.0)
    buffer_high_discount = ReplayBuffer(capacity=1, observation_dim=6, action_dim=2)
    buffer_high_discount.push(observation, action, 1.0, next_observation, False, discount=0.99)

    agent_zero_discount = SACAgent(observation_dim=6, action_dim=2, hidden_sizes=(8, 8), seed=42)
    agent_high_discount = SACAgent(observation_dim=6, action_dim=2, hidden_sizes=(8, 8), seed=42)

    metrics_zero_discount = agent_zero_discount.update(buffer_zero_discount.sample(1, rng=np.random.default_rng(0)))
    metrics_high_discount = agent_high_discount.update(buffer_high_discount.sample(1, rng=np.random.default_rng(0)))

    assert metrics_zero_discount["critic_loss"] != pytest.approx(metrics_high_discount["critic_loss"])


def test_save_and_load_round_trips_policy_weights(tmp_path: Path) -> None:
    agent = _agent()
    buffer = _filled_buffer()
    rng = np.random.default_rng(2)
    agent.update(buffer.sample(16, rng=rng))  # move weights away from their initial values
    checkpoint_path = tmp_path / "policy.pt"
    agent.save(checkpoint_path)

    reloaded = _agent()
    reloaded.load(checkpoint_path)

    observation = np.full(6, -0.2, dtype=np.float32)
    assert np.allclose(
        agent.act(observation, deterministic=True),
        reloaded.act(observation, deterministic=True),
    )
