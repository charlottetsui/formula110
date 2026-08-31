from __future__ import annotations

from pathlib import Path

import numpy as np

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
