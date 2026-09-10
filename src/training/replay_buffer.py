"""Fixed-capacity replay buffer shared across self-play controller copies.

Every ``TrainableController`` copy created via ``copy_for_car`` during a
self-play race pushes into the same buffer instance, so experience from all
cars in all races accumulates into one pool (docs/rl_design.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """One sampled minibatch, ready for a SAC update step."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray


class ReplayBuffer:
    """Circular, preallocated replay buffer for continuous observations/actions.

    Each transition carries its own bootstrap ``discount`` (rather than a
    single scalar gamma shared by every sample) so that n-step returns work:
    a transition spanning n real ticks bootstraps with `gamma**n`, and a
    transition truncated early by episode termination (fewer than n ticks
    available) bootstraps with `gamma**actual_n` instead -- see
    `training.controller`'s n-step windowing for how these are produced.
    """

    def __init__(self, *, capacity: int, observation_dim: int, action_dim: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        self._capacity = capacity
        self._observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._discounts = np.zeros((capacity,), dtype=np.float32)
        self._write_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        *,
        discount: float,
    ) -> None:
        index = self._write_index
        self._observations[index] = observation
        self._actions[index] = action
        self._rewards[index] = reward
        self._next_observations[index] = next_observation
        self._dones[index] = 1.0 if done else 0.0
        self._discounts[index] = discount
        self._write_index = (index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, *, rng: np.random.Generator) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if self._size < batch_size:
            raise ValueError(f"buffer has {self._size} transitions, cannot sample {batch_size}")
        indices = rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            dones=self._dones[indices],
            discounts=self._discounts[indices],
        )
