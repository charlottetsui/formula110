"""Soft Actor-Critic: tanh-Gaussian policy, twin critics, auto entropy temperature.

Implemented directly on top of ``torch`` rather than a `gymnasium.Env` +
`stable-baselines3` (docs/rl_design.md section 4): the simulator has no
standard `step()` loop for a single car, so there is no `Env` to hand a
library-provided SAC implementation. Everything here runs on CPU, matching
the controller's inference-time CPU-only requirement (README.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn, optim

from training.replay_buffer import ReplayBatch

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
_LOG_PROB_EPSILON = 1e-6


def _mlp(sizes: tuple[int, ...], *, final_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        is_last_layer = index == len(sizes) - 2
        if not is_last_layer or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy over the ``[-1, 1]`` throttle/steer action."""

    def __init__(self, *, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.trunk = _mlp((observation_dim, *hidden_sizes), final_activation=True)
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(observation)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Return a reparameterized ``(action, log_prob)`` sample, log_prob summed over action dims."""
        mean, log_std = self(observation)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + _LOG_PROB_EPSILON)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic_action(self, observation: Tensor) -> Tensor:
        mean, _log_std = self(observation)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """State-action value network: `Q(observation, action) -> scalar`."""

    def __init__(self, *, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.net = _mlp((observation_dim + action_dim, *hidden_sizes, 1), final_activation=False)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([observation, action], dim=-1))


class SACAgent:
    """Soft Actor-Critic agent: actor, twin critics, twin target critics, auto alpha."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int = 2,
        hidden_sizes: tuple[int, ...] = (128, 128),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        target_entropy: float | None = None,
        seed: int = 0,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cpu")

        torch.manual_seed(seed)
        self.policy = GaussianPolicy(observation_dim=observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.q1 = QNetwork(observation_dim=observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.q2 = QNetwork(observation_dim=observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.q1_target = QNetwork(observation_dim=observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.q2_target = QNetwork(observation_dim=observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for parameter in (*self.q1_target.parameters(), *self.q2_target.parameters()):
            parameter.requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy

        self._actor_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self._critic_optimizer = optim.Adam((*self.q1.parameters(), *self.q2.parameters()), lr=critic_lr)
        self._alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
        """Return a single ``[-1, 1]``-bounded action for one observation."""
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic_action(observation_tensor)
            else:
                action, _log_prob = self.policy.sample(observation_tensor)
        return action.squeeze(0).numpy()

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """Run one SAC gradient step (critics, actor, entropy temperature) on `batch`."""
        observations = torch.as_tensor(batch.observations, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).unsqueeze(-1)
        next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32).unsqueeze(-1)
        discounts = torch.as_tensor(batch.discounts, dtype=torch.float32).unsqueeze(-1)

        critic_loss = self._update_critics(observations, actions, rewards, next_observations, dones, discounts)
        actor_loss, mean_log_prob = self._update_actor(observations)
        alpha_loss = self._update_alpha(mean_log_prob)
        self._soft_update_targets()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": float(self.alpha.item()),
        }

    def _update_critics(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
        discounts: Tensor,
    ) -> float:
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_observations)
            target_q = torch.min(
                self.q1_target(next_observations, next_actions),
                self.q2_target(next_observations, next_actions),
            )
            target_q -= self.alpha * next_log_probs
            # `discounts` is gamma**n per-transition (not the scalar self.gamma), since an
            # n-step transition bootstraps n ticks ahead -- see training.replay_buffer and
            # training.controller's n-step windowing.
            target_value = rewards + (1.0 - dones) * discounts * target_q

        current_q1 = self.q1(observations, actions)
        current_q2 = self.q2(observations, actions)
        loss = nn.functional.mse_loss(current_q1, target_value) + nn.functional.mse_loss(current_q2, target_value)

        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()
        return float(loss.item())

    def _update_actor(self, observations: Tensor) -> tuple[float, Tensor]:
        actions, log_probs = self.policy.sample(observations)
        q_value = torch.min(self.q1(observations, actions), self.q2(observations, actions))
        loss = (self.alpha.detach() * log_probs - q_value).mean()

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return float(loss.item()), log_probs.detach()

    def _update_alpha(self, log_probs: Tensor) -> float:
        loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()

        self._alpha_optimizer.zero_grad()
        loss.backward()
        self._alpha_optimizer.step()
        return float(loss.item())

    def _soft_update_targets(self) -> None:
        for target, online in ((self.q1_target, self.q1), (self.q2_target, self.q2)):
            for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
                target_parameter.mul_(1.0 - self.tau).add_(online_parameter.detach(), alpha=self.tau)

    def save(self, path: str | Path, *, include_training_state: bool = True) -> None:
        """Save policy weights, and (by default) critics/optimizers/alpha to resume training."""
        payload: dict[str, Any] = {"policy": self.policy.state_dict()}
        if include_training_state:
            payload |= {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
            }
        torch.save(payload, path)

    def load_policy_only(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """Load only the actor's weights from a checkpoint, leaving critics/log_alpha at their fresh init.

        Use when resuming into a training distribution different enough from
        the one the checkpoint was trained under (e.g. a fixed, unfamiliar
        opponent) that the checkpoint's own critics/entropy temperature are
        poor priors for it -- a fully-converged actor typically has very low
        `alpha` (little exploration noise) and confident-but-wrong critic
        estimates for states it never saw in training, which `load` would
        carry over verbatim. This keeps the warm-started actor but restores a
        fresh exploration budget (`log_alpha` at its `__init__` default) and
        critics that learn the new distribution's Q-values from scratch
        instead of inheriting stale ones.
        """
        payload = torch.load(path, map_location=map_location, weights_only=True)
        self.policy.load_state_dict(payload["policy"])

    def load(self, path: str | Path, *, map_location: str = "cpu") -> None:
        """Load a checkpoint saved by `save`. Loads whichever keys are present."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        self.policy.load_state_dict(payload["policy"])
        modules = (("q1", self.q1), ("q2", self.q2), ("q1_target", self.q1_target), ("q2_target", self.q2_target))
        for key, module in modules:
            if key in payload:
                module.load_state_dict(payload[key])
        if "log_alpha" in payload:
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"])
