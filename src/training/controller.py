"""The SAC agent, packaged as a `RobotController` for self-play (docs/rl_design.md section 3).

`TrainableController` builds its own observation and reward from each
`RobotSensors` snapshot, pushes transitions into a shared replay buffer, and
periodically runs a SAC gradient step -- all inside `__call__`. Multiple
copies (created by the simulator via `copy_for_car` during a self-play race)
share one `TrainingState`, so they act as parallel data-collection streams
feeding one policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racing.student.api import RobotCommand, RobotSensors
from training.observation import encode_observation
from training.replay_buffer import ReplayBuffer
from training.reward import is_terminal, step_reward
from training.sac import SACAgent

DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_UPDATE_EVERY_N_STEPS = 4
DEFAULT_BATCH_SIZE = 256


@dataclass
class TrainingState:
    """Shared learning state: one buffer and one agent for every controller copy."""

    agent: SACAgent
    buffer: ReplayBuffer
    rng: np.random.Generator
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    update_every_n_steps: int = DEFAULT_UPDATE_EVERY_N_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    global_step: int = 0
    update_metrics: list[dict[str, float]] = field(default_factory=list)

    def record_step(self) -> None:
        self.global_step += 1

    def should_update(self) -> bool:
        return (
            self.global_step % self.update_every_n_steps == 0
            and len(self.buffer) >= self.batch_size
            and len(self.buffer) >= self.warmup_steps
        )

    def maybe_update(self) -> None:
        if not self.should_update():
            return
        batch = self.buffer.sample(self.batch_size, rng=self.rng)
        metrics = self.agent.update(batch)
        metrics["global_step"] = float(self.global_step)
        metrics["buffer_size"] = float(len(self.buffer))
        self.update_metrics.append(metrics)


class TrainableController:
    """`RobotController` that is also the SAC agent's data-collection/training hook.

    Pass ``training=False`` to get a frozen, deterministic evaluation
    controller that shares the trained policy but never writes to the
    replay buffer or updates weights -- used to evaluate a checkpoint
    against a baseline via `run_headless_head_to_head`.
    """

    def __init__(self, *, state: TrainingState, training: bool) -> None:
        self._state = state
        self.training = training
        self._previous_sensors: RobotSensors | None = None
        self._previous_observation: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = encode_observation(sensors)
        if self.training and self._previous_observation is not None and self._previous_action is not None:
            assert self._previous_sensors is not None
            reward = step_reward(self._previous_sensors, sensors)
            self._state.buffer.push(
                self._previous_observation,
                self._previous_action,
                reward,
                observation,
                is_terminal(sensors),
            )
            self._state.record_step()
            self._state.maybe_update()

        action = self._select_action(observation)
        self._previous_sensors = sensors
        self._previous_observation = observation
        self._previous_action = action
        return RobotCommand(throttle=float(action[0]), steer=float(action[1]))

    def _select_action(self, observation: np.ndarray) -> np.ndarray:
        if self.training and len(self._state.buffer) < self._state.warmup_steps:
            return self._state.rng.uniform(-1.0, 1.0, size=self._state.agent.action_dim).astype(np.float32)
        return self._state.agent.act(observation, deterministic=not self.training)

    def copy_for_car(self) -> TrainableController:
        """Return a fresh controller for one car/race, sharing this instance's learning state."""
        return TrainableController(state=self._state, training=self.training)
