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
from training.trajectory import WEIGHT_TRAJECTORY_BONUS, BestTrajectoryTracker, bonus_from_snapshot

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
    trajectory: BestTrajectoryTracker | None = None
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

    Pass ``training=False`` to get a frozen evaluation controller that
    shares the trained policy but never writes to the replay buffer or
    updates weights -- used to evaluate a checkpoint against a baseline via
    `run_headless_head_to_head`. By default this also selects the
    deterministic (mean, no exploration noise) action, matching how a
    packaged controller like `controllers.sac_candidate` would run it.

    Pass ``deterministic`` explicitly to decouple action selection from
    ``training`` -- e.g. ``training=False, deterministic=False`` evaluates
    a frozen checkpoint using *stochastic* (sampled) actions, the same way
    actions were chosen during training, without writing to the buffer or
    updating weights. Used to check whether a policy's dangerous
    deterministic behavior is an eval-time artifact (see
    docs/lab_notebook.md's 2026-09-01 entries on the seed-909 causal-test
    chain) or genuinely what the policy learned.
    """

    def __init__(self, *, state: TrainingState, training: bool, deterministic: bool | None = None) -> None:
        self._state = state
        self.training = training
        self._deterministic = (not training) if deterministic is None else deterministic
        self._previous_sensors: RobotSensors | None = None
        self._previous_observation: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None
        self._trajectory_snapshot: np.ndarray | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = encode_observation(sensors)
        if self.training and self._state.trajectory is not None and self._trajectory_snapshot is None:
            # Taken once, on this episode's first call -- see training.trajectory's module
            # docstring for why reading a frozen snapshot (not the live, shared tracker) is
            # required to avoid a same-race rival's mid-episode progress corrupting this bonus.
            self._trajectory_snapshot = self._state.trajectory.snapshot()

        if self.training and self._previous_observation is not None and self._previous_action is not None:
            assert self._previous_sensors is not None
            reward = step_reward(self._previous_sensors, sensors)
            if self._state.trajectory is not None:
                assert self._trajectory_snapshot is not None
                reward += WEIGHT_TRAJECTORY_BONUS * bonus_from_snapshot(
                    self._trajectory_snapshot,
                    previous_tick=self._previous_sensors.tick,
                    previous_distance_m=self._previous_sensors.odometry.distance_m,
                    current_tick=sensors.tick,
                    current_distance_m=sensors.odometry.distance_m,
                )
                self._state.trajectory.update(tick=sensors.tick, distance_m=sensors.odometry.distance_m)
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
        return self._state.agent.act(observation, deterministic=self._deterministic)

    def copy_for_car(self) -> TrainableController:
        """Return a fresh controller for one car/race, sharing this instance's learning state."""
        return TrainableController(state=self._state, training=self.training, deterministic=self._deterministic)
