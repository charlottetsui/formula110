"""The SAC agent, packaged as a `RobotController` for self-play (docs/rl_design.md section 3).

`TrainableController` builds its own observation and reward from each
`RobotSensors` snapshot, pushes transitions into a shared replay buffer, and
periodically runs a SAC gradient step -- all inside `__call__`. Multiple
copies (created by the simulator via `copy_for_car` during a self-play race)
share one `TrainingState`, so they act as parallel data-collection streams
feeding one policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from controllers.leaderboard_expert import create_controller as create_expert_controller
from racing.student.api import RobotCommand, RobotSensors
from training.observation import encode_observation
from training.replay_buffer import ReplayBuffer
from training.reward import WALL_PROXIMITY_SPEED_SCALE_MPS, WEIGHT_DAMAGE, WEIGHT_PROGRESS, is_terminal, step_reward
from training.sac import SACAgent
from training.trajectory import WEIGHT_TRAJECTORY_BONUS, BestTrajectoryTracker, bonus_from_snapshot

DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_UPDATE_EVERY_N_STEPS = 4
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_STEP = 1
# Added 2026-09-11 for residual_base mode: bounds how much the policy's per-tick correction
# can shift the expert's command in either action dimension. Chosen as a moderate fraction of
# the full [-1, 1] action range -- large enough to matter, small enough that the expert's
# command still dominates and the policy is learning a genuine correction, not reproducing a
# full independent action. Not yet tuned by a dedicated sweep.
RESIDUAL_ACTION_SCALE = 0.3


@dataclass(frozen=True, slots=True)
class _PendingStep:
    """One raw, single-tick transition awaiting inclusion in an n-step window."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class TrainingState:
    """Shared learning state: one buffer and one agent for every controller copy."""

    agent: SACAgent
    buffer: ReplayBuffer
    rng: np.random.Generator
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    update_every_n_steps: int = DEFAULT_UPDATE_EVERY_N_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    n_step: int = DEFAULT_N_STEP
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

    Pass ``expert_match=True`` to run a private shadow instance of
    `controllers.leaderboard_expert.Controller` alongside training, fed the
    same sensors every tick purely to compute `training.reward`'s
    `WEIGHT_EXPERT_MATCH` bonus -- it never controls the car. Self-play's
    opponent is unaffected; see that weight's docstring for why this is a
    training-distribution-safe way to combine this track with the project's
    separate imitation-learning work.

    Pass ``residual_base=True`` for a structurally different combination:
    the expert's command becomes the base action every tick, and the SAC
    policy only ever learns a bounded *correction* on top of it, scaled by
    ``residual_scale`` (default ``RESIDUAL_ACTION_SCALE``) -- "residual
    reinforcement learning." The
    policy's raw (unscaled) output is what gets pushed to the replay
    buffer and what SAC's Bellman backup is defined over, since that is
    the actual action space the policy controls; the reward reflects the
    real physical consequence of the *blended* command the car received.
    Unlike the inference-time hybrid shield (`controllers.hybrid_controller`,
    a hard switch that was found to trade fewer close calls for more
    catastrophic ones -- see docs/lab_notebook.md's 2026-09-11 entry), the
    expert's influence here is continuous on every tick, so there is no
    discontinuity for the policy to trip over. Mutually exclusive with
    ``expert_match`` (raises ``ValueError`` if both are set) -- the
    expert-match bonus assumes ``previous_action`` is an absolute
    throttle/steer command, which it no longer is once the policy's
    output is a residual instead.

    ``wall_proximity_speed_scale_mps`` and ``damage_weight`` (2026-09-12)
    override ``training.reward``'s constants of the same intent for this
    controller's own reward calculation, both defaulting to the module
    constants -- opt-in, residual-mode-specific "accept more risk on
    purpose" levers, see those constants' docstrings in ``training.reward``.
    """

    def __init__(
        self,
        *,
        state: TrainingState,
        training: bool,
        deterministic: bool | None = None,
        expert_match: bool = False,
        residual_base: bool = False,
        residual_scale: float = RESIDUAL_ACTION_SCALE,
        progress_weight: float = WEIGHT_PROGRESS,
        curvature_aware_center_offset: bool = False,
        wall_proximity_speed_scale_mps: float = WALL_PROXIMITY_SPEED_SCALE_MPS,
        damage_weight: float = WEIGHT_DAMAGE,
    ) -> None:
        if expert_match and residual_base:
            raise ValueError("expert_match and residual_base are mutually exclusive action-composition modes")
        self._state = state
        self.training = training
        self._deterministic = (not training) if deterministic is None else deterministic
        self._previous_sensors: RobotSensors | None = None
        self._previous_observation: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None
        self._trajectory_snapshot: np.ndarray | None = None
        self._n_step_window: deque[_PendingStep] = deque()
        self._expert_match = expert_match
        self._residual_base = residual_base
        self._residual_scale = residual_scale
        self._progress_weight = progress_weight
        self._curvature_aware_center_offset = curvature_aware_center_offset
        self._wall_proximity_speed_scale_mps = wall_proximity_speed_scale_mps
        self._damage_weight = damage_weight
        # A private shadow instance of the expert controller. In expert_match mode it's fed
        # every tick's real sensors purely to compute what it would have done (never used for
        # control) -- see training.reward's WEIGHT_EXPERT_MATCH docstring. In residual_base
        # mode it actually supplies the base action every tick, per this class's own docstring.
        self._shadow_expert = create_expert_controller() if (expert_match or residual_base) else None
        self._previous_expert_action: tuple[float, float] | None = None

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = encode_observation(sensors)
        if self.training and self._state.trajectory is not None and self._trajectory_snapshot is None:
            # Taken once, on this episode's first call -- see training.trajectory's module
            # docstring for why reading a frozen snapshot (not the live, shared tracker) is
            # required to avoid a same-race rival's mid-episode progress corrupting this bonus.
            self._trajectory_snapshot = self._state.trajectory.snapshot()

        if self.training and self._previous_observation is not None and self._previous_action is not None:
            assert self._previous_sensors is not None
            reward = step_reward(
                self._previous_sensors,
                sensors,
                previous_action=(float(self._previous_action[0]), float(self._previous_action[1])),
                expert_action=self._previous_expert_action,
                progress_weight=self._progress_weight,
                curvature_aware_center_offset=self._curvature_aware_center_offset,
                wall_proximity_speed_scale_mps=self._wall_proximity_speed_scale_mps,
                damage_weight=self._damage_weight,
            )
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
            self._push_n_step(
                _PendingStep(
                    observation=self._previous_observation,
                    action=self._previous_action,
                    reward=reward,
                    next_observation=observation,
                    done=is_terminal(sensors),
                )
            )
            self._state.record_step()
            self._state.maybe_update()

        if self._expert_match and self._shadow_expert is not None:
            expert_command = self._shadow_expert(sensors)
            self._previous_expert_action = (expert_command.throttle, expert_command.steer)

        action = self._select_action(observation)
        self._previous_sensors = sensors
        self._previous_observation = observation
        self._previous_action = action

        if self._residual_base:
            assert self._shadow_expert is not None
            # Diagnosed 2026-09-11 (experiments/2026-09-11_residual-expert-base-seed8000/
            # notes.md): a repeated stuck-against-the-same-wall-spot loop, not a single
            # high-speed crash. The expert's stuck-recovery maneuver (a fixed reverse +
            # hard steer-away for a set number of ticks -- see leaderboard_expert.py) is
            # deliberate and precise; adding an independent, untargeted correction on top
            # of it every tick can dilute it just enough that the car never fully clears
            # the obstacle before driving straight back into it. A recovery maneuver can
            # both start and be mid-flight within a single call (the trigger check and the
            # "still recovering" branch both live inside leaderboard_expert.py's __call__),
            # so check the counter on both sides of calling the expert -- either a nonzero
            # value before the call (already recovering) or a jump above zero after it
            # (just triggered this tick) means this command was a recovery command.
            recovery_before = getattr(self._shadow_expert, "_recovery_ticks_remaining", 0)
            base_command = self._shadow_expert(sensors)
            recovery_after = getattr(self._shadow_expert, "_recovery_ticks_remaining", 0)
            if recovery_before > 0 or recovery_after > 0:
                return base_command
            return RobotCommand(
                throttle=_clamp_unit(base_command.throttle + self._residual_scale * float(action[0])),
                steer=_clamp_unit(base_command.steer + self._residual_scale * float(action[1])),
            )
        return RobotCommand(throttle=float(action[0]), steer=float(action[1]))

    def _push_n_step(self, step: _PendingStep) -> None:
        """Buffer one raw tick, pushing a completed n-step transition to the shared buffer.

        Holds up to `self._state.n_step` raw ticks in a per-car sliding window;
        once it reaches that size, the oldest tick's n-step return (discounted
        sum of the window's rewards, bootstrapping `gamma**n` ticks ahead) is
        pushed and the window slides forward by one. If `step.done`, there are
        no further ticks coming to complete the remaining partial windows, so
        every window still pending is flushed immediately instead of waiting
        to fill -- each bootstraps from the terminal tick with `done=True`
        (see docs/rl_design.md section 4 for why this shortens the credit-
        assignment path from a risky action to a delayed crash penalty).
        """
        self._n_step_window.append(step)
        if step.done:
            while self._n_step_window:
                self._emit_n_step_window(tuple(self._n_step_window))
                self._n_step_window.popleft()
            return
        if len(self._n_step_window) >= self._state.n_step:
            self._emit_n_step_window(tuple(self._n_step_window))
            self._n_step_window.popleft()

    def _emit_n_step_window(self, window: tuple[_PendingStep, ...]) -> None:
        gamma = self._state.agent.gamma
        discounted_reward = 0.0
        discount = 1.0
        reached_done = False
        for pending_step in window:
            discounted_reward += discount * pending_step.reward
            discount *= gamma
            if pending_step.done:
                reached_done = True
                break
        first, last = window[0], window[-1]
        self._state.buffer.push(
            first.observation,
            first.action,
            discounted_reward,
            last.next_observation,
            reached_done,
            discount=discount,
        )

    def _select_action(self, observation: np.ndarray) -> np.ndarray:
        if self.training and len(self._state.buffer) < self._state.warmup_steps:
            return self._state.rng.uniform(-1.0, 1.0, size=self._state.agent.action_dim).astype(np.float32)
        return self._state.agent.act(observation, deterministic=self._deterministic)

    def copy_for_car(self) -> TrainableController:
        """Return a fresh controller for one car/race, sharing this instance's learning state."""
        return TrainableController(
            state=self._state,
            training=self.training,
            deterministic=self._deterministic,
            expert_match=self._expert_match,
            residual_base=self._residual_base,
            residual_scale=self._residual_scale,
            progress_weight=self._progress_weight,
            curvature_aware_center_offset=self._curvature_aware_center_offset,
            wall_proximity_speed_scale_mps=self._wall_proximity_speed_scale_mps,
            damage_weight=self._damage_weight,
        )


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))
