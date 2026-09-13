"""Submission-ready combined-approach controller (SAC + expert, residual composition).

The literal combined controller, packaged for submission the same way
`race_faster.py` packages the SAC-only checkpoint: self-contained, no
dependency outside `src/controllers/` (the only exception being
`controllers.leaderboard_expert`, a sibling module within this same
package -- `scripts/export_student_controllers.py`'s dependency walker
follows and bundles it automatically, the same way it already does for
`hybrid_controller.py`).

Every tick, `controllers.leaderboard_expert.Controller` (Lucy's hand-written
expert, from the separate imitation-learning track) supplies the base
`(throttle, steer)` action, and a trained SAC network adds a bounded
correction on top -- "residual reinforcement learning," see
`training.controller`'s `residual_base` mode and docs/rl_design.md section 6
(causal test 37) for the design and the training-time evidence. Neither
half alone produces the output.

Loads the checkpoint from causal test 37's `v2`
(`experiments/2026-09-11_residual-expert-base-v2-seed8000/checkpoints/
policy_final.pt`, policy weights only, trimmed 414KB -> 85KB, dropping
critics/optimizer state): 0/20 eliminations on the standard baselines,
0/10 against `leaderboard_expert` directly, ~11.4s avg lap time (vs.
plain SAC's ~15.5s -- roughly 27% faster) across every evaluation run to
date. A safety-focused alternative checkpoint
(`2026-09-12_residual-seedsweep-12000`, an order of magnitude lower
damage and the first checkpoint on this track to beat the expert
outright in a race, at a small pace cost) exists but is not packaged here
-- see that experiment's notes.md if prioritizing safety margin over pace.

Four further attempts to close the remaining pace gap to the expert's own
raw, safety-unconstrained pace (8.94s solo) -- widening/narrowing the
correction's scale, a 5-seed initialization sweep, reweighting the reward
toward speed, and a cornering-specific reward shape -- all failed cleanly
(docs/rl_design.md section 6, causal test 37 and its follow-ups); this
checkpoint represents the practical best found, not an unfinished search.

Always acts deterministically (the policy mean, not a sampled action),
same convention as `race_faster.py` -- this is inference, not training.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from controllers.leaderboard_expert import create_controller as create_expert_controller
from racing import RobotCommand, RobotSensors

RACING_NAME = "Combined Candidate"
RACING_COLOR = "#22D3AA"

_CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "combined_candidate_policy.pt"
_HIDDEN_SIZES = (128, 128)
_ACTION_DIM = 2
_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
# Matches training.controller.RESIDUAL_ACTION_SCALE, the value this checkpoint was
# trained with -- must stay in lockstep with the checkpoint, not independently tunable.
_RESIDUAL_ACTION_SCALE = 0.3

_MAX_SPEED_MPS = 20.0
_WALL_LIDAR_CAP_M = 20.0
_OBSTACLE_LIDAR_CAP_M = 20.0
_CENTER_OFFSET_CAP_M = 5.0
_LOOKAHEAD_OFFSET_CAP_M = 5.0
_YAW_RATE_CAP_DEGREES_PER_S = 180.0
_WALL_LIDAR_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
_OBSTACLE_LIDAR_BEAM_ANGLES_DEGREES: tuple[float, ...] = _WALL_LIDAR_BEAM_ANGLES_DEGREES
_LOOKAHEAD_COUNT = 3

_OBSERVATION_DIM = (
    1  # signed speed
    + 1  # heading error
    + 1  # center offset
    + _LOOKAHEAD_COUNT  # lookahead offsets
    + len(_WALL_LIDAR_BEAM_ANGLES_DEGREES)  # wall lidar beams
    + len(_OBSTACLE_LIDAR_BEAM_ANGLES_DEGREES)  # nearby-obstacle (wall/robot) lidar beams
    + 1  # yaw rate
    + 1  # wall contact flag
    + 1  # robot contact flag
    + 1  # damage
)


def _encode_observation(sensors: RobotSensors) -> np.ndarray:
    """Sensor snapshot -> fixed-size, roughly [-1, 1]-scaled observation vector.

    Mirrors `training.observation.encode_observation` (docs/rl_design.md
    section 2.1) and `race_faster.py`'s own copy of it; duplicated here
    rather than imported so this module has no dependency outside
    `src/controllers/`.
    """
    wall_beams = tuple(
        _scale_distance(sensors.wall_lidar.distance_at_angle_degrees(angle_degrees), cap=_WALL_LIDAR_CAP_M)
        for angle_degrees in _WALL_LIDAR_BEAM_ANGLES_DEGREES
    )
    obstacle_beams = tuple(
        _scale_distance(sensors.lidar.distance_at_angle_degrees(angle_degrees), cap=_OBSTACLE_LIDAR_CAP_M)
        for angle_degrees in _OBSTACLE_LIDAR_BEAM_ANGLES_DEGREES
    )
    lookahead = _padded_lookahead(sensors.camera.lookahead_offsets_m)
    features = (
        _clip_ratio(sensors.odometry.speed_mps, _MAX_SPEED_MPS),
        _clip_ratio(sensors.camera.heading_error_degrees, 180.0),
        _clip_ratio(sensors.camera.center_offset_m, _CENTER_OFFSET_CAP_M),
        *(_clip_ratio(offset_m, _LOOKAHEAD_OFFSET_CAP_M) for offset_m in lookahead),
        *wall_beams,
        *obstacle_beams,
        _clip_ratio(sensors.imu.yaw_rate_degrees_per_s, _YAW_RATE_CAP_DEGREES_PER_S),
        1.0 if sensors.contact.wall > 0.0 else 0.0,
        1.0 if sensors.contact.robot > 0.0 else 0.0,
        sensors.contact.damage,
    )
    observation = np.asarray(features, dtype=np.float32)
    if observation.shape != (_OBSERVATION_DIM,):
        raise ValueError(f"encoded observation has shape {observation.shape}, expected ({_OBSERVATION_DIM},)")
    return observation


def _padded_lookahead(offsets_m: tuple[float, ...]) -> tuple[float, ...]:
    if len(offsets_m) >= _LOOKAHEAD_COUNT:
        return offsets_m[:_LOOKAHEAD_COUNT]
    return offsets_m + (0.0,) * (_LOOKAHEAD_COUNT - len(offsets_m))


def _clip_ratio(value: float, cap: float) -> float:
    if not math.isfinite(value):
        value = math.copysign(cap, value) if value != 0.0 else 0.0
    return max(-1.0, min(1.0, value / cap))


def _scale_distance(distance_m: float, *, cap: float) -> float:
    if not math.isfinite(distance_m):
        return 1.0
    return max(0.0, min(1.0, distance_m / cap))


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


class _GaussianPolicyHead(nn.Module):
    """Inference-only tanh-Gaussian policy: mirrors `training.sac.GaussianPolicy`.

    Only the pieces needed to compute the deterministic action are kept;
    the critics, sampling, and optimizer state used during training are
    dropped along with the `training` import.
    """

    def __init__(self, *, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = (observation_dim, *hidden_sizes)
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(observation)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), _LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def deterministic_action(self, observation: Tensor) -> Tensor:
        mean, _log_std = self(observation)
        return torch.tanh(mean)


class Controller:
    """Every tick: `leaderboard_expert`'s command, plus a bounded SAC correction on top.

    During the expert's own stuck-recovery maneuver (a fixed reverse + hard
    steer for a set number of ticks), the correction is suppressed entirely
    and the expert's command passes through unmodified -- diluting a
    deliberate escape maneuver with an unrelated correction was diagnosed as
    a real crash cause during training (docs/rl_design.md section 6, causal
    test 37) and fixed there; this packaged version must replicate that
    exact composition, since the checkpoint's weights were trained under it.
    """

    def __init__(self) -> None:
        self._policy = _GaussianPolicyHead(
            observation_dim=_OBSERVATION_DIM, action_dim=_ACTION_DIM, hidden_sizes=_HIDDEN_SIZES
        )
        payload = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self._policy.load_state_dict(payload["policy"])
        self._policy.to("cpu")
        self._policy.eval()
        self._expert = create_expert_controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = _encode_observation(sensors)
        with torch.inference_mode():
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            correction = self._policy.deterministic_action(observation_tensor).squeeze(0).numpy()

        recovery_before = getattr(self._expert, "_recovery_ticks_remaining", 0)
        base_command = self._expert(sensors)
        recovery_after = getattr(self._expert, "_recovery_ticks_remaining", 0)
        if recovery_before > 0 or recovery_after > 0:
            return base_command

        return RobotCommand(
            throttle=_clamp_unit(base_command.throttle + _RESIDUAL_ACTION_SCALE * float(correction[0])),
            steer=_clamp_unit(base_command.steer + _RESIDUAL_ACTION_SCALE * float(correction[1])),
        )

    def copy_for_car(self) -> Controller:
        # A fresh instance per car/race -- the expert's recovery timer and previous-steer
        # state must not be shared across cars sharing a race (mirrors hybrid_controller.py).
        return Controller()


def create_controller() -> Controller:
    return Controller()
