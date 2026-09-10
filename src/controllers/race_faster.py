"""Submission-ready SAC controller (Gradescope "improved" module).

Self-contained on purpose: unlike `sac_candidate.py` (a local-dev viewer
that imports `training`, which lives outside `src/controllers/`), this
module inlines the frozen actor network and observation encoding so the
whole thing packages with `scripts/export_student_controllers.py` and runs
with no training-only dependency. It loads the checkpoint from
`docs/rl_design.md` section 6, causal test 32
(`experiments/2026-09-08_seed8000-resumed-short/checkpoints/policy_final.pt`,
policy weights only): `2026-09-08_seed-sweep-v2-8000` (found via best-of-N
genuine-network-initialization seed sampling) fine-tuned for 10 more
self-play races via `--resume-from`. 0/20 eliminations, 20/20 race wins
against both baselines across all 5 fixed evaluation seeds, 6.75 avg
laps, 15.54s avg best lap time, ~30 m/s max speed. Always acts
deterministically (the policy mean, not a sampled action) since this is
inference, not training.

**Prior known risk, resolved by fine-tuning (2026-09-08):** the
checkpoint this one is fine-tuned from had a documented near-miss (1/20
races, 0.5942 damage from a hard corner-entry wall impact, diagnosed in
`experiments/2026-09-08_seed-sweep-v2-8000/notes.md`). Ten additional
self-play races (resumed, not retrained from scratch) reduced the
worst-case damage across all 20 evaluation races to 0.0047 -- a 126x
reduction -- at a small pace cost (14.96s -> 15.54s, +3.9%). This also
strictly beats a separately-found safety-focused alternative
(`2026-09-08_seed-sweep-v2-10000`) on both lap time and damage. Fine-
tuning duration is itself sensitive, not a smooth dial: a longer resume
(+40 races) was tried in parallel and *overshot* this sweet spot,
partially regressing on both safety and pace -- see
`experiments/2026-09-08_seed8000-resumed-short/notes.md` and
`experiments/2026-09-08_seed8000-resumed/notes.md` for the comparison.

Superseded checkpoints, oldest to newest: `2026-09-01_more-training-
seed110` (races=20, the original packaged checkpoint) ->
`2026-09-08_seed-sweep-1000` (races=40, 17.19s) ->
`2026-09-08_seed-sweep-v2-8000` (14.96s, one near-miss) ->
`2026-09-08_seed8000-resumed-short` (this one, +10 races, 15.54s,
near-miss resolved) -- see `docs/lab_notebook.md`'s 2026-09-01
(continued, 13) and 2026-09-08 entries for the full history, including
the network-initialization bug found and fixed on 2026-09-08 that makes
"seed 1000" no longer reproducible as the checkpoint it once was.

The observation encoding below must stay in lockstep with whichever
checkpoint is loaded -- it changed shape (17 -> 24 dims) on 2026-09-07
when `training.observation` added obstacle-LiDAR beams
(`sensors.lidar`, distinct from `wall_lidar`) to detect other cars, which
this checkpoint was trained with. A checkpoint saved before that change
cannot be loaded by this version of the code, and vice versa -- there is
no migration path, only re-exporting from a compatible checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from racing import RobotCommand, RobotSensors

RACING_NAME = "Race Faster"
RACING_COLOR = "#4C8DFF"

_CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "race_faster_policy.pt"
_HIDDEN_SIZES = (128, 128)
_ACTION_DIM = 2
_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0

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
    section 2.1); duplicated here rather than imported so this module has
    no dependency outside `src/controllers/`.
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
    def __init__(self) -> None:
        self._policy = _GaussianPolicyHead(
            observation_dim=_OBSERVATION_DIM, action_dim=_ACTION_DIM, hidden_sizes=_HIDDEN_SIZES
        )
        payload = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self._policy.load_state_dict(payload["policy"])
        self._policy.to("cpu")
        self._policy.eval()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = _encode_observation(sensors)
        with torch.inference_mode():
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            action = self._policy.deterministic_action(observation_tensor).squeeze(0).numpy()
        return RobotCommand(throttle=float(action[0]), steer=float(action[1]))


def create_controller() -> Controller:
    return Controller()
