"""Submission-ready combined-approach controller (SAC + imitation clone, residual composition).

The literal combined controller, packaged for submission the same way
`race_faster.py` packages the SAC-only checkpoint: self-contained, no
dependency outside `src/controllers/` (the only exception being
`controllers.imitation`, a sibling module within this same package --
`scripts/export_student_controllers.py`'s dependency walker follows and
bundles it automatically, the same way it already does for
`hybrid_controller.py`).

Every tick, `controllers.imitation.Controller` (Lucy's behavioral clone,
trained via imitation learning on `controllers.leaderboard_expert`'s
trajectories -- the finished, trained artifact of the separate
imitation-learning track) supplies the base `(throttle, steer)` action.
On top of it, a trained SAC network adds a bounded correction -- but
*only* on ticks judged a wall- or competitor-proximity hazard by the same
proximity check `training.reward.in_hazard` uses (duplicated inline below,
`_in_hazard`, for this module's own no-training-dependency rule); every
other tick, the clone's command passes through completely unmodified.
"Hazard-gated residual reinforcement learning" -- see
`training.controller`'s `residual_base`/`residual_hazard_gated` modes
(`residual_base_source="clone"`) and docs/rl_design.md section 6 (causal
test 37, and its 2026-09-13 clone-base follow-ups) for the design and the
training-time evidence.

Deliberately built on the clone rather than `leaderboard_expert.Controller`
itself (the hand-written rule-based controller the clone was trained to
imitate, and what every earlier checkpoint on this track used) so the
submitted controller visibly combines this track's own trained model with
the imitation-learning track's own trained model -- two learned artifacts,
not one learned and one hand-coded.

That choice has a real, measured cost, not just an upside -- worth stating
plainly rather than only citing the win. `controllers.imitation.Controller`
run completely alone is already excellent: 0/10 eliminated and a 6/10 win
rate against `leaderboard_expert` itself, an 8.8-8.9s average best lap
(`experiments/2026-09-13_clone-alone-baseline/`). Its one real weakness is
2/10 eliminations against `default_student_controller` specifically. Three
designs were tried against that weakness, in order:

1. A uniform per-tick correction at three scales (0.1/0.3/0.5) --
   `experiments/2026-09-13_residual-clone-base-seed8000/` and its
   `-scale010-`/`-scale050-` siblings. Scale 0.3 fixed the
   `default_student_controller` eliminations (0/10) but cost ~45% of the
   clone's pace everywhere (best lap 8.77s -> 12.77s) and introduced a
   *worse* elimination rate against `leaderboard_expert` (0/10 -> 2/10)
   that the raw clone never had. A strictly worse trade than doing nothing.
2. Hazard-gating that same correction (scale 0.3) so it only fires on
   proximity-hazard ticks -- `experiments/2026-09-13_residual-clone-
   hazard-gated-seed8000/`, this checkpoint. Pareto-better than (1) on
   every single metric tested: `default_student_controller` eliminations
   2/10 -> 1/10 (a real, if partial, fix) at a much smaller pace cost
   (8.77s -> 10.33s, not 12.77s), and the new `leaderboard_expert`
   weakness shrank from 2/10 -> 1/10 eliminated instead of growing.
3. Widening the gated correction further (scale 0.6,
   `experiments/2026-09-13_residual-clone-hazard-gated-scale06-seed8000/`)
   made things worse across the board -- `default_student_controller`
   eliminations rose to 3/10 and a new 1/10 elimination against
   `crash_fast` appeared, which neither the clone alone nor either 0.3
   variant ever had. Larger corrections during a hazard evidently
   destabilize more often than they help; not pursued further.

Honest bottom line: this checkpoint measurably improves the clone's one
identified weakness (`default_student_controller`: 2/10 -> 1/10
eliminated) at a real pace cost (8.77s -> 10.33s best lap) and a smaller
new weakness against `leaderboard_expert` (0/10 -> 1/10 eliminated, and
its lucky 6/10 win rate there drops to 0/10). It is not a strict,
unconditional win over running the clone alone -- it is the best
combined-approach trade found so far between the clone's one flaw and
introducing new ones, not a checkpoint claimed to dominate the clone on
every axis.

Loads the checkpoint from run (2) above
(`experiments/2026-09-13_residual-clone-hazard-gated-seed8000/checkpoints/
policy_final.pt`, policy weights only, trimmed 414KB -> 85KB, dropping
critics/optimizer state, same convention as every prior checkpoint on this
track) -- a from-scratch run at the reference config (seed=8000, races=40,
round_seconds=120, n_step=3, hidden_size=128), not yet seed-swept the way
the expert-base lineage was across causal test 37's many follow-ups.

Always acts deterministically (the policy mean, not a sampled action),
same convention as `race_faster.py` -- this is inference, not training.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from controllers.imitation import create_controller as create_clone_controller
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

# Matches training.reward's WALL_WARNING_DISTANCE_M/WALL_WARNING_BEAM_ANGLES_DEGREES and
# ROBOT_WARNING_DISTANCE_M/ROBOT_WARNING_ANGLE_DEGREES -- the exact hazard definition this
# checkpoint's residual_hazard_gated training used to decide which ticks to correct.
_WALL_WARNING_DISTANCE_M = 6.0
_WALL_WARNING_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-20.0, 0.0, 20.0)
_ROBOT_WARNING_DISTANCE_M = 8.0
_ROBOT_WARNING_ANGLE_DEGREES = 45.0

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


def _proximity_ratio(distance_m: float, *, warning_distance_m: float) -> float:
    if not math.isfinite(distance_m) or distance_m >= warning_distance_m:
        return 0.0
    return (warning_distance_m - max(0.0, distance_m)) / warning_distance_m


def _in_hazard(sensors: RobotSensors) -> bool:
    """Mirrors `training.reward.in_hazard` exactly; duplicated for the no-dependency rule.

    True on a wall- or competitor-proximity hazard tick -- the SAC correction applies only
    then; every other tick, `controllers.imitation.Controller`'s command passes through
    unmodified, matching the `residual_hazard_gated` mode this checkpoint was trained with.
    """
    wall_hazard = any(
        _proximity_ratio(
            sensors.wall_lidar.distance_at_angle_degrees(angle_degrees), warning_distance_m=_WALL_WARNING_DISTANCE_M
        )
        > 0.0
        for angle_degrees in _WALL_WARNING_BEAM_ANGLES_DEGREES
    )
    ahead_competitors = [
        competitor
        for competitor in sensors.camera.competitors
        if abs(competitor.angle_degrees) <= _ROBOT_WARNING_ANGLE_DEGREES
    ]
    robot_hazard = bool(ahead_competitors) and (
        _proximity_ratio(
            min(competitor.distance_m for competitor in ahead_competitors),
            warning_distance_m=_ROBOT_WARNING_DISTANCE_M,
        )
        > 0.0
    )
    return wall_hazard or robot_hazard


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
    """The imitation clone's command, plus a bounded SAC correction on hazard ticks only.

    On an ordinary tick (no wall/competitor proximity hazard, per `_in_hazard`), the clone's
    command passes through completely unmodified -- preserving its already-strong solo pace.
    On a hazard tick, a bounded correction is added on top, same composition as
    `training.controller`'s `residual_hazard_gated` mode this checkpoint was trained under.
    """

    def __init__(self) -> None:
        self._policy = _GaussianPolicyHead(
            observation_dim=_OBSERVATION_DIM, action_dim=_ACTION_DIM, hidden_sizes=_HIDDEN_SIZES
        )
        payload = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self._policy.load_state_dict(payload["policy"])
        self._policy.to("cpu")
        self._policy.eval()
        self._clone = create_clone_controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        # Always called, every tick, regardless of hazard status -- the clone is stateful
        # (an observation-history window), and must see every tick to stay consistent.
        base_command = self._clone(sensors)
        if not _in_hazard(sensors):
            return base_command

        observation = _encode_observation(sensors)
        with torch.inference_mode():
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            correction = self._policy.deterministic_action(observation_tensor).squeeze(0).numpy()

        return RobotCommand(
            throttle=_clamp_unit(base_command.throttle + _RESIDUAL_ACTION_SCALE * float(correction[0])),
            steer=_clamp_unit(base_command.steer + _RESIDUAL_ACTION_SCALE * float(correction[1])),
        )

    def copy_for_car(self) -> Controller:
        # A fresh instance per car/race -- the clone's own previous-command/history state must
        # not be shared across cars sharing a race (mirrors hybrid_controller.py).
        return Controller()


def create_controller() -> Controller:
    return Controller()
