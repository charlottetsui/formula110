"""Local-dev viewer for the current SAC checkpoint (not submission-ready).

Loads a saved `SACAgent` checkpoint and runs it deterministically, so you
can `uv run racing` / `h2h --watch` whatever the SAC track is currently
training or tuning. This is *not* a packaged leaderboard submission: it
imports `training`, which lives outside `src/controllers/` and is not
included when `scripts/export_student_controllers.py` zips a submission.
Self-contained packaging is deferred to the refinement plan
(docs/rl_design.md section 6).

Point at a specific checkpoint with the ``FORMULA110_SAC_CHECKPOINT`` env
var; defaults to the most recent experiment's checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from racing import RobotCommand, RobotSensors
from training.observation import OBSERVATION_DIM, encode_observation
from training.sac import SACAgent

RACING_NAME = "SAC Candidate"
RACING_COLOR = "#4C8DFF"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "2026-09-01_center-weight-6x" / "checkpoints" / "policy_final.pt"
HIDDEN_SIZES = (128, 128)


class Controller:
    def __init__(self) -> None:
        checkpoint_env = os.environ.get("FORMULA110_SAC_CHECKPOINT")
        checkpoint_path = Path(checkpoint_env) if checkpoint_env else DEFAULT_CHECKPOINT
        self._agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=HIDDEN_SIZES)
        self._agent.load(checkpoint_path)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = encode_observation(sensors)
        action = self._agent.act(observation, deterministic=True)
        return RobotCommand(throttle=float(action[0]), steer=float(action[1]))


def create_controller() -> Controller:
    return Controller()
