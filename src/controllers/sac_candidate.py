"""Local-dev viewer for the current SAC checkpoint (not submission-ready).

Loads a saved `SACAgent` checkpoint and runs it deterministically, so you
can `uv run racing` / `h2h --watch` whatever the SAC track is currently
training or tuning. This is *not* a packaged leaderboard submission: it
imports `training`, which lives outside `src/controllers/` and is not
included when `scripts/export_student_controllers.py` zips a submission.
Self-contained packaging is deferred to the refinement plan
(docs/rl_design.md section 6).

With no override, this picks whichever `experiments/*/checkpoints/policy_final.pt`
was written most recently -- so a fresh training run becomes "the"
controller the next time this loads, with no file to remember to edit.
Point at a specific checkpoint instead with the
``FORMULA110_SAC_CHECKPOINT`` env var (e.g. to compare an older run, or
while a newer training run is in progress and you don't want to switch
yet).
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
HIDDEN_SIZES = (128, 128)


def _latest_checkpoint() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "experiments").glob("*/checkpoints/policy_final.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "no experiments/*/checkpoints/policy_final.pt found -- run scripts/train_sac.py first, "
            "or set FORMULA110_SAC_CHECKPOINT to a specific checkpoint path"
        )
    return candidates[-1]


class Controller:
    def __init__(self) -> None:
        checkpoint_env = os.environ.get("FORMULA110_SAC_CHECKPOINT")
        checkpoint_path = Path(checkpoint_env) if checkpoint_env else _latest_checkpoint()
        self._agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=HIDDEN_SIZES)
        self._agent.load(checkpoint_path)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = encode_observation(sensors)
        action = self._agent.act(observation, deterministic=True)
        return RobotCommand(throttle=float(action[0]), steer=float(action[1]))


def create_controller() -> Controller:
    return Controller()
