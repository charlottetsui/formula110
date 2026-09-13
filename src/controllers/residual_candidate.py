"""Local-dev viewer for the residual-RL combined-approach checkpoint (not submission-ready).

The literal combined controller: every tick, `controllers.leaderboard_expert`
(Lucy's hand-written expert, from the separate imitation-learning track)
supplies the base action, and a trained SAC network (Charlotte's RL track)
adds a bounded correction on top -- see `training.controller`'s
`residual_base` mode and docs/rl_design.md section 6 (causal test 37) for
the design this implements. Neither controller alone produces the output;
every command is genuinely both.

Not a packaged leaderboard submission -- it imports `training`, which lives
outside `src/controllers/` and is not included when
`scripts/export_student_controllers.py` zips a submission (mirrors
`sac_candidate.py`'s same caveat).

Defaults to the current best/reference checkpoint
(`2026-09-11_residual-expert-base-v2-seed8000`: 0 eliminations in every
evaluation run to date, ~25-35% faster laps than plain SAC alone). Point at
a different checkpoint (e.g. `2026-09-12_residual-seedsweep-12000`, the
safety-standout alternative that's also the first checkpoint on this track
to beat `leaderboard_expert` outright in a race) with the
``FORMULA110_RESIDUAL_CHECKPOINT`` env var, and override the correction's
scale with ``FORMULA110_RESIDUAL_SCALE`` (default matches
`training.controller.RESIDUAL_ACTION_SCALE`) -- e.g.::

    FORMULA110_RESIDUAL_CHECKPOINT=experiments/2026-09-12_residual-seedsweep-12000/checkpoints/policy_final.pt \\
      uv run racing h2h --watch --challenger-module controllers.residual_candidate \\
      --incumbent-module controllers.leaderboard_expert
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from racing import RobotCommand, RobotSensors
from training.controller import RESIDUAL_ACTION_SCALE, TrainableController, TrainingState
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent

RACING_NAME = "Residual Candidate"
RACING_COLOR = "#F59E0B"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIDDEN_SIZES = (128, 128)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "experiments" / "2026-09-11_residual-expert-base-v2-seed8000" / "checkpoints" / "policy_final.pt"
)


class Controller:
    def __init__(self) -> None:
        checkpoint_env = os.environ.get("FORMULA110_RESIDUAL_CHECKPOINT")
        checkpoint_path = Path(checkpoint_env) if checkpoint_env else DEFAULT_CHECKPOINT
        scale_env = os.environ.get("FORMULA110_RESIDUAL_SCALE")
        residual_scale = float(scale_env) if scale_env else RESIDUAL_ACTION_SCALE

        agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=HIDDEN_SIZES)
        agent.load(checkpoint_path)
        state = TrainingState(
            agent=agent,
            buffer=ReplayBuffer(capacity=1, observation_dim=OBSERVATION_DIM, action_dim=2),
            rng=np.random.default_rng(0),
        )
        # training=False, deterministic=True: a frozen, non-learning evaluation controller,
        # same convention as sac_candidate.py -- this never writes to the buffer or updates
        # weights, and always takes the policy's mean (no exploration noise).
        self._controller = TrainableController(
            state=state, training=False, deterministic=True, residual_base=True, residual_scale=residual_scale
        )

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        return self._controller(sensors)

    def copy_for_car(self) -> Controller:
        return Controller()


def create_controller() -> Controller:
    return Controller()
