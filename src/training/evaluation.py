"""Evaluate a (frozen) SAC policy against fixed baseline controllers.

Shared by ``scripts/train_sac.py`` (evaluate right after training) and
``scripts/eval_sac.py`` (re-evaluate an existing checkpoint, e.g. against a
baseline it wasn't evaluated against originally). Every run reports the
simulator's own public race stats (`HeadToHeadResult`), not the training
proxy reward -- see docs/rl_design.md section 2.3.

Two baselines are evaluated by default: `crash_fast` (a near-vacuous floor
-- full throttle, no steering, crashes almost immediately) and
`default_student_controller` (a real center-line-following baseline). A
controller that only beats `crash_fast` has not demonstrated much; see
docs/lab_notebook.md's 2026-09-01 entry for why the second baseline was
added.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from controllers.crash_fast import control as crash_fast_control
from racing.race.head_to_head import HeadToHeadResult, HeadToHeadRole, run_headless_head_to_head
from racing.student.api import RobotController, default_student_controller
from training.controller import TrainableController, TrainingState
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent

BASELINE_CONTROLLERS: dict[str, RobotController] = {
    "crash_fast": crash_fast_control,
    "default_student_controller": default_student_controller,
}


def role_distance_m(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    """Sum one side's scored distance across every race in a `HeadToHeadResult`."""
    return sum(
        (race.challenger if role == "challenger" else race.incumbent).team_sum_distance_m for race in result.races
    )


def evaluate_against_baselines(
    agent: SACAgent,
    *,
    eval_seeds: Iterable[int],
    eval_races: int,
    eval_round_seconds: float,
    baselines: dict[str, RobotController] | None = None,
) -> list[dict[str, object]]:
    """Run the frozen policy in `agent` against every baseline, across every seed."""
    resolved_baselines = BASELINE_CONTROLLERS if baselines is None else baselines
    # `rng`, `warmup_steps`, etc. are unused here: every controller built from this
    # state is `training=False`, which always takes the deterministic-policy path.
    eval_state = TrainingState(
        agent=agent,
        buffer=ReplayBuffer(capacity=1, observation_dim=OBSERVATION_DIM, action_dim=2),
        rng=np.random.default_rng(0),
    )
    records: list[dict[str, object]] = []
    for baseline_name, baseline_controller in resolved_baselines.items():
        for seed in eval_seeds:
            trained_controller = TrainableController(state=eval_state, training=False)
            result = run_headless_head_to_head(
                challenger_controller=trained_controller,
                incumbent_controller=baseline_controller,
                challenger_name="sac-trained",
                incumbent_name=baseline_name,
                race_count=eval_races,
                round_seconds=eval_round_seconds,
                random_seed=seed,
            )
            record = result.to_dict()
            record["eval_seed"] = seed
            record["baseline"] = baseline_name
            records.append(record)
            print(
                f"[eval] baseline={baseline_name} seed={seed}: "
                f"sac-trained {role_distance_m(result, 'challenger'):.1f}m vs "
                f"{baseline_name} {role_distance_m(result, 'incumbent'):.1f}m "
                f"(sac wins {result.challenger_wins}/{result.race_count})"
            )
    return records
