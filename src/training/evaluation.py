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
from training.controller import DEFAULT_RESIDUAL_BASE_SOURCE, RESIDUAL_ACTION_SCALE, TrainableController, TrainingState
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
    deterministic: bool = True,
    residual_base: bool = False,
    residual_scale: float = RESIDUAL_ACTION_SCALE,
    residual_base_source: str = DEFAULT_RESIDUAL_BASE_SOURCE,
    residual_hazard_gated: bool = False,
) -> list[dict[str, object]]:
    """Run the frozen policy in `agent` against every baseline, across every seed.

    ``deterministic=True`` (the default) picks the policy's mean action
    every tick, matching how a packaged controller would run it.
    ``deterministic=False`` samples actions the same way training does
    (with exploration noise), while still never writing to a replay buffer
    or updating weights -- used to check whether dangerous deterministic
    behavior is an eval-time artifact rather than what the policy actually
    learned.

    Pass ``residual_base=True`` if `agent` was trained with
    `TrainableController`'s ``residual_base`` mode -- its output is a
    correction on top of a base controller, not an absolute command, and
    evaluating it without this flag would silently treat that correction
    as the whole action. Pass ``residual_base_source`` to match whichever
    base ("expert" or "clone") the checkpoint was actually trained against.
    """
    resolved_baselines = BASELINE_CONTROLLERS if baselines is None else baselines
    # `rng` is only consulted by the stochastic (deterministic=False) path here
    # (via SACAgent.act's own torch sampling, not this rng directly); `warmup_steps`
    # is unused since every controller built from this state has `training=False`.
    eval_state = TrainingState(
        agent=agent,
        buffer=ReplayBuffer(capacity=1, observation_dim=OBSERVATION_DIM, action_dim=2),
        rng=np.random.default_rng(0),
    )
    records: list[dict[str, object]] = []
    for baseline_name, baseline_controller in resolved_baselines.items():
        for seed in eval_seeds:
            trained_controller = TrainableController(
                state=eval_state,
                training=False,
                deterministic=deterministic,
                residual_base=residual_base,
                residual_scale=residual_scale,
                residual_base_source=residual_base_source,
                residual_hazard_gated=residual_hazard_gated,
            )
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
