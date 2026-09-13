#!/usr/bin/env python3
"""Re-evaluate an existing SAC checkpoint against baseline controllers.

Unlike `train_sac.py`, this does no training -- it loads a saved policy
(from `SACAgent.save`, e.g. `experiments/<run>/checkpoints/policy_final.pt`)
and runs it against every baseline in `training.evaluation.BASELINE_CONTROLLERS`
across a fixed multi-seed evaluation set, exactly like the post-training
evaluation in `train_sac.py`. Use this to compare an already-trained
checkpoint against a baseline it wasn't evaluated against when it was
trained, without paying for a fresh (and possibly non-identical) training
run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from controllers.leaderboard_expert import create_controller as create_expert_controller
from racing.student.api import RobotController
from training.controller import RESIDUAL_ACTION_SCALE
from training.evaluation import BASELINE_CONTROLLERS, evaluate_against_baselines
from training.observation import OBSERVATION_DIM
from training.sac import SACAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (110, 42, 7, 2024, 8675309)
ACTION_DIM = 2


@dataclass(frozen=True)
class EvalSacArguments:
    checkpoint: Path
    hidden_size: int
    eval_seeds: tuple[int, ...]
    eval_races: int
    eval_round_seconds: float
    residual_expert_base: bool
    residual_action_scale: float
    baseline: str
    experiment_dir: Path


def parse_args() -> EvalSacArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a SACAgent.save() checkpoint")
    parser.add_argument("--hidden-size", type=int, default=128, help="must match the checkpoint's architecture")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-races", type=int, default=2, help="head-to-head races per evaluation seed")
    parser.add_argument("--eval-round-seconds", type=float, default=20.0)
    parser.add_argument(
        "--residual-expert-base",
        action="store_true",
        help=(
            "pass this if the checkpoint was trained with train_sac.py's --residual-expert-base -- "
            "its output is a correction on top of controllers.leaderboard_expert, not an absolute "
            "command, and evaluating without this flag would silently misread it as one"
        ),
    )
    parser.add_argument(
        "--residual-action-scale",
        type=float,
        default=RESIDUAL_ACTION_SCALE,
        help="must match the value passed to train_sac.py's --residual-action-scale when this checkpoint was trained",
    )
    parser.add_argument(
        "--baseline",
        choices=("standard", "leaderboard-expert"),
        default="standard",
        help=(
            "'standard' (default): evaluate against training.evaluation.BASELINE_CONTROLLERS "
            "(crash_fast, default_student_controller). 'leaderboard-expert': evaluate against "
            "controllers.leaderboard_expert.Controller as a live opponent instead -- the same "
            "stress test used throughout docs/rl_design.md section 6's causal test 37 chain, "
            "since a checkpoint's own base action already comes from that same expert in "
            "residual mode and this is the matchup that surfaces discontinuity/recovery bugs "
            "the standard baselines don't."
        ),
    )
    parser.add_argument("--experiment-dir", type=Path, required=True, help="output directory under experiments/")
    arguments = parser.parse_args()
    return EvalSacArguments(
        checkpoint=arguments.checkpoint,
        hidden_size=arguments.hidden_size,
        eval_seeds=tuple(arguments.eval_seeds),
        eval_races=arguments.eval_races,
        eval_round_seconds=arguments.eval_round_seconds,
        residual_expert_base=arguments.residual_expert_base,
        residual_action_scale=arguments.residual_action_scale,
        baseline=arguments.baseline,
        experiment_dir=arguments.experiment_dir,
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_sizes=(args.hidden_size,) * 2)
    agent.load(args.checkpoint)

    baselines: dict[str, RobotController] | None = BASELINE_CONTROLLERS
    if args.baseline == "leaderboard-expert":
        baselines = {"leaderboard_expert": create_expert_controller()}
    eval_results = evaluate_against_baselines(
        agent,
        eval_seeds=args.eval_seeds,
        eval_races=args.eval_races,
        eval_round_seconds=args.eval_round_seconds,
        baselines=baselines,
        residual_base=args.residual_expert_base,
        residual_scale=args.residual_action_scale,
    )

    (experiment_dir / "config.yaml").write_text(
        f"checkpoint: {args.checkpoint}\n"
        f"hidden_size: {args.hidden_size}\n"
        f"eval_seeds: {args.eval_seeds}\n"
        f"eval_races: {args.eval_races}\n"
        f"eval_round_seconds: {args.eval_round_seconds}\n"
        f"baseline: {args.baseline}\n"
    )
    (experiment_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))
    print(f"\n[done] evidence written to {experiment_dir}")


if __name__ == "__main__":
    main()
