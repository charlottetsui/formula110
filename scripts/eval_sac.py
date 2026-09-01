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

from training.evaluation import evaluate_against_baselines
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
    experiment_dir: Path


def parse_args() -> EvalSacArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a SACAgent.save() checkpoint")
    parser.add_argument("--hidden-size", type=int, default=128, help="must match the checkpoint's architecture")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-races", type=int, default=2, help="head-to-head races per evaluation seed")
    parser.add_argument("--eval-round-seconds", type=float, default=20.0)
    parser.add_argument("--experiment-dir", type=Path, required=True, help="output directory under experiments/")
    arguments = parser.parse_args()
    return EvalSacArguments(
        checkpoint=arguments.checkpoint,
        hidden_size=arguments.hidden_size,
        eval_seeds=tuple(arguments.eval_seeds),
        eval_races=arguments.eval_races,
        eval_round_seconds=arguments.eval_round_seconds,
        experiment_dir=arguments.experiment_dir,
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_sizes=(args.hidden_size,) * 2)
    agent.load(args.checkpoint)

    eval_results = evaluate_against_baselines(
        agent, eval_seeds=args.eval_seeds, eval_races=args.eval_races, eval_round_seconds=args.eval_round_seconds
    )

    (experiment_dir / "config.yaml").write_text(
        f"checkpoint: {args.checkpoint}\n"
        f"hidden_size: {args.hidden_size}\n"
        f"eval_seeds: {args.eval_seeds}\n"
        f"eval_races: {args.eval_races}\n"
        f"eval_round_seconds: {args.eval_round_seconds}\n"
    )
    (experiment_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))
    print(f"\n[done] evidence written to {experiment_dir}")


if __name__ == "__main__":
    main()
