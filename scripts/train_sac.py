#!/usr/bin/env python3
"""Run a self-play SAC training run and evaluate it against baseline controllers.

Implements the minimum experiment from docs/rl_design.md section 5: train
via self-play (`run_headless_head_to_head` with both sides pointing at the
same in-training policy, per section 3), then evaluate the frozen policy
against every baseline in `training.evaluation.BASELINE_CONTROLLERS`
across a fixed multi-seed evaluation set. All evidence is written under
`experiments/<slug>/` per `experiments/README.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from racing.race.head_to_head import run_headless_head_to_head
from training.controller import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_UPDATE_EVERY_N_STEPS,
    DEFAULT_WARMUP_STEPS,
    TrainableController,
    TrainingState,
)
from training.evaluation import evaluate_against_baselines, role_distance_m
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.sac import SACAgent
from training.trajectory import BestTrajectoryTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (110, 42, 7, 2024, 8675309)
ACTION_DIM = 2


@dataclass(frozen=True)
class TrainSacArguments:
    races: int
    round_seconds: float
    copies_per_side: int
    seed: int
    buffer_capacity: int
    warmup_steps: int
    update_every_n_steps: int
    batch_size: int
    hidden_size: int
    trajectory_bonus: bool
    eval_seeds: tuple[int, ...]
    eval_races: int
    eval_round_seconds: float
    experiment_dir: Path


def parse_args() -> TrainSacArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=int, default=6, help="self-play races to train over")
    parser.add_argument("--round-seconds", type=float, default=15.0, help="seconds per self-play race")
    parser.add_argument("--copies-per-side", type=int, default=1, help="cars per side during self-play")
    parser.add_argument("--seed", type=int, default=110, help="random seed for self-play race spawns")
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--update-every-n-steps", type=int, default=DEFAULT_UPDATE_EVERY_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--trajectory-bonus",
        action="store_true",
        help="reward relative to the best-known distance-at-tick seen so far this run (training.trajectory)",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-races", type=int, default=2, help="head-to-head races per evaluation seed")
    parser.add_argument("--eval-round-seconds", type=float, default=20.0)
    parser.add_argument("--experiment-dir", type=Path, required=True, help="output directory under experiments/")
    arguments = parser.parse_args()
    return TrainSacArguments(
        races=arguments.races,
        round_seconds=arguments.round_seconds,
        copies_per_side=arguments.copies_per_side,
        seed=arguments.seed,
        buffer_capacity=arguments.buffer_capacity,
        warmup_steps=arguments.warmup_steps,
        update_every_n_steps=arguments.update_every_n_steps,
        batch_size=arguments.batch_size,
        hidden_size=arguments.hidden_size,
        trajectory_bonus=arguments.trajectory_bonus,
        eval_seeds=tuple(arguments.eval_seeds),
        eval_races=arguments.eval_races,
        eval_round_seconds=arguments.eval_round_seconds,
        experiment_dir=arguments.experiment_dir,
    )


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def train(args: TrainSacArguments) -> tuple[SACAgent, TrainingState, float]:
    agent = SACAgent(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(args.hidden_size, args.hidden_size),
    )
    buffer = ReplayBuffer(capacity=args.buffer_capacity, observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)
    trajectory = BestTrajectoryTracker(max_ticks=int(args.round_seconds * 60) + 1) if args.trajectory_bonus else None
    state = TrainingState(
        agent=agent,
        buffer=buffer,
        rng=np.random.default_rng(args.seed),
        warmup_steps=args.warmup_steps,
        update_every_n_steps=args.update_every_n_steps,
        batch_size=args.batch_size,
        trajectory=trajectory,
    )
    challenger = TrainableController(state=state, training=True)
    incumbent = TrainableController(state=state, training=True)

    started_at = time.monotonic()
    result = run_headless_head_to_head(
        challenger_controller=challenger,
        incumbent_controller=incumbent,
        challenger_name="sac-in-training-a",
        incumbent_name="sac-in-training-b",
        race_count=args.races,
        round_seconds=args.round_seconds,
        random_seed=args.seed,
        copies_per_side=args.copies_per_side,
    )
    training_seconds = time.monotonic() - started_at

    print(f"[train] self-play finished in {training_seconds:.1f}s")
    print(f"[train] transitions collected: {len(buffer)}, gradient updates: {len(state.update_metrics)}")
    print(
        f"[train] self-play scored distance a={role_distance_m(result, 'challenger'):.1f}m "
        f"b={role_distance_m(result, 'incumbent'):.1f}m over {args.races} race(s)"
    )
    return agent, state, training_seconds


def write_config(
    path: Path, args: TrainSacArguments, *, training_seconds: float, transitions: int, updates: int
) -> None:
    payload = {
        **asdict(args),
        "experiment_dir": str(args.experiment_dir),
        "git_commit": git_commit_hash(),
        "training_seconds": round(training_seconds, 2),
        "transitions_collected": transitions,
        "gradient_updates": updates,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
    }
    with path.open("w") as config_file:
        for key, value in payload.items():
            config_file.write(f"{key}: {value}\n")


def write_metrics_csv(path: Path, update_metrics: list[dict[str, float]]) -> None:
    if not update_metrics:
        path.write_text("global_step,buffer_size,critic_loss,actor_loss,alpha_loss,alpha\n")
        return
    fieldnames = list(update_metrics[0].keys())
    with path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(update_metrics)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    agent, state, training_seconds = train(args)
    eval_results = evaluate_against_baselines(
        agent, eval_seeds=args.eval_seeds, eval_races=args.eval_races, eval_round_seconds=args.eval_round_seconds
    )

    agent.save(checkpoints_dir / "policy_final.pt", include_training_state=True)
    write_config(
        experiment_dir / "config.yaml",
        args,
        training_seconds=training_seconds,
        transitions=len(state.buffer),
        updates=len(state.update_metrics),
    )
    write_metrics_csv(experiment_dir / "metrics.csv", state.update_metrics)
    (experiment_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))

    print(f"\n[done] evidence written to {experiment_dir}")


if __name__ == "__main__":
    main()
