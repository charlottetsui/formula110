#!/usr/bin/env python3
"""Compare a checkpoint's deterministic vs. stochastic evaluation behavior.

Diagnostic for the train/eval mismatch hypothesis raised in
docs/lab_notebook.md's 2026-09-01 entries: does a checkpoint's dangerous
*deterministic* (mean-action) behavior differ from its *stochastic*
(sampled, like training) behavior at the same states? If deterministic
racing is much more extreme/dangerous than stochastic racing with the same
weights, that points to an eval-time artifact rather than something
reward tuning alone can fix. If both look similarly extreme, the policy
genuinely learned that behavior.

Runs `training.evaluation.evaluate_against_baselines` twice against the
same checkpoint -- once with `deterministic=True`, once with
`deterministic=False` -- and writes both result sets plus a summary
comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from training.evaluation import BASELINE_CONTROLLERS, evaluate_against_baselines
from training.observation import OBSERVATION_DIM
from training.sac import SACAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (110, 42, 7, 2024, 8675309)
ACTION_DIM = 2


@dataclass(frozen=True)
class CompareArguments:
    checkpoint: Path
    hidden_size: int
    baseline: str
    eval_seeds: tuple[int, ...]
    eval_races: int
    eval_round_seconds: float
    experiment_dir: Path


def parse_args() -> CompareArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a SACAgent.save() checkpoint")
    parser.add_argument("--hidden-size", type=int, default=128, help="must match the checkpoint's architecture")
    parser.add_argument("--baseline", choices=tuple(BASELINE_CONTROLLERS), default="crash_fast")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-races", type=int, default=2)
    parser.add_argument("--eval-round-seconds", type=float, default=120.0)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    arguments = parser.parse_args()
    return CompareArguments(
        checkpoint=arguments.checkpoint,
        hidden_size=arguments.hidden_size,
        baseline=arguments.baseline,
        eval_seeds=tuple(arguments.eval_seeds),
        eval_races=arguments.eval_races,
        eval_round_seconds=arguments.eval_round_seconds,
        experiment_dir=arguments.experiment_dir,
    )


def summarize(records: list[dict[str, object]], baseline: str) -> dict[str, float]:
    damages: list[float] = []
    max_speeds: list[float] = []
    distances: list[float] = []
    eliminated = 0
    total = 0
    for record in records:
        if record["baseline"] != baseline:
            continue
        for race in cast("list[dict[str, Any]]", record["races"]):
            challenger = cast("dict[str, Any]", race["challenger"])
            damage = float(challenger["damages"][0])
            damages.append(damage)
            max_speeds.append(float(challenger["max_speeds_mps"][0]))
            distances.append(float(challenger["distances_m"][0]))
            eliminated += 1 if damage >= 1.0 else 0
            total += 1
    return {
        "avg_damage": sum(damages) / len(damages),
        "avg_max_speed_mps": sum(max_speeds) / len(max_speeds),
        "avg_scored_distance_m": sum(distances) / len(distances),
        "elimination_rate": eliminated / total,
        "race_count": float(total),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    agent = SACAgent(observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_sizes=(args.hidden_size,) * 2)
    agent.load(args.checkpoint)
    baselines = {args.baseline: BASELINE_CONTROLLERS[args.baseline]}

    print("=== deterministic (mean action, matches a packaged controller) ===")
    deterministic_records = evaluate_against_baselines(
        agent,
        eval_seeds=args.eval_seeds,
        eval_races=args.eval_races,
        eval_round_seconds=args.eval_round_seconds,
        baselines=baselines,
        deterministic=True,
    )
    print("\n=== stochastic (sampled action, matches training exploration) ===")
    stochastic_records = evaluate_against_baselines(
        agent,
        eval_seeds=args.eval_seeds,
        eval_races=args.eval_races,
        eval_round_seconds=args.eval_round_seconds,
        baselines=baselines,
        deterministic=False,
    )

    deterministic_summary = summarize(deterministic_records, args.baseline)
    stochastic_summary = summarize(stochastic_records, args.baseline)

    print("\n=== summary ===")
    print(f"{'metric':<24}{'deterministic':>16}{'stochastic':>16}")
    for key in deterministic_summary:
        print(f"{key:<24}{deterministic_summary[key]:>16.3f}{stochastic_summary[key]:>16.3f}")

    (experiment_dir / "eval_results_deterministic.json").write_text(json.dumps(deterministic_records, indent=2))
    (experiment_dir / "eval_results_stochastic.json").write_text(json.dumps(stochastic_records, indent=2))
    (experiment_dir / "summary.json").write_text(
        json.dumps({"deterministic": deterministic_summary, "stochastic": stochastic_summary}, indent=2)
    )
    print(f"\n[done] evidence written to {experiment_dir}")


if __name__ == "__main__":
    main()
