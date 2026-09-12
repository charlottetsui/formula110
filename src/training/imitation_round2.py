"""Layout-disjoint demonstrations, learner corrections and driving selection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

from controllers.clone_features import FEATURE_VERSION, ObservationHistory, release_brake
from controllers.imitation import Controller as Clone
from controllers.leaderboard_expert import Controller as Expert
from racing import RobotCommand, RobotSensors, load_student_submission, run_headless_head_to_head
from racing.game.recording import robot_command_to_dict, robot_sensors_to_dict
from racing.race.rules import HeadToHeadRaceRules
from training.imitation import provenance
from training.track_scenarios import Scenario, suite


class CorrectionCollector:
    def __init__(
        self,
        stream: TextIO,
        scenario: Scenario,
        rows: list[tuple[NDArray[np.float32], NDArray[np.float32], int]],
        groups: list[int],
        learner: Path | None = None,
    ) -> None:
        self.stream, self.scenario, self.rows, self.groups = stream, scenario, rows, groups
        self.learner_path = learner
        self.learner = None if learner is None else Clone(learner)
        self.teacher, self.history = Expert(), ObservationHistory()
        self.previous = RobotCommand()
        self.episode = -1

    def copy_for_car(self) -> CorrectionCollector:
        copy = CorrectionCollector(self.stream, self.scenario, self.rows, self.groups, self.learner_path)
        copy.episode = len(self.groups)
        self.groups.append(self.scenario.layout_seed)
        return copy

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = self.history.append(sensors, self.previous)
        label = self.teacher.advise(sensors, self.previous)
        applied = label
        # Contiguous learner blocks allow drift to develop; expert blocks supply
        # corrections. Model predictions use the same actual history as labels.
        if self.learner is not None and (sensors.tick // 60 + self.episode) % 3 == 0:
            action = self.learner.predict(observation)
            applied = release_brake(RobotCommand(throttle=float(action[0]), steer=float(action[1])), self.previous)
        # Deterministic disturbance blocks produce off-center/slow states.
        if self.learner is not None and 200 <= sensors.tick % 500 < 212:
            sign = 1 if (sensors.tick // 500 + self.episode) % 2 else -1
            applied = release_brake(
                RobotCommand(throttle=-0.3, steer=float(np.clip(applied.steer + sign * 0.35, -1, 1))), self.previous
            )
        self.rows.append((observation, np.asarray([label.throttle, label.steer], dtype=np.float32), self.episode))
        self.stream.write(
            json.dumps(
                {
                    "record_type": "corrective_expert_step",
                    "schema_version": 1,
                    "episode_id": self.episode,
                    "layout_seed": self.scenario.layout_seed,
                    "spawn_seed": self.scenario.spawn_seed,
                    "tick": sensors.tick,
                    "sensors": robot_sensors_to_dict(sensors),
                    "previous_applied_action": robot_command_to_dict(self.previous),
                    "expert_action": robot_command_to_dict(label),
                    "applied_action": robot_command_to_dict(applied),
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.previous = applied
        return applied


class ProbeController:
    """Driver with applied-action-aware history for controlled recovery/traffic probes."""

    def __init__(self, model: Path | None = None, *, disturb: bool = False, throttle_scale: float = 1.0) -> None:
        self.model_path, self.disturb, self.throttle_scale = model, disturb, throttle_scale
        self.clone = None if model is None else Clone(model)
        self.teacher, self.history = Expert(), ObservationHistory()
        self.previous = RobotCommand()

    def copy_for_car(self) -> ProbeController:
        return ProbeController(self.model_path, disturb=self.disturb, throttle_scale=self.throttle_scale)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = self.history.append(sensors, self.previous)
        if self.clone is None:
            request = self.teacher.advise(sensors, self.previous)
        else:
            action = self.clone.predict(observation)
            request = RobotCommand(throttle=float(action[0]), steer=float(action[1]))
        if request.throttle > 0:
            request = RobotCommand(throttle=request.throttle * self.throttle_scale, steer=request.steer)
        if self.disturb and 200 <= sensors.tick % 500 < 212:
            sign = 1 if sensors.tick // 500 % 2 else -1
            request = RobotCommand(throttle=-0.3, steer=float(np.clip(request.steer + sign * 0.35, -1, 1)))
        self.previous = release_brake(request, self.previous)
        return self.previous


def stress_evaluate(model: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    results = []
    for scenario in (
        Scenario(0, 42, "default", 60),
        Scenario(101, 4301, "radial", 60),
        Scenario(102, 4302, "stretch", 60),
    ):
        for disturbed in (False, True):
            for name, path in (("expert", None), ("clone", model)):
                result = run_headless_head_to_head(
                    challenger_controller=ProbeController(path, disturb=disturbed),
                    incumbent_controller=ProbeController(throttle_scale=0.6),
                    challenger_copies=1,
                    incumbent_copies=2,
                    random_seed=scenario.spawn_seed,
                    race_count=2,
                    round_seconds=scenario.seconds,
                    rules=HeadToHeadRaceRules(marshal_enabled=False),
                    track_samples=scenario.samples(),
                )
                results.append(
                    {"scenario": scenario.metadata(), "disturbed": disturbed, "model": name, "result": result.to_dict()}
                )
                print(
                    f"stress {scenario.family} {disturbed} {name}: "
                    f"{sum(r.challenger.team_sum_distance_m for r in result.races):.1f}m; "
                    f"{sum(r.challenger.elimination_count for r in result.races)} eliminations",
                    flush=True,
                )
    output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "provenance": provenance(),
                "results": results,
            },
            indent=2,
        )
    )


def collect_round2(output: Path, learner: Path, recovery_only: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows: list[tuple[NDArray[np.float32], NDArray[np.float32], int]] = []
    groups: list[int] = []
    evidence = []
    manifest = provenance()
    with gzip.open(output / "trajectories.jsonl.gz", "wt") as stream:
        for split in ("train",) if recovery_only else ("train", "validation"):
            scenarios = suite(split)
            if recovery_only:
                scenarios += [Scenario(0, seed, "default", 60) for seed in (42, 110, 7, 2024, 8675309)]
            for scenario in scenarios:
                modes = (
                    ("corrections",)
                    if recovery_only
                    else (("expert", "corrections") if split == "train" else ("expert",))
                )
                for mode in modes:
                    collector = CorrectionCollector(
                        stream, scenario, rows, groups, learner if mode == "corrections" else None
                    )
                    result = run_headless_head_to_head(
                        challenger_controller=collector,
                        incumbent_controller=ProbeController(throttle_scale=0.6) if recovery_only else collector,
                        challenger_copies=1,
                        incumbent_copies=2 if recovery_only else 1,
                        random_seed=scenario.spawn_seed,
                        race_count=1,
                        round_seconds=scenario.seconds,
                        rules=HeadToHeadRaceRules(marshal_enabled=False),
                        track_samples=scenario.samples(),
                    )
                    evidence.append(
                        {"split": split, "scenario": scenario.metadata(), "mode": mode, "result": result.to_dict()}
                    )
                    print(f"{split} {scenario.family} {scenario.layout_seed} {mode}: {len(rows)} samples", flush=True)
    np.savez_compressed(
        output / "dataset.npz",
        observations=np.stack([r[0] for r in rows]),
        actions=np.stack([r[1] for r in rows]),
        episode_ids=np.asarray([r[2] for r in rows]),
        episode_seeds=np.asarray(groups),
        feature_version=FEATURE_VERSION,
    )

    (output / "collection.json").write_text(
        json.dumps(
            {
                "provenance": manifest,
                "learner_sha256": hashlib.sha256(learner.read_bytes()).hexdigest(),
                "group_semantics": "episode_seeds contains layout seeds, not spawn seeds",
                "samples": len(rows),
                "episodes": len(groups),
                "recovery_only": recovery_only,
                "results": evidence,
            },
            indent=2,
        )
    )


def merge_datasets(datasets: list[Path], output: Path) -> None:
    """Aggregate corrective labels without losing episode/layout boundaries."""
    if output.exists():
        raise FileExistsError(output)
    observations, actions, episodes, groups = [], [], [], []
    offset = 0
    for path in datasets:
        with np.load(path, allow_pickle=False) as data:
            if int(data["feature_version"]) != FEATURE_VERSION:
                raise ValueError("Dataset representation mismatch")
            observations.append(data["observations"])
            actions.append(data["actions"])
            episodes.append(data["episode_ids"] + offset)
            groups.append(data["episode_seeds"])
            offset += len(data["episode_seeds"])
    np.savez_compressed(
        output,
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        episode_ids=np.concatenate(episodes),
        episode_seeds=np.concatenate(groups),
        feature_version=FEATURE_VERSION,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {"sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in datasets]},
            indent=2,
        )
    )


def evaluate_layouts(models: list[Path], output: Path, split: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    results = []
    for scenario in suite(split):
        expert = load_student_submission("controllers.leaderboard_expert").controller
        for name, controller in [("expert", expert), *[(str(m), Clone(m)) for m in models]]:
            result = run_headless_head_to_head(
                challenger_controller=controller,
                incumbent_controller=expert,
                random_seed=scenario.spawn_seed,
                race_count=2,
                round_seconds=scenario.seconds,
                rules=HeadToHeadRaceRules(marshal_enabled=False),
                track_samples=scenario.samples(),
            )
            stats = [r.challenger for r in result.races]
            summary = {
                "distance_m": sum(s.team_sum_distance_m for s in stats),
                "max_damage": max(max(s.damages) for s in stats),
                "eliminations": sum(s.elimination_count for s in stats),
                "wall_seconds": sum(s.total_wall_contact_seconds for s in stats),
                "low_progress_seconds": sum(s.total_low_progress_seconds for s in stats),
            }
            results.append(
                {"scenario": scenario.metadata(), "model": name, "summary": summary, "result": result.to_dict()}
            )
            print(f"{split} {scenario.layout_seed} {name}: {summary}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": split,
                "provenance": provenance(),
                "model_hashes": {str(m): hashlib.sha256(m.read_bytes()).hexdigest() for m in models},
                "results": results,
            },
            indent=2,
        )
    )


def select_on_validation(report: Path, output: Path) -> str:
    """Choose only from driving validation; test results cannot select a model."""
    data = json.loads(report.read_text())
    if data["split"] != "validation":
        raise ValueError("Selection requires validation results, never final test results")
    if output.exists():
        raise FileExistsError(output)
    baseline = {
        r["scenario"]["layout_seed"]: r["summary"]["distance_m"] for r in data["results"] if r["model"] == "expert"
    }
    summaries: list[dict[str, Any]] = []
    for name in data["model_hashes"]:
        records = [r for r in data["results"] if r["model"] == name]
        ratios = [r["summary"]["distance_m"] / max(1, baseline[r["scenario"]["layout_seed"]]) for r in records]
        max_damage = max(r["summary"]["max_damage"] for r in records)
        eliminations = sum(r["summary"]["eliminations"] for r in records)
        mean_damage = float(np.mean([r["summary"]["max_damage"] for r in records]))
        summary = {
            "model": name,
            "min_distance_ratio": min(ratios),
            "mean_distance_ratio": float(np.mean(ratios)),
            "max_damage": max_damage,
            "eliminations": eliminations,
            "qualified": min(ratios) >= 0.9 and max_damage <= 0.25 and eliminations == 0,
            "selection_score": float(np.mean(ratios)) - 0.5 * mean_damage,
        }
        summaries.append(summary)
    qualified = [s for s in summaries if s["qualified"]]
    selected = max(qualified, key=lambda s: float(s["selection_score"]))["model"] if qualified else None
    output.write_text(
        json.dumps(
            {
                "selection_rule": "no eliminations, worst layout >=90% teacher distance, "
                "max damage <=25%; rank mean distance ratio minus 0.5*mean maximum damage",
                "selected": selected,
                "candidates": summaries,
            },
            indent=2,
        )
    )
    if selected is None:
        raise ValueError("No candidate passed the driving-validation gate")
    return str(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--learner", type=Path, required=True)
    collect.add_argument("--recovery-only", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--models", type=Path, nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test", "final_test"), required=True)
    selection = commands.add_parser("select")
    selection.add_argument("--report", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    stress = commands.add_parser("stress")
    stress.add_argument("--model", type=Path, required=True)
    stress.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--datasets", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect_round2(args.output, args.learner, args.recovery_only)
    elif args.command == "evaluate":
        evaluate_layouts(args.models, args.output, args.split)
    elif args.command == "select":
        print(select_on_validation(args.report, args.output))
    elif args.command == "stress":
        stress_evaluate(args.model, args.output)
    else:
        merge_datasets(args.datasets, args.output)


if __name__ == "__main__":
    main()
