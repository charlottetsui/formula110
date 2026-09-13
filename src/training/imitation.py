"""Expert trajectory collection, supervised fitting and matched race evaluation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from controllers.clone_features import FEATURE_VERSION, OBSERVATION_DIM, ObservationHistory
from controllers.imitation import Controller as Clone
from controllers.leaderboard_expert import Controller as Expert
from racing import RobotCommand, RobotSensors, load_student_submission, run_headless_head_to_head
from racing.game.recording import robot_command_to_dict, robot_sensors_to_dict
from racing.race.rules import HeadToHeadRaceRules
from training.sac import GaussianPolicy


def provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    paths = [
        "src/controllers/leaderboard_expert.py",
        "src/controllers/clone_features.py",
        "src/controllers/imitation.py",
        "src/training/imitation.py",
        "src/training/sac.py",
        "src/training/imitation_round2.py",
        "src/training/track_scenarios.py",
        "src/training/imitation_handoff.py",
        "src/racing/race/head_to_head.py",
        "uv.lock",
    ]
    return {
        "git_commit": revision.stdout.strip(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    }


class Collector:
    def __init__(
        self,
        stream: TextIO,
        seed: int,
        records: list[tuple[NDArray[np.float32], NDArray[np.float32], int]],
        episodes: list[int],
    ) -> None:
        self.stream, self.seed, self.records, self.episodes = stream, seed, records, episodes
        self.episode = -1
        self.teacher = Expert()
        self.history = ObservationHistory()
        self.previous = RobotCommand()

    def copy_for_car(self) -> Collector:
        copy = Collector(self.stream, self.seed, self.records, self.episodes)
        copy.episode = len(self.episodes)
        self.episodes.append(self.seed)
        return copy

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = self.history.append(sensors, self.previous)
        action = self.teacher(sensors)
        self.records.append((observation, np.asarray([action.throttle, action.steer], dtype=np.float32), self.episode))
        self.stream.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "expert_control_step",
                    "episode_id": self.episode,
                    "seed": self.seed,
                    "tick": sensors.tick,
                    "sensors": robot_sensors_to_dict(sensors),
                    "previous_applied_action": robot_command_to_dict(self.previous),
                    "expert_action": robot_command_to_dict(action),
                    "applied_action": robot_command_to_dict(action),
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.previous = action
        return action


def collect(directory: Path, seeds: list[int], seconds: float) -> None:
    if not seeds or len(set(seeds)) != len(seeds) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Use unique seeds and a positive finite duration")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records: list[tuple[NDArray[np.float32], NDArray[np.float32], int]] = []
    episodes: list[int] = []
    results: list[dict[str, object]] = []
    with gzip.open(directory / "trajectories.jsonl.gz", "wt") as stream:
        for seed in seeds:
            collector = Collector(stream, seed, records, episodes)
            result = run_headless_head_to_head(
                challenger_controller=collector,
                incumbent_controller=collector,
                challenger_name="expert",
                incumbent_name="expert",
                random_seed=seed,
                race_count=1,
                round_seconds=seconds,
                rules=HeadToHeadRaceRules(marshal_enabled=False),
            )
            results.append(result.to_dict())
            print(f"Collected seed {seed}: {len(records)} total ticks", flush=True)
    if not records:
        raise ValueError("No expert samples collected")
    np.savez_compressed(
        directory / "dataset.npz",
        observations=np.stack([r[0] for r in records]),
        actions=np.stack([r[1] for r in records]),
        episode_ids=np.asarray([r[2] for r in records]),
        episode_seeds=np.asarray(episodes),
        feature_version=FEATURE_VERSION,
    )
    (directory / "race_results.json").write_text(json.dumps(results, indent=2))
    teacher_path = Path(__file__).parents[1] / "controllers" / "leaderboard_expert.py"
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "seconds": seconds,
                "dt_s": 1 / 60,
                "marshal_enabled": False,
                "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                "feature_version": FEATURE_VERSION,
                "samples": len(records),
                "episodes": len(episodes),
                "collection_seconds": time.monotonic() - started,
                "provenance": provenance(),
            },
            indent=2,
        )
    )


def export_policy(policy: GaussianPolicy, path: Path) -> None:
    layers = [layer for layer in policy.trunk if isinstance(layer, nn.Linear)] + [policy.mean_head]
    payload = {"feature_version": np.asarray(FEATURE_VERSION)}
    for index, layer in enumerate(layers):
        payload[f"weight_{index}"] = layer.weight.detach().numpy()
        payload[f"bias_{index}"] = layer.bias.detach().numpy()
    np.savez_compressed(path, allow_pickle=False, **payload)


def fit(
    dataset: Path,
    directory: Path,
    validation_seeds: list[int],
    epochs: int,
    seed: int,
    hidden_size: int = 128,
    initial_actor: Path | None = None,
) -> None:
    if epochs < 1 or hidden_size < 1:
        raise ValueError("Epochs and hidden size must be positive")
    if directory.exists():
        raise FileExistsError(directory)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(dataset, allow_pickle=False) as data:
        if int(data["feature_version"].item()) != FEATURE_VERSION:
            raise ValueError("Dataset feature version mismatch")
        observations = torch.from_numpy(data["observations"])
        actions = torch.from_numpy(data["actions"])
        row_seeds = data["episode_seeds"][data["episode_ids"]]
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM or actions.shape != (len(observations), 2):
        raise ValueError("Dataset observation/action dimensions do not match the policy")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all() or torch.any(actions.abs() > 1):
        raise ValueError("Dataset must contain finite observations and bounded actions")
    if not validation_seeds or not set(validation_seeds).issubset(set(row_seeds.tolist())):
        raise ValueError("All validation seeds must be present in the dataset")
    validation = np.isin(row_seeds, validation_seeds)
    train_indices, val_indices = np.flatnonzero(~validation), np.flatnonzero(validation)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Need nonempty training and validation seed groups")
    directory.mkdir(parents=True, exist_ok=False)
    policy = GaussianPolicy(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=(hidden_size, hidden_size))
    if initial_actor is not None:
        payload = torch.load(initial_actor, map_location="cpu", weights_only=True)
        if payload["feature_version"] != FEATURE_VERSION or payload["observation_dim"] != OBSERVATION_DIM:
            raise ValueError("Initial actor representation mismatch")
        policy.load_state_dict(payload["policy"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    # Equal total sampling mass for braking, neutral/partial, and full acceleration.
    targets = actions.numpy()[train_indices, 0]
    groups = np.where(targets < -0.1, 0, np.where(targets > 0.9, 2, 1))
    counts = np.bincount(groups, minlength=3)
    probabilities = 1.0 / np.maximum(counts[groups], 1)
    probabilities /= probabilities.sum()
    best = float("inf")
    with (directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_throttle_mse", "val_steer_mse"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            sampled = rng.choice(train_indices, size=len(train_indices), replace=True, p=probabilities)
            total = 0.0
            for start in range(0, len(sampled), 512):
                indices = sampled[start : start + 512]
                prediction = policy.deterministic_action(observations[indices])
                errors = (prediction - actions[indices]).square().mean(dim=0)
                loss = errors[0] + 4.0 * errors[1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            with torch.no_grad():
                errors = (
                    (policy.deterministic_action(observations[val_indices]) - actions[val_indices]).square().mean(0)
                )
            score = float(errors[0] + 4.0 * errors[1])
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": total / len(sampled),
                    "val_throttle_mse": float(errors[0]),
                    "val_steer_mse": float(errors[1]),
                }
            )
            stream.flush()
            if score < best:
                best = score
                export_policy(policy, directory / "policy.npz")
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "feature_version": FEATURE_VERSION,
                        "observation_dim": OBSERVATION_DIM,
                        "hidden_size": hidden_size,
                        "epoch": epoch,
                    },
                    directory / "actor.pt",
                )
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch}: validation weighted MSE {score:.5f}", flush=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "validation_seeds": validation_seeds,
                "training_seeds": sorted(set(row_seeds[~validation].tolist())),
                "epochs": epochs,
                "seed": seed,
                "initial_actor": None if initial_actor is None else str(initial_actor),
                "initial_actor_sha256": None
                if initial_actor is None
                else hashlib.sha256(initial_actor.read_bytes()).hexdigest(),
                "hidden_size": hidden_size,
                "observation_dim": OBSERVATION_DIM,
                "feature_version": FEATURE_VERSION,
                "best_validation_weighted_mse": best,
                "train_samples": len(train_indices),
                "validation_samples": len(val_indices),
                "throttle_group_counts": counts.tolist(),
                "selection": "minimum validation throttle MSE + 4 * steer MSE",
                "fit_seconds": time.monotonic() - started,
                "provenance": provenance(),
                "batch_size": 512,
                "learning_rate": 3e-4,
                "steering_loss_weight": 4.0,
            },
            indent=2,
        )
    )


def evaluate(model: Path, output: Path, seeds: list[int], seconds: float) -> None:
    if not seeds or len(set(seeds)) != len(seeds) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Use unique seeds and a positive finite duration")
    if output.exists():
        raise FileExistsError(output)
    results = []
    for seed in seeds:
        record: dict[str, object] = {"seed": seed}
        expert = load_student_submission("controllers.leaderboard_expert").controller
        for name, controller in (("expert", expert), ("clone", Clone(model))):
            result = run_headless_head_to_head(
                challenger_controller=controller,
                incumbent_controller=expert,
                challenger_name=name,
                incumbent_name="expert-opponent",
                random_seed=seed,
                race_count=2,
                round_seconds=seconds,
                rules=HeadToHeadRaceRules(marshal_enabled=False),
            )
            record[name] = result.to_dict()
            print(
                f"{name} seed {seed}: {sum(r.challenger.team_sum_distance_m for r in result.races):.1f} m", flush=True
            )
        results.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "results": results,
                "provenance": provenance(),
            },
            indent=2,
        )
    )
