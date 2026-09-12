"""Initialize SAC from a clone without changing its deterministic driving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from controllers.clone_features import FEATURE_VERSION, OBSERVATION_DIM, ObservationHistory, release_brake
from controllers.imitation import Controller as Clone
from racing import RobotCommand, RobotSensors
from training.sac import SACAgent


def initialize_sac(actor_path: Path, *, seed: int = 0, log_std: float = -3.5) -> SACAgent:
    """Ready for critic warmup; critics are fresh, not pretrained value estimates.

    The actor's unused BC variance head is replaced with controlled exploration.
    Do not enable the original training loop: its observations and warmup differ.
    """
    if not -8 <= log_std <= -2:
        raise ValueError("Choose conservative initial log standard deviation in [-8, -2]")
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    if payload["feature_version"] != FEATURE_VERSION or payload["observation_dim"] != OBSERVATION_DIM:
        raise ValueError("Actor representation is incompatible with cloning history")
    hidden_size = int(payload["hidden_size"])
    agent = SACAgent(
        observation_dim=OBSERVATION_DIM, hidden_sizes=(hidden_size, hidden_size), seed=seed, gamma=0.997, actor_lr=3e-5
    )
    agent.policy.load_state_dict(payload["policy"])
    with torch.no_grad():
        agent.policy.log_std_head.weight.zero_()
        agent.policy.log_std_head.bias.fill_(log_std)
        agent.log_alpha.fill_(float(np.log(0.01)))
    return agent


class HandoffController:
    """SAC inference with the exact clone observation and action contract.

    This adapter deliberately performs no replay writes or learning. The next
    training stage must record requested and applied actions separately and
    handle real episode endings before enabling updates.
    """

    def __init__(self, agent: SACAgent, *, deterministic: bool = True) -> None:
        self.agent, self.deterministic = agent, deterministic
        self.history = ObservationHistory()
        self.previous = RobotCommand()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            self.previous = RobotCommand()
        observation = self.history.append(sensors, self.previous)
        action = self.agent.act(observation, deterministic=self.deterministic)
        requested = RobotCommand(throttle=float(action[0]), steer=float(action[1]))
        self.previous = release_brake(requested, self.previous)
        return self.previous

    def copy_for_car(self) -> HandoffController:
        return HandoffController(self.agent, deterministic=self.deterministic)


def prepare(actor_path: Path, numpy_path: Path, output: Path, log_std: float = -4.6) -> None:
    """Write a reproducible SAC initialization and check mean-network parity."""
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    agent = initialize_sac(actor_path, log_std=log_std)
    clone = Clone(numpy_path)
    inputs = np.random.default_rng(110).normal(0, 0.5, (256, OBSERVATION_DIM)).astype(np.float32)
    error = max(float(np.max(np.abs(clone.predict(obs) - agent.act(obs, deterministic=True)))) for obs in inputs)
    if error > 2e-6:
        raise ValueError(f"Actor/NumPy mean parity failed: {error}")
    output.mkdir(parents=True)
    agent.save(output / "initial_sac.pt")
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    payload["policy"] = agent.policy.state_dict()
    torch.save(payload, output / "actor.pt")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "imitation complete; SAC initialization, before critic warmup",
                "actor_source": str(actor_path),
                "actor_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
                "numpy_source": str(numpy_path),
                "numpy_sha256": hashlib.sha256(numpy_path.read_bytes()).hexdigest(),
                "feature_version": FEATURE_VERSION,
                "observation_dim": OBSERVATION_DIM,
                "hidden_size": int(payload["hidden_size"]),
                "log_std": log_std,
                "initial_alpha": 0.01,
                "gamma": 0.997,
                "actor_lr": 3e-5,
                "critic_lr": 3e-4,
                "initialization_seed": 0,
                "maximum_mean_difference": error,
                "critics_trained": False,
                "replay_seeded": False,
                "actor_updates": 0,
                "action_contract": (
                    "Critic actions are policy requests before release_brake; history uses applied commands."
                ),
                "demonstration_contract": (
                    "Dataset actions are expert LABELS. In corrective data these differ from actions executed "
                    "by the simulator. Use JSONL applied_action for transitions, not labels paired with next states."
                ),
                "next_training_requirements": [
                    "Build replay using valid within-episode transitions and real terminal handling",
                    "Warm critics with the actor frozen before enabling small actor updates",
                    "Retain expert labels for a decaying imitation loss",
                    "Evaluate and checkpoint against this frozen baseline",
                ],
            },
            indent=2,
        )
    )


def load_prepared(directory: Path) -> SACAgent:
    """Restore configuration as well as weights; generic SAC.load does not do this."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["feature_version"] != FEATURE_VERSION or manifest["observation_dim"] != OBSERVATION_DIM:
        raise ValueError("Prepared SAC representation mismatch")
    width = int(manifest["hidden_size"])
    agent = SACAgent(
        observation_dim=OBSERVATION_DIM,
        hidden_sizes=(width, width),
        seed=manifest["initialization_seed"],
        gamma=manifest["gamma"],
        actor_lr=manifest["actor_lr"],
        critic_lr=manifest["critic_lr"],
    )
    agent.load(directory / "initial_sac.pt")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--numpy-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-std", type=float, default=-4.6)
    arguments = parser.parse_args()
    prepare(arguments.actor, arguments.numpy_policy, arguments.output, arguments.log_std)
