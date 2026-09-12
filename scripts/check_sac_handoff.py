#!/usr/bin/env python3
"""Evaluate initialized SAC with a frozen actor, including low-noise exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from controllers.imitation import Controller
from racing import load_student_submission, run_headless_head_to_head
from racing.race.rules import HeadToHeadRaceRules
from training.imitation_handoff import HandoffController, load_prepared
from training.track_scenarios import Scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/imitation-v2-data/dataset.npz"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    manifest = json.loads((args.directory / "manifest.json").read_text())
    policy_path = Path(manifest["numpy_source"])
    agent = load_prepared(args.directory)
    clone = Controller(policy_path)
    before = [p.detach().clone() for p in agent.policy.parameters()]
    maximum_error = 0.0
    rows = []
    for scenario in (
        Scenario(0, 110, "default", 30),
        Scenario(0, 42, "default", 30),
        Scenario(101, 4301, "radial", 30),
    ):
        for deterministic in (True, False):
            expert = load_student_submission("controllers.leaderboard_expert").controller
            controller = HandoffController(agent, deterministic=deterministic)
            result = run_headless_head_to_head(
                challenger_controller=controller,
                incumbent_controller=expert,
                race_count=2,
                round_seconds=scenario.seconds,
                random_seed=scenario.spawn_seed,
                rules=HeadToHeadRaceRules(marshal_enabled=False),
                track_samples=scenario.samples(),
            )
            rows.append({"scenario": scenario.metadata(), "deterministic": deterministic, "result": result.to_dict()})
            print(
                scenario.family,
                scenario.spawn_seed,
                deterministic,
                sum(r.challenger.team_sum_distance_m for r in result.races),
                "eliminations",
                sum(r.challenger.elimination_count for r in result.races),
                flush=True,
            )
    with np.load(args.dataset, allow_pickle=False) as dataset:
        for observation in dataset["observations"][::100]:
            error = float(np.max(np.abs(clone.predict(observation) - agent.act(observation, deterministic=True))))
            maximum_error = max(maximum_error, error)
    unchanged = all(torch.equal(a, b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    if not unchanged or maximum_error > 2e-6:
        raise AssertionError("Handoff changed actor behavior")
    args.output.write_text(
        json.dumps(
            {
                "actor_unchanged": unchanged,
                "max_recorded_observation_error": maximum_error,
                "initial_sac_sha256": hashlib.sha256((args.directory / "initial_sac.pt").read_bytes()).hexdigest(),
                "actor_updates": 0,
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
