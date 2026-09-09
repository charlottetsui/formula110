from __future__ import annotations

# Torch exposes these functions without complete parameter annotations.
# pyright: reportUnknownMemberType=false
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from controllers.clone_features import FRAME_DIM, OBSERVATION_DIM, ObservationHistory, encode_frame, release_brake
from controllers.imitation import Controller
from racing import OdometrySensors, RobotCommand, RobotSensors
from training.imitation import Collector, export_policy, fit
from training.sac import GaussianPolicy


def test_observation_retains_expert_speed_range_and_handles_no_hits() -> None:
    slow = encode_frame(RobotSensors(odometry=OdometrySensors(speed_mps=23)), RobotCommand())
    fast = encode_frame(RobotSensors(odometry=OdometrySensors(speed_mps=38)), RobotCommand())
    assert slow[0] < fast[0]
    assert np.isfinite(fast).all()
    assert fast.shape == (FRAME_DIM,)


def test_history_is_causal_bounded_and_resets_at_new_episode() -> None:
    history = ObservationHistory()
    sensors = RobotSensors()
    first = history.append(sensors, RobotCommand())
    observed = first
    for tick in range(1, 70):
        observed = history.append(replace(sensors, tick=tick), RobotCommand(throttle=0.5, steer=-0.2))
    assert len(history.frames) == 33
    assert observed[-2:] == pytest.approx([0.5, -0.2])
    assert history.append(sensors, RobotCommand()) == pytest.approx(first)


def test_brake_release_is_exactly_one_tick() -> None:
    request = RobotCommand(throttle=0.001, steer=0.4)
    neutral = release_brake(request, RobotCommand(throttle=-1.0))
    assert neutral.throttle == 0.0
    assert neutral.steer == 0.4
    assert release_brake(request, neutral) == request


def test_numpy_export_matches_torch_and_copies_have_independent_history(tmp_path: Path) -> None:
    torch.manual_seed(10)
    policy = GaussianPolicy(observation_dim=OBSERVATION_DIM, action_dim=2, hidden_sizes=(16, 16))
    path = tmp_path / "policy.npz"
    export_policy(policy, path)
    clone = Controller(path)
    samples = np.random.default_rng(5).normal(size=(10, OBSERVATION_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = policy.deterministic_action(torch.from_numpy(samples)).numpy()
    assert np.stack([clone.predict(sample) for sample in samples]) == pytest.approx(expected, abs=1e-6)
    clone(RobotSensors())
    other = clone.copy_for_car()
    assert len(other.history.frames) == 0
    assert other.previous == RobotCommand()
    assert len(clone.history.frames) == 1


def test_collector_pairs_pre_action_observation_and_separates_car_state() -> None:
    stream = io.StringIO()
    prototype = Collector(stream, 1001, [], [])
    first, second = prototype.copy_for_car(), prototype.copy_for_car()
    action = first(RobotSensors())
    first(RobotSensors(tick=1))
    second(RobotSensors())
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["previous_applied_action"] == {"throttle": 0.0, "steer": 0.0}
    assert rows[1]["previous_applied_action"]["throttle"] == action.throttle
    assert rows[0]["episode_id"] != rows[2]["episode_id"]
    assert prototype.records[1][0][FRAME_DIM - 2] == action.throttle
    assert rows[2]["previous_applied_action"] == rows[0]["previous_applied_action"]


def test_fit_holds_out_entire_seed_groups(tmp_path: Path) -> None:
    dataset = tmp_path / "data.npz"
    rng = np.random.default_rng(8)
    np.savez(
        dataset,
        observations=rng.normal(size=(12, OBSERVATION_DIM)).astype(np.float32),
        actions=np.zeros((12, 2), dtype=np.float32),
        episode_ids=np.repeat([0, 1, 2], 4),
        episode_seeds=np.asarray([1001, 1001, 2001]),
        feature_version=1,
    )
    fit(dataset, tmp_path / "run", [2001], 1, 0, 16)
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["training_seeds"] == [1001]
    assert config["train_samples"] == 8
    assert config["validation_samples"] == 4
    assert Controller(tmp_path / "run" / "policy.npz")(RobotSensors()).throttle == pytest.approx(0.0, abs=1.0)
