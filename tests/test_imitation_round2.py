from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from controllers.imitation import Controller
from controllers.leaderboard_expert import Controller as Expert
from racing import RobotCommand, RobotSensors, load_student_submission, run_headless_head_to_head
from racing.track.world import TrackPoint
from training.imitation_handoff import HandoffController, initialize_sac, load_prepared, prepare
from training.imitation_round2 import CorrectionCollector, merge_datasets, select_on_validation
from training.track_scenarios import Scenario, suite, validate_centerline


def test_layout_split_excludes_validation_and_test_geometry() -> None:
    groups = [{s.layout_seed for s in suite(split)} for split in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    a, b = Scenario(10, 1), Scenario(10, 2)
    assert a.samples() == b.samples()  # spawn seed cannot alter geometry
    assert a.samples() != Scenario(11, 1).samples()


def test_crossing_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="crosses"):
        validate_centerline((TrackPoint(0, 0), TrackPoint(10, 10), TrackPoint(0, 10), TrackPoint(10, 0)))


def test_selection_cannot_use_final_test_results(tmp_path: Path) -> None:
    report = tmp_path / "test.json"
    report.write_text(json.dumps({"split": "test"}))
    with pytest.raises(ValueError, match="never final test"):
        select_on_validation(report, tmp_path / "selection.json")


def test_merging_preserves_episode_ids_and_layout_groups(tmp_path: Path) -> None:
    sources: list[Path] = []
    for index in range(2):
        source = tmp_path / f"data-{index}.npz"
        np.savez(
            source,
            observations=np.zeros((2, 336), dtype=np.float32),
            actions=np.full((2, 2), index, dtype=np.float32),
            episode_ids=np.asarray([0, 0]),
            episode_seeds=np.asarray([101 + index]),
            feature_version=1,
        )
        sources.append(source)
    output = tmp_path / "merged.npz"
    merge_datasets(sources, output)
    with np.load(output) as data:
        assert data["episode_ids"].tolist() == [0, 0, 1, 1]
        assert data["episode_seeds"].tolist() == [101, 102]
        assert data["actions"][:, 0].tolist() == [0, 0, 1, 1]


def test_prepared_sac_round_trip_preserves_initialization(tmp_path: Path) -> None:
    source = Path("experiments/2026-09-08_imitation-v1")
    prepare(source / "actor.pt", source / "policy.npz", tmp_path / "handoff")
    agent = load_prepared(tmp_path / "handoff")
    assert agent.gamma == pytest.approx(0.997)
    assert np.allclose(agent.policy.log_std_head.bias.detach().numpy(), -4.6)
    assert float(agent.alpha.detach()) == pytest.approx(0.01)


def test_expert_advice_uses_applied_throttle_instead_of_unexecuted_advice() -> None:
    teacher = Expert()
    request = teacher.advise(RobotSensors(tick=1), RobotCommand(throttle=-0.2, steer=0.0))
    assert request.throttle == 0
    # Rejected advice does not release the actual braking state.
    again = teacher.advise(RobotSensors(tick=2), RobotCommand(throttle=-0.2, steer=0.0))
    assert again.throttle == 0
    released = teacher.advise(RobotSensors(tick=3), RobotCommand())
    assert released.throttle > 0


def test_corrections_record_labels_separately_from_perturbed_actions() -> None:
    stream = io.StringIO()
    collector = CorrectionCollector(stream, Scenario(10, 1), [], [], Path("src/controllers/imitation_policy.npz"))
    car = collector.copy_for_car()
    car(RobotSensors(tick=200))
    car(RobotSensors(tick=201))
    first, second = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert first["expert_action"] != first["applied_action"]
    assert second["previous_applied_action"] == first["applied_action"]
    assert car.rows[0][1][0] == first["expert_action"]["throttle"]


def test_handoff_preserves_clone_actions_and_sets_conservative_noise() -> None:
    root = Path("experiments/2026-09-08_imitation-v1")
    agent = initialize_sac(root / "actor.pt")
    clone, sac = Controller(root / "policy.npz"), HandoffController(agent)
    for tick in range(40):
        sensors = RobotSensors(tick=tick)
        expected, actual = clone(sensors), sac(sensors)
        assert actual.throttle == pytest.approx(expected.throttle, abs=2e-6)
        assert actual.steer == pytest.approx(expected.steer, abs=2e-6)
    assert float(agent.alpha.detach()) == pytest.approx(0.01)
    assert np.allclose(agent.policy.log_std_head.bias.detach().numpy(), -3.5)
    assert len(sac.copy_for_car().history.frames) == 0


def test_custom_geometry_reaches_camera_and_collision_simulation() -> None:
    # Actual physics smoke test: samples alter sensed geometry under matched placement seeds.
    observations: list[RobotSensors] = []
    expert = load_student_submission("controllers.leaderboard_expert").controller
    for layout in (Scenario(10, 9), Scenario(11, 9)):
        run_headless_head_to_head(
            challenger_controller=expert,
            incumbent_controller=expert,
            race_count=1,
            round_seconds=0.05,
            random_seed=9,
            track_samples=layout.samples(),
            sensor_sample_callback=lambda entry, sensors: observations.append(sensors),
        )
    assert observations[0].camera.lookahead_offsets_m != observations[len(observations) // 2].camera.lookahead_offsets_m
