from __future__ import annotations

from math import hypot, sqrt

import pytest

from racing.race.progress import resolve_track
from racing.track.procedural import (
    MAXIMUM_PROCEDURAL_LENGTH_RATIO,
    MINIMUM_PROCEDURAL_LENGTH_RATIO,
    MUGELLO_REFERENCE_LENGTH_M,
    TRACK_ID_PROCEDURAL,
    ProceduralTrackConfig,
    generate_procedural_track,
    hairpin_count,
    longest_straight_length_m,
    maximum_hairpin_turn_degrees,
    minimum_coherent_turn_radius_m,
    procedural_track_length_for_seed,
    procedural_track_validation_error,
    reflection_asymmetry_score,
    resample_closed_path,
)
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint, sampled_track_centerline, total_track_length


def _normalized_shape_distance(
    first: tuple[TrackPoint, ...],
    second: tuple[TrackPoint, ...],
) -> float:
    def normalized_points(points: tuple[TrackPoint, ...]) -> tuple[complex, ...]:
        length_m = total_track_length(points)
        samples = resample_closed_path(points, spacing_m=length_m / 180)
        center_x = sum(point.x for point in samples) / len(samples)
        center_z = sum(point.z for point in samples) / len(samples)
        centered = tuple(complex(point.x - center_x, point.z - center_z) for point in samples)
        scale = sqrt(sum(abs(point) ** 2 for point in centered) / len(centered))
        return tuple(point / scale for point in centered)

    first_normalized = normalized_points(first)
    second_normalized = normalized_points(second)
    sample_count = len(first_normalized)
    best_distance = float("inf")
    for shift in range(sample_count):
        cross = sum(
            first_normalized[index].conjugate() * second_normalized[(index + shift) % sample_count]
            for index in range(sample_count)
        )
        rotation = cross / abs(cross) if cross else complex(1.0)
        distance = sqrt(
            sum(
                abs(first_normalized[index] * rotation - second_normalized[(index + shift) % sample_count]) ** 2
                for index in range(sample_count)
            )
            / sample_count
        )
        best_distance = min(best_distance, distance)
    return best_distance


def test_procedural_track_is_reproducible_and_seeded() -> None:
    first = generate_procedural_track(110)
    repeated = generate_procedural_track(110)
    different = generate_procedural_track(2026)

    assert first == repeated
    assert first.track_id == "procedural-110"
    assert first.points != different.points


@pytest.mark.parametrize("seed", range(64))
def test_representative_procedural_tracks_are_valid_and_uniform(seed: int) -> None:
    config = ProceduralTrackConfig()
    layout = generate_procedural_track(seed, config=config)
    segment_lengths = tuple(
        hypot(
            layout.points[(index + 1) % len(layout.points)].x - point.x,
            layout.points[(index + 1) % len(layout.points)].z - point.z,
        )
        for index, point in enumerate(layout.points)
    )

    assert procedural_track_validation_error(layout.points, config=config) is None
    assert total_track_length(layout.points) == pytest.approx(
        procedural_track_length_for_seed(seed, config=config), abs=0.01
    )
    assert config.minimum_length_m <= total_track_length(layout.points) <= config.maximum_length_m
    assert longest_straight_length_m(layout.points, config=config) >= config.minimum_straight_length_m
    assert maximum_hairpin_turn_degrees(layout.points, config=config) >= config.minimum_hairpin_turn_degrees
    assert config.minimum_hairpin_count <= hairpin_count(layout.points, config=config) <= config.maximum_hairpin_count
    assert minimum_coherent_turn_radius_m(layout.points, config=config) <= config.maximum_technical_turn_radius_m
    assert reflection_asymmetry_score(layout.points) >= config.minimum_reflection_asymmetry
    assert max(segment_lengths) - min(segment_lengths) < config.sample_spacing_m * 0.08


def test_default_length_range_is_within_twenty_five_percent_of_mugello() -> None:
    config = ProceduralTrackConfig()

    assert config.minimum_length_m == pytest.approx(MUGELLO_REFERENCE_LENGTH_M * MINIMUM_PROCEDURAL_LENGTH_RATIO)
    assert config.maximum_length_m == pytest.approx(MUGELLO_REFERENCE_LENGTH_M * MAXIMUM_PROCEDURAL_LENGTH_RATIO)


@pytest.mark.parametrize(
    ("seed", "expected_hairpin_count"),
    ((7, 3), (42, 4), (110, 5)),
)
def test_seeded_tracks_cover_three_to_five_hairpins(seed: int, expected_hairpin_count: int) -> None:
    config = ProceduralTrackConfig()

    layout = generate_procedural_track(seed, config=config)

    assert hairpin_count(layout.points, config=config) == expected_hairpin_count


@pytest.mark.parametrize(("first_seed", "second_seed"), ((7, 19), (0, 42)))
def test_seeds_with_the_same_hairpin_count_still_have_distinct_outlines(
    first_seed: int,
    second_seed: int,
) -> None:
    first = generate_procedural_track(first_seed)
    second = generate_procedural_track(second_seed)

    assert hairpin_count(first.points) == hairpin_count(second.points)
    assert _normalized_shape_distance(first.points, second.points) > 0.10


@pytest.mark.parametrize(
    "target_length_m",
    (
        MUGELLO_REFERENCE_LENGTH_M * MINIMUM_PROCEDURAL_LENGTH_RATIO,
        MUGELLO_REFERENCE_LENGTH_M * MAXIMUM_PROCEDURAL_LENGTH_RATIO,
    ),
)
def test_procedural_track_accepts_both_length_boundaries(target_length_m: float) -> None:
    config = ProceduralTrackConfig(target_length_m=target_length_m)

    layout = generate_procedural_track(110, config=config)

    assert total_track_length(layout.points) == pytest.approx(target_length_m, abs=0.01)
    assert procedural_track_validation_error(layout.points, config=config) is None


def test_procedural_generation_reports_incompatible_constraint_failure() -> None:
    impossible_config = ProceduralTrackConfig(minimum_bend_radius_m=100.0, maximum_attempts=2)

    with pytest.raises(ValueError, match=r"cannot fit the configured hairpin count"):
        generate_procedural_track(110, config=impossible_config)


def test_resolved_default_track_preserves_existing_mugello_samples() -> None:
    resolved = resolve_track()

    assert resolved.track_id == TRACK_ID_MUGELLO_SHORT
    assert resolved.seed is None
    assert resolved.samples == sampled_track_centerline(samples_per_segment=10)


def test_resolved_procedural_track_shares_samples_with_progress_model() -> None:
    resolved = resolve_track(TRACK_ID_PROCEDURAL, 110)

    assert resolved.track_id == "procedural-110"
    assert resolved.seed == 110
    assert resolved.samples == resolved.model.points


def test_resolving_procedural_track_requires_seed() -> None:
    with pytest.raises(ValueError, match="require a track seed"):
        resolve_track(TRACK_ID_PROCEDURAL)
