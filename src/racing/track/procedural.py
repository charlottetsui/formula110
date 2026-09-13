"""Deterministic procedural closed-course generation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, hypot, isfinite, pi, sin, sqrt
from random import Random

from racing.track.world import TrackLayout, TrackPoint, total_track_length

TRACK_ID_PROCEDURAL = "procedural"
MUGELLO_REFERENCE_LENGTH_M = 183.065982031
MINIMUM_PROCEDURAL_LENGTH_RATIO = 0.75
MAXIMUM_PROCEDURAL_LENGTH_RATIO = 1.25


@dataclass(frozen=True, slots=True)
class ProceduralTrackConfig:
    """Shape and safety constraints for a generated track."""

    reference_length_m: float = MUGELLO_REFERENCE_LENGTH_M
    minimum_length_ratio: float = MINIMUM_PROCEDURAL_LENGTH_RATIO
    maximum_length_ratio: float = MAXIMUM_PROCEDURAL_LENGTH_RATIO
    target_length_m: float | None = None
    control_point_count: int = 36
    samples_per_control_point: int = 12
    sample_spacing_m: float = 0.9
    shape_variation: float = 0.20
    minimum_straight_length_m: float = 14.0
    straight_lateral_tolerance_m: float = 0.20
    minimum_hairpin_turn_degrees: float = 150.0
    minimum_hairpin_count: int = 3
    maximum_hairpin_count: int = 5
    maximum_hairpin_length_m: float = 64.0
    minimum_hairpin_turn_coherence: float = 0.82
    minimum_technical_turn_degrees: float = 75.0
    maximum_technical_turn_length_m: float = 30.0
    maximum_technical_turn_radius_m: float = 11.0
    minimum_bend_radius_m: float = 6.75
    minimum_nonlocal_clearance_m: float = 14.0
    envelope_half_width_m: float = 6.5
    minimum_reflection_asymmetry: float = 0.10
    maximum_attempts: int = 64

    def __post_init__(self) -> None:
        if self.reference_length_m <= 0.0:
            raise ValueError("reference_length_m must be positive")
        if not 0.0 < self.minimum_length_ratio <= self.maximum_length_ratio:
            raise ValueError("length ratios must be positive and ordered")
        if (
            self.target_length_m is not None
            and not self.minimum_length_m <= self.target_length_m <= self.maximum_length_m
        ):
            raise ValueError("target_length_m must be within the configured length range")
        if self.control_point_count < 8:
            raise ValueError("control_point_count must be at least 8")
        if self.samples_per_control_point < 2:
            raise ValueError("samples_per_control_point must be at least 2")
        if self.sample_spacing_m <= 0.0:
            raise ValueError("sample_spacing_m must be positive")
        if not 0.0 <= self.shape_variation <= 1.25:
            raise ValueError("shape_variation must be between 0.0 and 1.25")
        if self.minimum_straight_length_m <= 0.0 or self.straight_lateral_tolerance_m <= 0.0:
            raise ValueError("straight constraints must be positive")
        if not 90.0 <= self.minimum_hairpin_turn_degrees < 180.0:
            raise ValueError("minimum_hairpin_turn_degrees must be between 90 and 180")
        if not 3 <= self.minimum_hairpin_count <= self.maximum_hairpin_count:
            raise ValueError("hairpin counts must be ordered and at least 3")
        if self.maximum_hairpin_length_m <= 0.0:
            raise ValueError("maximum_hairpin_length_m must be positive")
        if not 0.0 < self.minimum_hairpin_turn_coherence <= 1.0:
            raise ValueError("minimum_hairpin_turn_coherence must be between 0 and 1")
        if not 0.0 < self.minimum_technical_turn_degrees < self.minimum_hairpin_turn_degrees:
            raise ValueError("minimum_technical_turn_degrees must be below the hairpin threshold")
        if self.maximum_technical_turn_length_m <= 0.0:
            raise ValueError("maximum_technical_turn_length_m must be positive")
        if self.maximum_technical_turn_radius_m <= self.envelope_half_width_m:
            raise ValueError("maximum_technical_turn_radius_m must clear the track envelope")
        shortest_target_length_m = self.minimum_length_m if self.target_length_m is None else self.target_length_m
        if shortest_target_length_m <= 2.0 * pi * self.maximum_technical_turn_radius_m:
            raise ValueError("track length must leave room for the technical turn radius")
        if self.minimum_bend_radius_m <= self.envelope_half_width_m:
            raise ValueError("minimum_bend_radius_m must clear the track envelope")
        if self.minimum_nonlocal_clearance_m <= 0.0:
            raise ValueError("minimum_nonlocal_clearance_m must be positive")
        if self.minimum_nonlocal_clearance_m <= 2 * self.envelope_half_width_m:
            raise ValueError("minimum_nonlocal_clearance_m must clear both sides of the track envelope")
        if not 0.0 <= self.minimum_reflection_asymmetry <= 1.0:
            raise ValueError("minimum_reflection_asymmetry must be between 0 and 1")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least 1")

    @property
    def minimum_length_m(self) -> float:
        """Return the shortest accepted generated lap."""
        return self.reference_length_m * self.minimum_length_ratio

    @property
    def maximum_length_m(self) -> float:
        """Return the longest accepted generated lap."""
        return self.reference_length_m * self.maximum_length_ratio


DEFAULT_PROCEDURAL_TRACK_CONFIG = ProceduralTrackConfig()


def generate_procedural_track(
    seed: int,
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> TrackLayout:
    """Generate the first valid closed course in a deterministic seed sequence."""
    target_length_m = procedural_track_length_for_seed(seed, config=config)
    failure_reason = "no candidate was generated"
    for attempt in range(config.maximum_attempts):
        try:
            centerline_points = _candidate_centerline_points(
                seed=seed,
                attempt=attempt,
                target_length_m=target_length_m,
                config=config,
            )
        except ValueError as error:
            failure_reason = str(error)
            continue
        scale = target_length_m / total_track_length(centerline_points)
        scaled_points = tuple(TrackPoint(point.x * scale, point.z * scale) for point in centerline_points)
        samples = resample_closed_path(scaled_points, spacing_m=config.sample_spacing_m)
        resample_scale = target_length_m / total_track_length(samples)
        samples = tuple(TrackPoint(point.x * resample_scale, point.z * resample_scale) for point in samples)
        failure_reason = procedural_track_validation_error(samples, config=config) or ""
        if not failure_reason:
            labeled_samples = (
                TrackPoint(samples[0].x, samples[0].z, "Generated start"),
                *samples[1:],
            )
            return TrackLayout(
                track_id=f"{TRACK_ID_PROCEDURAL}-{seed}",
                points=labeled_samples,
                start_position=labeled_samples[0],
            )
    raise ValueError(
        f"could not generate a valid procedural track for seed {seed} "
        f"after {config.maximum_attempts} attempts: {failure_reason}"
    )


def procedural_track_length_for_seed(
    seed: int,
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> float:
    """Choose a deterministic lap length inside the configured Mugello-relative range."""
    if config.target_length_m is not None:
        return config.target_length_m
    rng = Random(seed * 2_000_033 + 71_339)
    return rng.uniform(config.minimum_length_m, config.maximum_length_m)


def resample_closed_path(
    points: tuple[TrackPoint, ...],
    *,
    spacing_m: float,
) -> tuple[TrackPoint, ...]:
    """Resample a closed path into nearly equal traveled-distance segments."""
    if len(points) < 3:
        raise ValueError("a closed path needs at least three points")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")

    segment_lengths = tuple(
        hypot(points[(index + 1) % len(points)].x - point.x, points[(index + 1) % len(points)].z - point.z)
        for index, point in enumerate(points)
    )
    if any(length <= 0.0 for length in segment_lengths):
        raise ValueError("a closed path cannot contain zero-length segments")
    total_length = sum(segment_lengths)
    sample_count = max(3, round(total_length / spacing_m))
    step = total_length / sample_count

    samples: list[TrackPoint] = []
    segment_index = 0
    segment_start_distance = 0.0
    for sample_index in range(sample_count):
        target_distance = sample_index * step
        while segment_start_distance + segment_lengths[segment_index] < target_distance:
            segment_start_distance += segment_lengths[segment_index]
            segment_index = (segment_index + 1) % len(points)
        start = points[segment_index]
        end = points[(segment_index + 1) % len(points)]
        fraction = (target_distance - segment_start_distance) / segment_lengths[segment_index]
        samples.append(
            TrackPoint(
                x=start.x + (end.x - start.x) * fraction,
                z=start.z + (end.z - start.z) * fraction,
            )
        )
    return tuple(samples)


def procedural_track_validation_error(
    samples: tuple[TrackPoint, ...],
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> str | None:
    """Return a useful rejection reason, or ``None`` for a valid generated path."""
    if len(samples) < 12:
        return "track needs at least 12 samples"
    if any(not isfinite(point.x) or not isfinite(point.z) for point in samples):
        return "track contains non-finite coordinates"

    segment_lengths = tuple(
        hypot(samples[(index + 1) % len(samples)].x - point.x, samples[(index + 1) % len(samples)].z - point.z)
        for index, point in enumerate(samples)
    )
    if min(segment_lengths) <= 1e-6:
        return "track contains a zero-length segment"
    if max(segment_lengths) - min(segment_lengths) > config.sample_spacing_m * 0.08:
        return "track samples are not uniformly spaced"
    lap_length_m = sum(segment_lengths)
    if not config.minimum_length_m <= lap_length_m <= config.maximum_length_m:
        return f"lap length {lap_length_m:.2f}m is outside {config.minimum_length_m:.2f}-{config.maximum_length_m:.2f}m"
    if _closed_path_self_intersects(samples):
        return "track centerline intersects itself"

    minimum_radius = _minimum_bend_radius(samples)
    if minimum_radius < config.minimum_bend_radius_m:
        return f"minimum bend radius {minimum_radius:.2f}m is too tight"
    minimum_clearance = _minimum_nonlocal_clearance(
        samples,
        local_distance_m=max(config.minimum_nonlocal_clearance_m * 1.5, config.minimum_bend_radius_m),
    )
    if minimum_clearance < config.minimum_nonlocal_clearance_m:
        return f"nonlocal centerline clearance {minimum_clearance:.2f}m is too small"

    for offset in (-config.envelope_half_width_m, config.envelope_half_width_m):
        offset_points = _offset_closed_path(samples, offset=offset)
        if _closed_path_self_intersects(offset_points):
            return f"track envelope at offset {offset:.2f}m intersects itself"
    asymmetry = reflection_asymmetry_score(samples)
    if asymmetry < config.minimum_reflection_asymmetry:
        return f"reflection asymmetry {asymmetry:.3f} is below {config.minimum_reflection_asymmetry:.3f}"
    longest_straight_m = longest_straight_length_m(samples, config=config)
    if longest_straight_m < config.minimum_straight_length_m:
        return f"longest straight {longest_straight_m:.2f}m is too short"
    hairpin_turn_degrees = maximum_hairpin_turn_degrees(samples, config=config)
    if hairpin_turn_degrees < config.minimum_hairpin_turn_degrees:
        return f"tightest turn {hairpin_turn_degrees:.1f} degrees is not a hairpin"
    measured_hairpin_count = hairpin_count(samples, config=config)
    if not config.minimum_hairpin_count <= measured_hairpin_count <= config.maximum_hairpin_count:
        return (
            f"hairpin count {measured_hairpin_count} is outside "
            f"{config.minimum_hairpin_count}-{config.maximum_hairpin_count}"
        )
    technical_turn_radius_m = minimum_coherent_turn_radius_m(samples, config=config)
    if technical_turn_radius_m > config.maximum_technical_turn_radius_m:
        return (
            f"tightest coherent turn radius {technical_turn_radius_m:.2f}m "
            f"exceeds {config.maximum_technical_turn_radius_m:.2f}m"
        )
    return None


def longest_straight_length_m(
    samples: tuple[TrackPoint, ...],
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> float:
    """Measure the longest low-deviation chord along a generated centerline."""
    if len(samples) < 3:
        return 0.0
    segment_lengths = tuple(
        hypot(samples[(index + 1) % len(samples)].x - point.x, samples[(index + 1) % len(samples)].z - point.z)
        for index, point in enumerate(samples)
    )
    best_length_m = 0.0
    maximum_segment_count = max(2, len(samples) // 3)
    for start_index, start in enumerate(samples):
        arc_length_m = 0.0
        for segment_count in range(1, maximum_segment_count + 1):
            segment_index = (start_index + segment_count - 1) % len(samples)
            arc_length_m += segment_lengths[segment_index]
            end = samples[(start_index + segment_count) % len(samples)]
            chord_length_m = hypot(end.x - start.x, end.z - start.z)
            if chord_length_m <= 0.0 or chord_length_m / arc_length_m < 0.995:
                continue
            maximum_deviation_m = (
                max(
                    _distance_to_line(samples[(start_index + offset) % len(samples)], start=start, end=end)
                    for offset in range(1, segment_count)
                )
                if segment_count > 1
                else 0.0
            )
            if maximum_deviation_m <= config.straight_lateral_tolerance_m:
                best_length_m = max(best_length_m, arc_length_m)
    return best_length_m


def maximum_hairpin_turn_degrees(
    samples: tuple[TrackPoint, ...],
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> float:
    """Measure the largest coherent heading reversal within the hairpin distance budget."""
    if len(samples) < 3:
        return 0.0
    segment_lengths = tuple(
        hypot(samples[(index + 1) % len(samples)].x - point.x, samples[(index + 1) % len(samples)].z - point.z)
        for index, point in enumerate(samples)
    )
    headings = tuple(
        degrees(
            atan2(
                samples[(index + 1) % len(samples)].x - point.x,
                samples[(index + 1) % len(samples)].z - point.z,
            )
        )
        for index, point in enumerate(samples)
    )
    best_turn_degrees = 0.0
    for start_index in range(len(samples)):
        path_length_m = 0.0
        signed_turn_degrees = 0.0
        absolute_turn_degrees = 0.0
        for offset in range(len(samples) - 1):
            segment_index = (start_index + offset) % len(samples)
            path_length_m += segment_lengths[segment_index]
            if path_length_m > config.maximum_hairpin_length_m:
                break
            next_segment_index = (segment_index + 1) % len(samples)
            heading_delta = _heading_delta_degrees(headings[segment_index], headings[next_segment_index])
            signed_turn_degrees += heading_delta
            absolute_turn_degrees += abs(heading_delta)
            if absolute_turn_degrees <= 0.0:
                continue
            coherence = abs(signed_turn_degrees) / absolute_turn_degrees
            if coherence >= config.minimum_hairpin_turn_coherence:
                best_turn_degrees = max(best_turn_degrees, abs(signed_turn_degrees))
    return best_turn_degrees


def hairpin_count(
    samples: tuple[TrackPoint, ...],
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> int:
    """Count distinct coherent turning regions that meet the hairpin threshold."""
    if len(samples) < 3:
        return 0
    segment_lengths = tuple(
        hypot(samples[(index + 1) % len(samples)].x - point.x, samples[(index + 1) % len(samples)].z - point.z)
        for index, point in enumerate(samples)
    )
    headings = tuple(
        degrees(
            atan2(
                samples[(index + 1) % len(samples)].x - point.x,
                samples[(index + 1) % len(samples)].z - point.z,
            )
        )
        for index, point in enumerate(samples)
    )
    heading_deltas = tuple(
        _heading_delta_degrees(headings[index], headings[(index + 1) % len(headings)]) for index in range(len(headings))
    )
    significant_turns = tuple((index, delta) for index, delta in enumerate(heading_deltas) if abs(delta) >= 0.1)
    if not significant_turns:
        return 0

    first_sign_change = next(
        (index for index, (_, delta) in enumerate(significant_turns) if delta * significant_turns[index - 1][1] < 0.0),
        0,
    )
    ordered_turns = significant_turns[first_sign_change:] + significant_turns[:first_sign_change]
    regions: list[list[tuple[int, float]]] = []
    for sample_index, delta in ordered_turns:
        if regions and delta * regions[-1][-1][1] > 0.0:
            regions[-1].append((sample_index, delta))
        else:
            regions.append([(sample_index, delta)])

    count = 0
    for region in regions:
        turn_degrees = abs(sum(delta for _, delta in region))
        first_index = region[0][0]
        last_index = region[-1][0]
        segment_count = (last_index - first_index) % len(samples) + 1
        region_length_m = sum(segment_lengths[(first_index + offset) % len(samples)] for offset in range(segment_count))
        if turn_degrees >= config.minimum_hairpin_turn_degrees and region_length_m <= config.maximum_hairpin_length_m:
            count += 1
    return count


def minimum_coherent_turn_radius_m(
    samples: tuple[TrackPoint, ...],
    *,
    config: ProceduralTrackConfig = DEFAULT_PROCEDURAL_TRACK_CONFIG,
) -> float:
    """Estimate the tightest sustained corner from arc length and heading change."""
    if len(samples) < 3:
        return float("inf")
    segment_lengths = tuple(
        hypot(samples[(index + 1) % len(samples)].x - point.x, samples[(index + 1) % len(samples)].z - point.z)
        for index, point in enumerate(samples)
    )
    headings = tuple(
        degrees(
            atan2(
                samples[(index + 1) % len(samples)].x - point.x,
                samples[(index + 1) % len(samples)].z - point.z,
            )
        )
        for index, point in enumerate(samples)
    )
    minimum_radius_m = float("inf")
    for start_index in range(len(samples)):
        path_length_m = 0.0
        signed_turn_degrees = 0.0
        absolute_turn_degrees = 0.0
        for offset in range(len(samples) - 1):
            segment_index = (start_index + offset) % len(samples)
            path_length_m += segment_lengths[segment_index]
            if path_length_m > config.maximum_technical_turn_length_m:
                break
            next_segment_index = (segment_index + 1) % len(samples)
            heading_delta = _heading_delta_degrees(headings[segment_index], headings[next_segment_index])
            signed_turn_degrees += heading_delta
            absolute_turn_degrees += abs(heading_delta)
            if abs(signed_turn_degrees) < config.minimum_technical_turn_degrees:
                continue
            coherence = abs(signed_turn_degrees) / absolute_turn_degrees
            if coherence >= config.minimum_hairpin_turn_coherence:
                radius_m = path_length_m / (abs(signed_turn_degrees) * pi / 180.0)
                minimum_radius_m = min(minimum_radius_m, radius_m)
    return minimum_radius_m


def reflection_asymmetry_score(samples: tuple[TrackPoint, ...]) -> float:
    """Return normalized error from the outline's closest mirror reflection."""
    if len(samples) < 3:
        return 0.0
    center_x = sum(point.x for point in samples) / len(samples)
    center_z = sum(point.z for point in samples) / len(samples)
    centered = tuple(complex(point.x - center_x, point.z - center_z) for point in samples)
    scale = sqrt(sum(abs(point) ** 2 for point in centered) / len(centered))
    if scale <= 0.0:
        return 0.0
    normalized = tuple(point / scale for point in centered)
    best_error = float("inf")
    for shift in range(len(normalized)):
        reflected = tuple(normalized[(shift - index) % len(normalized)].conjugate() for index in range(len(normalized)))
        cross = sum(normalized[index].conjugate() * reflected[index] for index in range(len(normalized)))
        rotation = cross / abs(cross) if cross else complex(1.0)
        error = sqrt(
            sum(abs(normalized[index] * rotation - reflected[index]) ** 2 for index in range(len(normalized)))
            / len(normalized)
        )
        best_error = min(best_error, error)
    return best_error


def _distance_to_line(point: TrackPoint, *, start: TrackPoint, end: TrackPoint) -> float:
    chord_length_m = hypot(end.x - start.x, end.z - start.z)
    if chord_length_m <= 0.0:
        return float("inf")
    return abs((end.x - start.x) * (start.z - point.z) - (start.x - point.x) * (end.z - start.z)) / chord_length_m


def _heading_delta_degrees(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


def _candidate_centerline_points(
    *,
    seed: int,
    attempt: int,
    target_length_m: float,
    config: ProceduralTrackConfig,
) -> tuple[TrackPoint, ...]:
    rng = Random(seed * 1_000_003 + attempt * 97_409 + 11_027)
    base_hairpin_turn_radians = _generated_hairpin_turn_radians(config)
    hairpin_total = _generated_hairpin_count(
        target_length_m=target_length_m,
        hairpin_turn_radians=base_hairpin_turn_radians,
        config=config,
    )
    variation = min(1.0, config.shape_variation / 0.20)
    sector_turns = _asymmetric_sector_turns(
        hairpin_total=hairpin_total,
        rng=rng,
        variation=variation,
    )
    minimum_generated_turn_radians = (config.minimum_hairpin_turn_degrees + 3.0) * pi / 180.0
    hairpin_turns = tuple(
        min(
            170.0 * pi / 180.0,
            max(
                minimum_generated_turn_radians,
                base_hairpin_turn_radians + rng.uniform(-2.0, 4.0) * pi / 180.0 * variation,
            ),
        )
        for _ in range(hairpin_total)
    )
    recovery_turns = tuple(hairpin_turns[index] - sector_turns[index] for index in range(hairpin_total))
    if min(recovery_turns) <= 0.0:
        raise ValueError("seeded sector angles cannot form coherent counter-turns")

    minimum_feature_radius_m = max(
        config.minimum_bend_radius_m + 0.5,
        config.envelope_half_width_m + 0.75,
    )
    minimum_arc_length_m = sum(
        minimum_feature_radius_m * (hairpin_turns[index] + recovery_turns[index]) for index in range(hairpin_total)
    )
    straight_total_floor_m = config.minimum_straight_length_m + (2 * hairpin_total - 1) * 0.25
    extra_length_m = target_length_m - minimum_arc_length_m - straight_total_floor_m
    if extra_length_m < 0.0:
        raise ValueError("seeded hairpin angles exceed the lap-length budget")

    radius_budget_m = extra_length_m * rng.uniform(0.14, 0.32) * variation
    arc_weights = tuple(rng.uniform(0.70, 1.30) for _ in range(2 * hairpin_total))
    arc_weight_total = sum(arc_weights)
    hairpin_radii = tuple(
        minimum_feature_radius_m + radius_budget_m * arc_weights[2 * index] / arc_weight_total / hairpin_turns[index]
        for index in range(hairpin_total)
    )
    recovery_radii = tuple(
        minimum_feature_radius_m
        + radius_budget_m * arc_weights[2 * index + 1] / arc_weight_total / recovery_turns[index]
        for index in range(hairpin_total)
    )
    straight_total_m = target_length_m - sum(
        hairpin_turns[index] * hairpin_radii[index] + recovery_turns[index] * recovery_radii[index]
        for index in range(hairpin_total)
    )

    heading_radians = 0.0
    fixed_position_x = 0.0
    fixed_position_z = 0.0
    straight_headings: list[float] = []
    for index in range(hairpin_total):
        straight_headings.append(heading_radians)
        fixed_position_x, fixed_position_z, heading_radians = _arc_endpoint(
            x=fixed_position_x,
            z=fixed_position_z,
            heading_radians=heading_radians,
            turn_radians=hairpin_turns[index],
            radius_m=hairpin_radii[index],
        )
        straight_headings.append(heading_radians)
        fixed_position_x, fixed_position_z, heading_radians = _arc_endpoint(
            x=fixed_position_x,
            z=fixed_position_z,
            heading_radians=heading_radians,
            turn_radians=-recovery_turns[index],
            radius_m=recovery_radii[index],
        )

    desired_straight_weights = _straight_weights_for_seed(
        seed=seed,
        rng=rng,
        count=len(straight_headings),
    )
    desired_weight_total = sum(desired_straight_weights)
    desired_straights = tuple(straight_total_m * weight / desired_weight_total for weight in desired_straight_weights)
    straight_lengths = _closed_straight_lengths(
        headings=tuple(straight_headings),
        desired_lengths=desired_straights,
        total_length_m=straight_total_m,
        fixed_displacement=(-fixed_position_x, -fixed_position_z),
        minimum_length_m=0.25,
    )
    if max(straight_lengths) < config.minimum_straight_length_m:
        raise ValueError("closure cannot preserve the required straight")

    rotation = rng.uniform(-pi, pi)
    raw_sample_count = config.control_point_count * config.samples_per_control_point
    points: list[TrackPoint] = []
    position_x = 0.0
    position_z = 0.0
    heading_radians = 0.0

    def append_straight(length_m: float) -> None:
        nonlocal position_x, position_z
        sample_count = max(2, round(raw_sample_count * length_m / target_length_m))
        for index in range(sample_count):
            fraction = index / sample_count
            points.append(
                TrackPoint(
                    position_x + fraction * length_m * cos(heading_radians),
                    position_z + fraction * length_m * sin(heading_radians),
                )
            )
        position_x += length_m * cos(heading_radians)
        position_z += length_m * sin(heading_radians)

    def append_arc(turn_radians: float, radius_m: float) -> None:
        nonlocal position_x, position_z, heading_radians
        arc_length_m = abs(turn_radians) * radius_m
        sample_count = max(4, round(raw_sample_count * arc_length_m / target_length_m))
        turn_sign = 1.0 if turn_radians > 0.0 else -1.0
        center_x = position_x - turn_sign * radius_m * sin(heading_radians)
        center_z = position_z + turn_sign * radius_m * cos(heading_radians)
        start_angle = heading_radians - turn_sign * pi / 2.0
        for index in range(sample_count):
            angle = start_angle + turn_radians * index / sample_count
            points.append(
                TrackPoint(
                    center_x + radius_m * cos(angle),
                    center_z + radius_m * sin(angle),
                )
            )
        end_angle = start_angle + turn_radians
        position_x = center_x + radius_m * cos(end_angle)
        position_z = center_z + radius_m * sin(end_angle)
        heading_radians += turn_radians

    # Unequal angular sectors cluster corners onto one side of the course. Solving
    # both the approach and exit straights leaves enough freedom for a dominant
    # straight, an offset back section, or a dense technical complex.
    for index in range(hairpin_total):
        append_straight(straight_lengths[2 * index])
        append_arc(hairpin_turns[index], hairpin_radii[index])
        append_straight(straight_lengths[2 * index + 1])
        append_arc(-recovery_turns[index], recovery_radii[index])

    return tuple(
        TrackPoint(
            point.x * cos(rotation) - point.z * sin(rotation),
            point.x * sin(rotation) + point.z * cos(rotation),
        )
        for point in points
    )


def _arc_endpoint(
    *,
    x: float,
    z: float,
    heading_radians: float,
    turn_radians: float,
    radius_m: float,
) -> tuple[float, float, float]:
    turn_sign = 1.0 if turn_radians > 0.0 else -1.0
    center_x = x - turn_sign * radius_m * sin(heading_radians)
    center_z = z + turn_sign * radius_m * cos(heading_radians)
    end_angle = heading_radians - turn_sign * pi / 2.0 + turn_radians
    return (
        center_x + radius_m * cos(end_angle),
        center_z + radius_m * sin(end_angle),
        heading_radians + turn_radians,
    )


def _asymmetric_sector_turns(*, hairpin_total: int, rng: Random, variation: float) -> tuple[float, ...]:
    """Allocate the lap's heading change unevenly so corners form clusters."""
    profiles_degrees = {
        3: ((151.0, 134.0, 75.0), (149.0, 126.0, 85.0), (145.0, 140.0, 75.0)),
        4: ((150.0, 120.0, 65.0, 25.0), (148.0, 132.0, 60.0, 20.0), (145.0, 110.0, 80.0, 25.0)),
        5: (
            (120.0, 90.0, 65.0, 50.0, 35.0),
            (130.0, 85.0, 60.0, 50.0, 35.0),
            (110.0, 100.0, 75.0, 50.0, 25.0),
        ),
    }
    try:
        profile = profiles_degrees[hairpin_total][rng.randrange(3)]
    except KeyError as error:
        raise ValueError(f"no asymmetric sector profile for {hairpin_total} hairpins") from error
    jittered = [angle * (1.0 + rng.uniform(-0.08, 0.08) * variation) for angle in profile]
    scale = 360.0 / sum(jittered)
    sector_turns = [angle * scale * pi / 180.0 for angle in jittered]
    rotation = rng.randrange(hairpin_total)
    if rng.random() < 0.5:
        sector_turns.reverse()
    return tuple(sector_turns[rotation:] + sector_turns[:rotation])


def _straight_weights_for_seed(*, seed: int, rng: Random, count: int) -> tuple[float, ...]:
    weights = [rng.uniform(0.35, 1.65) for _ in range(count)]
    family = abs(seed) % 3
    if family == 0:
        weights[0] *= 5.0
        weights[count // 2] *= 0.35
    elif family == 1:
        weights[0] *= 3.5
        weights[max(1, count // 2)] *= 1.4
        weights[-1] *= 0.35
    else:
        for index in range(count):
            weights[index] *= (2.8, 0.45, 1.25, 0.65)[index % 4]
    return tuple(weights)


def _closed_straight_lengths(
    *,
    headings: tuple[float, ...],
    desired_lengths: tuple[float, ...],
    total_length_m: float,
    fixed_displacement: tuple[float, float],
    minimum_length_m: float = 0.0,
) -> tuple[float, ...]:
    constraints = (
        tuple(cos(heading) for heading in headings),
        tuple(sin(heading) for heading in headings),
        tuple(1.0 for _ in headings),
    )
    targets = (fixed_displacement[0], fixed_displacement[1], total_length_m)
    fixed_lengths: dict[int, float] = {}
    while len(fixed_lengths) <= len(headings) - 3:
        free_indices = tuple(index for index in range(len(headings)) if index not in fixed_lengths)
        adjusted_targets = tuple(
            targets[row] - sum(constraints[row][index] * length_m for index, length_m in fixed_lengths.items())
            for row in range(3)
        )
        gram_matrix = tuple(
            tuple(
                sum(constraints[row][index] * constraints[column][index] for index in free_indices)
                for column in range(3)
            )
            for row in range(3)
        )
        residuals = tuple(
            adjusted_targets[row] - sum(constraints[row][index] * desired_lengths[index] for index in free_indices)
            for row in range(3)
        )
        multipliers = _solve_three_by_three(gram_matrix, residuals)
        solved = {
            index: desired_lengths[index] + sum(constraints[row][index] * multipliers[row] for row in range(3))
            for index in free_indices
        }
        violations = tuple(
            (length_m - minimum_length_m, index) for index, length_m in solved.items() if length_m < minimum_length_m
        )
        if not violations:
            return tuple(
                fixed_lengths[index] if index in fixed_lengths else solved[index] for index in range(len(headings))
            )
        _, worst_index = min(violations)
        fixed_lengths[worst_index] = minimum_length_m
    raise ValueError(f"closure cannot keep every straight at least {minimum_length_m:.1f}m")


def _solve_three_by_three(
    matrix: tuple[tuple[float, ...], ...],
    values: tuple[float, ...],
) -> tuple[float, float, float]:
    augmented = [[*row, values[index]] for index, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-9:
            raise ValueError("seeded closure constraints are singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [augmented[row][index] - factor * augmented[column][index] for index in range(4)]
    return (augmented[0][3], augmented[1][3], augmented[2][3])


def _generated_hairpin_turn_radians(config: ProceduralTrackConfig) -> float:
    validation_margin_degrees = min(
        7.0,
        max(3.0, (180.0 - config.minimum_hairpin_turn_degrees) * 0.18),
    )
    return min(179.0, config.minimum_hairpin_turn_degrees + validation_margin_degrees) * pi / 180.0


def _generated_hairpin_count(
    *,
    target_length_m: float,
    hairpin_turn_radians: float,
    config: ProceduralTrackConfig,
) -> int:
    minimum_feature_radius_m = max(
        config.minimum_bend_radius_m + 0.5,
        config.envelope_half_width_m + 0.75,
    )
    feasible_counts = tuple(
        count
        for count in range(config.minimum_hairpin_count, config.maximum_hairpin_count + 1)
        if hairpin_turn_radians > 2.0 * pi / count
        and count
        * (
            config.minimum_straight_length_m
            + minimum_feature_radius_m * (hairpin_turn_radians + hairpin_turn_radians - 2.0 * pi / count)
        )
        <= target_length_m
    )
    if not feasible_counts:
        raise ValueError(f"track length {target_length_m:.2f}m cannot fit the configured hairpin count")
    return max(feasible_counts)


def _minimum_bend_radius(points: tuple[TrackPoint, ...]) -> float:
    minimum_radius = float("inf")
    for index, middle in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        a = hypot(middle.x - previous.x, middle.z - previous.z)
        b = hypot(following.x - middle.x, following.z - middle.z)
        c = hypot(previous.x - following.x, previous.z - following.z)
        twice_area = abs(
            (middle.x - previous.x) * (following.z - previous.z) - (middle.z - previous.z) * (following.x - previous.x)
        )
        if twice_area > 1e-9:
            minimum_radius = min(minimum_radius, a * b * c / (2.0 * twice_area))
    return minimum_radius


def _minimum_nonlocal_clearance(points: tuple[TrackPoint, ...], *, local_distance_m: float) -> float:
    average_spacing = total_track_length(points) / len(points)
    local_segment_count = max(1, ceil(local_distance_m / average_spacing))
    minimum_clearance = float("inf")
    for first_index, first in enumerate(points):
        for second_index in range(first_index + 1, len(points)):
            index_distance = min(second_index - first_index, len(points) - (second_index - first_index))
            if index_distance <= local_segment_count:
                continue
            second = points[second_index]
            minimum_clearance = min(minimum_clearance, hypot(second.x - first.x, second.z - first.z))
    return minimum_clearance


def _offset_closed_path(points: tuple[TrackPoint, ...], *, offset: float) -> tuple[TrackPoint, ...]:
    offset_points: list[TrackPoint] = []
    for index, point in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        tangent_x = following.x - previous.x
        tangent_z = following.z - previous.z
        tangent_length = hypot(tangent_x, tangent_z)
        offset_points.append(
            TrackPoint(
                x=point.x - tangent_z / tangent_length * offset,
                z=point.z + tangent_x / tangent_length * offset,
            )
        )
    return tuple(offset_points)


def _closed_path_self_intersects(points: tuple[TrackPoint, ...]) -> bool:
    for first_index, first_start in enumerate(points):
        first_end = points[(first_index + 1) % len(points)]
        for second_index in range(first_index + 1, len(points)):
            if second_index in (first_index, (first_index + 1) % len(points)):
                continue
            if first_index == 0 and second_index == len(points) - 1:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % len(points)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def _segments_intersect(first: TrackPoint, second: TrackPoint, third: TrackPoint, fourth: TrackPoint) -> bool:
    first_side = _cross(first, second, third)
    second_side = _cross(first, second, fourth)
    third_side = _cross(third, fourth, first)
    fourth_side = _cross(third, fourth, second)
    return first_side * second_side < -1e-12 and third_side * fourth_side < -1e-12


def _cross(origin: TrackPoint, first: TrackPoint, second: TrackPoint) -> float:
    return (first.x - origin.x) * (second.z - origin.z) - (first.z - origin.z) * (second.x - origin.x)
