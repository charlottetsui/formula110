from __future__ import annotations

from importlib import import_module
from math import cos, radians, sin
from typing import Any, cast

import pytest

from racing.graphics.track_mesh import clean_offset_path
from racing.graphics.track_rendering import (
    START_FINISH_ARGYLE_FLOOR_LENGTH,
    START_FINISH_ARGYLE_FLOOR_WIDTH,
    START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO,
    START_FINISH_BANNER_HEIGHT,
    START_FINISH_BANNER_OVERHANG,
    START_FINISH_BANNER_POLE_THICKNESS,
    START_FINISH_BANNER_POLE_WALL_CLEARANCE,
    START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO,
    START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION,
    TRACK_EDGE_BUFFER,
    TRACK_KERB_INNER_DISTANCE,
    TRACK_KERB_OUTER_DISTANCE,
    TRACK_LIGHT_NOMINAL_SPACING_M,
    TRACK_LIGHT_SIDE_DISTANCE,
    TRACK_SURFACE_Y,
    TRACK_WALL_THICKNESS,
    add_racing_scene_collisions,
    start_finish_banner_side_distances,
    start_finish_render_pose,
    start_finish_track_slice,
    track_light_layouts,
)
from racing.physics import create_physics_world
from racing.race.progress import default_track_progress_model, project_track_position, resolve_track
from racing.race.runtime import seeded_race_start_finish_pose
from racing.track.procedural import TRACK_ID_PROCEDURAL
from racing.track.world import TRACK_WIDTH, TrackPoint, sampled_track_centerline


def test_kerb_center_rays_hit_flat_floor_collider() -> None:
    core = cast(Any, import_module("panda3d.core"))
    world = create_physics_world()
    render = core.NodePath("render")
    samples = sampled_track_centerline(samples_per_segment=10)
    kerb_center_distance = (TRACK_KERB_INNER_DISTANCE + TRACK_KERB_OUTER_DISTANCE) / 2.0

    add_racing_scene_collisions(physics_world=world, render=render, samples=samples)

    for side in (-1, 1):
        kerb_center_points = clean_offset_path(samples, side * kerb_center_distance, 0.0)
        for point in kerb_center_points[::25]:
            hit = world.rayTestClosest(
                core.Point3(point[0], 3.0, point[2]),
                core.Point3(point[0], -1.0, point[2]),
            )

            assert hit.hasHit()
            assert hit.getNode().getName() == "grass-and-track-floor"
            assert float(hit.getHitPos()[1]) == pytest.approx(TRACK_SURFACE_Y, abs=1e-3)
            assert float(hit.getHitNormal()[1]) == pytest.approx(1.0, abs=1e-5)


def test_problem_seed_start_finish_poles_clear_outside_wall() -> None:
    model = default_track_progress_model()
    samples = sampled_track_centerline(samples_per_segment=10)
    pole_radius = START_FINISH_BANNER_POLE_THICKNESS / 2
    required_clearance = pole_radius + START_FINISH_BANNER_POLE_WALL_CLEARANCE

    for seed in (2, 5, 6):
        pose = seeded_race_start_finish_pose(model=model, random_seed=seed, race_index=1)
        render_pose = start_finish_render_pose(samples=samples, position=pose.position)
        track_slice = start_finish_track_slice(samples=samples, position=pose.position)
        side_distances = start_finish_banner_side_distances(
            samples=samples,
            position=pose.position,
        )

        assert render_pose.position.x == pytest.approx(track_slice.pose.position.x)
        assert render_pose.position.z == pytest.approx(track_slice.pose.position.z)
        assert render_pose.heading_degrees == pytest.approx(track_slice.pose.heading_degrees)

        for pole_distance, wall_distance in zip(
            side_distances,
            (track_slice.negative_wall_distance, track_slice.positive_wall_distance),
            strict=True,
        ):
            assert pole_distance >= wall_distance + required_clearance - 1e-9


def test_start_finish_banner_is_tall_enough_for_formula_logo() -> None:
    wall_outside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS
    nominal_side_distance = (
        wall_outside_distance + START_FINISH_BANNER_POLE_THICKNESS / 2 + START_FINISH_BANNER_POLE_WALL_CLEARANCE
    )
    nominal_width = nominal_side_distance * 2 + START_FINISH_BANNER_OVERHANG
    available_height = START_FINISH_BANNER_HEIGHT * (1 - 2 * START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION)
    full_width_logo_height = nominal_width / START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO

    assert available_height >= full_width_logo_height


def test_start_finish_floor_argyle_matches_texture_aspect_ratio() -> None:
    assert pytest.approx(START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO) == (
        START_FINISH_ARGYLE_FLOOR_WIDTH / START_FINISH_ARGYLE_FLOOR_LENGTH
    )


def test_procedural_track_lights_use_equal_distance_intervals_and_alternate_sides() -> None:
    resolved = resolve_track(TRACK_ID_PROCEDURAL, 110)
    layouts = track_light_layouts(resolved.samples, legacy_layout=False)
    target_projections = tuple(
        project_track_position(resolved.model, TrackPoint(layout.target[0], layout.target[2])) for layout in layouts
    )
    progress_distances = tuple(projection.progress_distance_m for projection in target_projections)
    gaps = tuple(
        (progress_distances[(index + 1) % len(layouts)] - distance) % resolved.model.total_length_m
        for index, distance in enumerate(progress_distances)
    )
    side_vectors = tuple((layout.post[0] - layout.target[0], layout.post[2] - layout.target[2]) for layout in layouts)
    signed_sides = tuple(
        side_x * -cos(radians(projection.heading_degrees)) + side_z * sin(radians(projection.heading_degrees))
        for (side_x, side_z), projection in zip(side_vectors, target_projections, strict=True)
    )

    assert len(layouts) % 2 == 0
    assert sum(gaps) / len(gaps) == pytest.approx(TRACK_LIGHT_NOMINAL_SPACING_M, rel=0.08)
    assert max(gaps) - min(gaps) < 1e-6
    assert all((x * x + z * z) ** 0.5 == pytest.approx(TRACK_LIGHT_SIDE_DISTANCE) for x, z in side_vectors)
    assert all(
        first * second < 0.0
        for first, second in zip(
            signed_sides,
            (*signed_sides[1:], signed_sides[0]),
            strict=True,
        )
    )
