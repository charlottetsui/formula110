"""Reproducible layout families for training, distinct from spawn seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi, sin
from random import Random

from racing.track.world import MUGELLO_SHORT_LAYOUT, TrackPoint, sampled_track_centerline


@dataclass(frozen=True)
class Scenario:
    layout_seed: int
    spawn_seed: int
    family: str = "radial"
    seconds: float = 40.0

    def samples(self) -> tuple[TrackPoint, ...] | None:
        if self.family == "default":
            return None
        rng = Random(self.layout_seed)
        if self.family == "stretch":
            sx, sz = rng.uniform(0.9, 1.5), rng.uniform(0.9, 1.6)
            mirror = -1 if self.layout_seed % 2 else 1
            points = tuple(TrackPoint(p.x * sx * mirror, p.z * sz) for p in MUGELLO_SHORT_LAYOUT)
        elif self.family == "radial":
            # Smooth star-shaped circuits: new curvature/straight lengths,
            # clockwise and counterclockwise. No intersections in control polygon.
            rx, rz = rng.uniform(30, 55), rng.uniform(23, 42)
            amplitude, phase = rng.uniform(0.04, 0.16), rng.uniform(0, 2 * pi)
            lobes = rng.choice([2, 3, 4])
            points = tuple(
                TrackPoint(
                    rx * (1 + amplitude * cos(lobes * a + phase)) * cos(a),
                    rz * (1 + amplitude * cos(lobes * a + phase)) * sin(a),
                )
                for a in (2 * pi * i / 16 for i in range(16))
            )
            if self.layout_seed % 2:
                points = tuple(reversed(points))
        else:
            raise ValueError(f"Unknown layout family {self.family}")
        samples = sampled_track_centerline(points)
        validate_centerline(samples)
        return samples

    def metadata(self) -> dict[str, object]:
        samples = self.samples()
        return {**asdict(self), "samples": None if samples is None else [[p.x, p.z] for p in samples]}


def validate_centerline(points: tuple[TrackPoint, ...]) -> None:
    """Reject crossing centerlines; this is not a certification of wall clearance."""

    def orientation(a: TrackPoint, b: TrackPoint, c: TrackPoint) -> float:
        return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)

    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        for j in range(i + 2, len(points)):
            if i == 0 and j == len(points) - 1:
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            if orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0:
                raise ValueError("Generated centerline crosses itself")


def suite(split: str) -> list[Scenario]:
    """Entire layouts are held out, including every start on that layout."""
    if split == "train":
        return [
            Scenario(0, 3100, "default"),
            *[Scenario(i, 3100 + i, "radial" if i % 3 else "stretch") for i in range(10, 26)],
        ]
    if split == "validation":
        return [Scenario(i, 4100 + i, "radial" if i % 2 else "stretch", 45) for i in range(101, 105)]
    if split == "test":
        return [Scenario(i, 5100 + i, "radial" if i % 2 else "stretch", 60) for i in range(201, 207)]
    if split == "final_test":
        return [Scenario(i, 6100 + i, "radial" if i % 2 else "stretch", 60) for i in range(301, 307)]
    raise ValueError(split)
