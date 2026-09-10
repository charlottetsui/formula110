"""Best-known-trajectory reward bonus: reward relative to your own past pace.

Every reward term in ``training.reward`` is a fixed, global constant applied
uniformly across the whole track (a speed cap, a wall-proximity threshold).
2026-09-02's causal tests (12-13) showed that tuning those constants keeps
producing the same failure shape: either uniformly more reckless or
uniformly more cautious everywhere, never "fast on the straights, careful
in the corners" -- because a single scalar has no way to express "here" vs.
"there." This module adds a *location/time-specific* signal instead: reward
for currently gaining ground faster than the best-known run ever has at
this same point in an episode, penalize for falling behind it. As training
accumulates faster (but still survived) episodes, the bar it's chasing
rises on its own -- a self-improving curriculum, not a fixed target.

There is no private track-position API (see docs/rl_design.md section 3),
so this uses ``odometry.distance_m`` indexed by ``sensors.tick`` as the
position proxy. Both reset to 0 for every fresh car/episode (SENSORS.md),
so they're directly comparable across different races and copies sharing
one tracker. A record at some tick can only exist if some earlier episode
actually survived to reach that tick -- a crashed run never writes records
past where it died, so the bonus can't reward chasing a pace nothing ever
survived at.

Causal test 14 (2026-09-02) found a real bug in a first version of this:
self-play runs two copies of the same policy in one race, controlled
sequentially within each physics tick, sharing one tracker. Reading and
writing the *live* shared record every tick meant a copy could be compared
against a "best" its own rival had just set in the same race, at the same
instant -- not a genuinely separate historical best -- producing a
systematic, adversarial corruption between the two copies (one grid
position starts ahead of the other) instead of the intended curriculum.
Fixed by decoupling reads from writes: each episode takes a frozen
``snapshot()`` of the tracker at its own first tick and reads bonuses from
that snapshot for its whole run, while ``update()`` still writes to the
live tracker so *future* episodes (which take their own snapshot at their
own start) benefit -- a same-race rival's mid-race progress can no longer
change what this episode is being compared against.
"""

from __future__ import annotations

import numpy as np

# Same scale as WEIGHT_PROGRESS's meters-per-tick units in training.reward,
# so this bonus is directly comparable to (and additive with) the existing
# progress reward rather than needing its own separate calibration pass.
WEIGHT_TRAJECTORY_BONUS = 1.0


def _clamped_index(tick: int, length: int) -> int:
    return max(0, min(tick, length - 1))


def _bonus_from_distances(
    distances: np.ndarray,
    *,
    previous_tick: int,
    previous_distance_m: float,
    current_tick: int,
    current_distance_m: float,
) -> float:
    previous_index = _clamped_index(previous_tick, len(distances))
    current_index = _clamped_index(current_tick, len(distances))
    distance_delta = current_distance_m - previous_distance_m
    best_delta = distances[current_index] - distances[previous_index]
    return float(distance_delta - best_delta)


def bonus_from_snapshot(
    snapshot: np.ndarray,
    *,
    previous_tick: int,
    previous_distance_m: float,
    current_tick: int,
    current_distance_m: float,
) -> float:
    """Return the trajectory bonus using a frozen `BestTrajectoryTracker.snapshot()`.

    Use this (not `BestTrajectoryTracker.bonus_m`) whenever a rival copy
    sharing the same tracker could be updating it mid-episode -- see the
    module docstring's account of the causal-test-14 bug this avoids.
    """
    return _bonus_from_distances(
        snapshot,
        previous_tick=previous_tick,
        previous_distance_m=previous_distance_m,
        current_tick=current_tick,
        current_distance_m=current_distance_m,
    )


class BestTrajectoryTracker:
    """Shared record of the best cumulative distance reached at each tick."""

    def __init__(self, *, max_ticks: int) -> None:
        if max_ticks < 1:
            raise ValueError("max_ticks must be at least one")
        self._best_distance_m = np.zeros(max_ticks, dtype=np.float64)

    def snapshot(self) -> np.ndarray:
        """Return a frozen copy of the current record, safe to read from across an episode."""
        return self._best_distance_m.copy()

    def bonus_m(
        self,
        *,
        previous_tick: int,
        previous_distance_m: float,
        current_tick: int,
        current_distance_m: float,
    ) -> float:
        """Return how much more (or less) ground was gained this tick vs. the *live* record.

        Prefer `bonus_from_snapshot` with a snapshot taken at episode start
        when multiple copies share this tracker concurrently (self-play) --
        this method reads the live, possibly-concurrently-updated record.
        """
        return _bonus_from_distances(
            self._best_distance_m,
            previous_tick=previous_tick,
            previous_distance_m=previous_distance_m,
            current_tick=current_tick,
            current_distance_m=current_distance_m,
        )

    def update(self, *, tick: int, distance_m: float) -> None:
        """Record `distance_m` as the new best if it beats the current record at `tick`."""
        index = _clamped_index(tick, len(self._best_distance_m))
        if distance_m > self._best_distance_m[index]:
            self._best_distance_m[index] = distance_m
