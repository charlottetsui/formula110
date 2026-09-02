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
"""

from __future__ import annotations

import numpy as np

# Same scale as WEIGHT_PROGRESS's meters-per-tick units in training.reward,
# so this bonus is directly comparable to (and additive with) the existing
# progress reward rather than needing its own separate calibration pass.
WEIGHT_TRAJECTORY_BONUS = 1.0


class BestTrajectoryTracker:
    """Shared record of the best cumulative distance reached at each tick."""

    def __init__(self, *, max_ticks: int) -> None:
        if max_ticks < 1:
            raise ValueError("max_ticks must be at least one")
        self._best_distance_m = np.zeros(max_ticks, dtype=np.float64)

    def bonus_m(
        self,
        *,
        previous_tick: int,
        previous_distance_m: float,
        current_tick: int,
        current_distance_m: float,
    ) -> float:
        """Return how much more (or less) ground was gained this tick vs. the best-known run."""
        previous_index = self._clamped_index(previous_tick)
        current_index = self._clamped_index(current_tick)
        distance_delta = current_distance_m - previous_distance_m
        best_delta = self._best_distance_m[current_index] - self._best_distance_m[previous_index]
        return distance_delta - best_delta

    def update(self, *, tick: int, distance_m: float) -> None:
        """Record `distance_m` as the new best if it beats the current record at `tick`."""
        index = self._clamped_index(tick)
        if distance_m > self._best_distance_m[index]:
            self._best_distance_m[index] = distance_m

    def _clamped_index(self, tick: int) -> int:
        return max(0, min(tick, len(self._best_distance_m) - 1))
