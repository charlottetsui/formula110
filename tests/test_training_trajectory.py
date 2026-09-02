from __future__ import annotations

import pytest

from training.trajectory import BestTrajectoryTracker


def test_bonus_equals_raw_progress_before_any_record_exists() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)

    bonus = tracker.bonus_m(previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=0.5)

    assert bonus == 0.5  # no record yet, so best_delta is 0 -- bonus is just the raw distance gained


def test_update_then_matching_the_record_gives_zero_bonus() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)
    tracker.update(tick=0, distance_m=0.0)
    tracker.update(tick=1, distance_m=1.0)

    bonus = tracker.bonus_m(previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.0)

    assert bonus == 0.0


def test_beating_the_record_gives_a_positive_bonus() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)
    tracker.update(tick=0, distance_m=0.0)
    tracker.update(tick=1, distance_m=1.0)

    bonus = tracker.bonus_m(previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.5)

    assert bonus == pytest.approx(0.5)


def test_falling_behind_the_record_gives_a_negative_bonus() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)
    tracker.update(tick=0, distance_m=0.0)
    tracker.update(tick=1, distance_m=1.0)

    bonus = tracker.bonus_m(previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=0.5)

    assert bonus == pytest.approx(-0.5)


def test_update_only_raises_the_record_never_lowers_it() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)
    tracker.update(tick=5, distance_m=10.0)
    tracker.update(tick=5, distance_m=3.0)  # a worse run at the same tick

    bonus = tracker.bonus_m(previous_tick=4, previous_distance_m=9.0, current_tick=5, current_distance_m=10.0)

    # best_delta at tick 5 is still relative to the 10.0 record, not the worse 3.0 update
    assert bonus == pytest.approx(1.0 - 10.0)


def test_ticks_beyond_max_ticks_are_clamped_to_the_last_slot() -> None:
    tracker = BestTrajectoryTracker(max_ticks=10)
    tracker.update(tick=9, distance_m=5.0)

    bonus_exactly_at_boundary = tracker.bonus_m(
        previous_tick=9, previous_distance_m=5.0, current_tick=9, current_distance_m=6.0
    )
    bonus_beyond_boundary = tracker.bonus_m(
        previous_tick=9, previous_distance_m=5.0, current_tick=50, current_distance_m=6.0
    )

    # tick 9 and tick 50 both clamp to the same last slot, so both see the same record
    assert bonus_exactly_at_boundary == bonus_beyond_boundary == pytest.approx(1.0)


def test_max_ticks_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_ticks"):
        BestTrajectoryTracker(max_ticks=0)
