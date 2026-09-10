from __future__ import annotations

import pytest

from training.trajectory import BestTrajectoryTracker, bonus_from_snapshot


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


def test_snapshot_is_unaffected_by_later_updates() -> None:
    tracker = BestTrajectoryTracker(max_ticks=100)
    tracker.update(tick=1, distance_m=1.0)
    snapshot = tracker.snapshot()

    tracker.update(tick=1, distance_m=5.0)  # a later, faster run updates the live tracker

    bonus_from_live_tracker = tracker.bonus_m(
        previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.0
    )
    bonus_from_frozen_snapshot = bonus_from_snapshot(
        snapshot, previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.0
    )

    assert bonus_from_live_tracker == pytest.approx(1.0 - 5.0)  # sees the later update
    assert bonus_from_frozen_snapshot == pytest.approx(1.0 - 1.0)  # still sees the record as of the snapshot


def test_snapshot_prevents_same_race_rival_from_corrupting_this_episodes_bonus() -> None:
    # Regression test for the causal-test-14 bug: two self-play copies sharing one
    # tracker, controlled sequentially within the same tick, must not see each
    # other's mid-race progress as a "record" to be judged against.
    tracker = BestTrajectoryTracker(max_ticks=100)
    copy_a_snapshot = tracker.snapshot()  # both copies snapshot before the race starts
    copy_b_snapshot = tracker.snapshot()

    # Tick 1: copy A (grid-advantaged) is processed first and writes to the live tracker...
    tracker.update(tick=1, distance_m=3.0)
    # ...then copy B is processed the same tick, behind A in this same race.
    bonus_b = bonus_from_snapshot(
        copy_b_snapshot, previous_tick=0, previous_distance_m=0.0, current_tick=1, current_distance_m=1.0
    )

    # B is judged against the pre-race snapshot (no record yet), not A's just-written 3.0m.
    assert bonus_b == pytest.approx(1.0)
    assert copy_a_snapshot is not copy_b_snapshot  # independent copies, not aliasing the live array
