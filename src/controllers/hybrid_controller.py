"""SAC drives; Lucy's rule-based expert takes the wheel during hazards.

The literal combined-approach controller: composes this track's SAC policy
(`controllers.race_faster`) with the separate imitation-learning track's
hand-written expert (`controllers.leaderboard_expert`, developed
independently on `lucy-il`) rather than blending their training or weights.
Every tick, both sub-controllers see the same sensors and produce a
command; this module picks whichever one drives based on a simple,
public-sensor hazard check -- SAC's command the rest of the time, the
expert's during a wall- or competitor-proximity hazard.

Why a hard switch at inference time, not a training change: two other
combined-approach mechanisms were tried and are documented in
docs/rl_design.md section 6 --

- Training SAC against the expert as a self-play opponent (causal test 35
  and its fine-tuning follow-up) changed the training *distribution*
  itself and caused severe instability at every dose and resume strategy
  tried.
- A reward bonus for matching the expert's action during hazard states
  (causal test 36) stayed inside ordinary self-play and produced a real,
  if modest and only-once-tested, improvement -- but it's a training-time
  nudge baked into the weights, invisible in the final controller.

This module is a third, structurally different approach: no retraining,
no risk of destabilizing either sub-controller's own tuning, and the
combination is literally visible in the code -- one racing.RobotController
built from both tracks' finished, independently-validated work.

Hazard thresholds mirror `training.reward`'s `WALL_WARNING_DISTANCE_M`/
`WALL_WARNING_BEAM_ANGLES_DEGREES`/`ROBOT_WARNING_DISTANCE_M`/
`ROBOT_WARNING_ANGLE_DEGREES` (duplicated, not imported, since this
package has no dependency outside `src/controllers/` -- see
`race_faster.py`'s docstring for why).
"""

from __future__ import annotations

import math

from controllers import leaderboard_expert, race_faster
from racing import RobotCommand, RobotSensors

RACING_NAME = "Hybrid (SAC + Expert Shield)"
RACING_COLOR = "#2ECC71"

_WALL_WARNING_DISTANCE_M = 6.0
_WALL_WARNING_BEAM_ANGLES_DEGREES: tuple[float, ...] = (-20.0, 0.0, 20.0)
_ROBOT_WARNING_DISTANCE_M = 8.0
_ROBOT_WARNING_ANGLE_DEGREES = 45.0


def _is_hazard(sensors: RobotSensors) -> bool:
    """Return whether a wall or competitor is close enough to hand off to the expert."""
    wall_hazard = any(
        _is_close(sensors.wall_lidar.distance_at_angle_degrees(angle), _WALL_WARNING_DISTANCE_M)
        for angle in _WALL_WARNING_BEAM_ANGLES_DEGREES
    )
    if wall_hazard:
        return True
    ahead_competitors = (
        competitor
        for competitor in sensors.camera.competitors
        if abs(competitor.angle_degrees) <= _ROBOT_WARNING_ANGLE_DEGREES
    )
    return any(_is_close(competitor.distance_m, _ROBOT_WARNING_DISTANCE_M) for competitor in ahead_competitors)


def _is_close(distance_m: float, warning_distance_m: float) -> bool:
    return math.isfinite(distance_m) and distance_m < warning_distance_m


class Controller:
    """Runs both sub-controllers every tick; returns the expert's command during a hazard.

    Both are called on every tick regardless of which command is used --
    `leaderboard_expert.Controller` is stateful (a stuck-recovery timer, the
    previous tick's steer/throttle for its own brake-then-accelerate logic),
    so calling it only once a hazard starts would leave it acting on stale
    state exactly when it's needed most.
    """

    def __init__(self) -> None:
        self._sac = race_faster.create_controller()
        self._expert = leaderboard_expert.create_controller()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        sac_command = self._sac(sensors)
        expert_command = self._expert(sensors)
        return expert_command if _is_hazard(sensors) else sac_command

    def copy_for_car(self) -> Controller:
        # A fresh instance per car/race -- `leaderboard_expert.Controller`'s recovery
        # timer and previous-steer state must not be shared across cars sharing a race.
        return Controller()


def create_controller() -> Controller:
    return Controller()
