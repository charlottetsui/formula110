"""Frozen behavioral clone. NumPy-only inference; no training or expert imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from controllers.clone_features import FEATURE_VERSION, OBSERVATION_DIM, ObservationHistory, release_brake
from racing import RobotCommand, RobotSensors

RACING_NAME = "Expert Clone"
RACING_COLOR = "#845EF7"
DEFAULT_MODEL_PATH = Path(__file__).with_name("imitation_policy.npz")


class Controller:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        with np.load(self.model_path, allow_pickle=False) as artifact:
            if int(artifact["feature_version"].item()) != FEATURE_VERSION:
                raise ValueError("Clone observation version mismatch")
            self.layers: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = [
                (artifact[f"weight_{i}"].copy(), artifact[f"bias_{i}"].copy()) for i in range(3)
            ]
        size = OBSERVATION_DIM
        for weight, bias in self.layers:
            if weight.ndim != 2 or weight.shape[1] != size or bias.shape != (weight.shape[0],):
                raise ValueError("Invalid clone network dimensions")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("Nonfinite clone weights")
            size = weight.shape[0]
        if size != 2:
            raise ValueError("Clone must output throttle and steer")
        self.history = ObservationHistory()
        self.previous = RobotCommand()

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        value = observation
        for weight, bias in self.layers[:-1]:
            value = np.maximum(weight @ value + bias, 0.0)
        weight, bias = self.layers[-1]
        return np.tanh(weight @ value + bias)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        if sensors.tick == 0:
            self.previous = RobotCommand()
        action = self.predict(self.history.append(sensors, self.previous))
        requested = RobotCommand(throttle=float(action[0]), steer=float(action[1]))
        self.previous = release_brake(requested, self.previous)
        return self.previous

    def copy_for_car(self) -> Controller:
        return Controller(self.model_path)


def create_controller() -> Controller:
    return Controller()
