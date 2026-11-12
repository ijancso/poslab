"""Common algorithm interface and the sensor-data container passed to algorithms.

Algorithms only ever see ``SensorData`` - never the ground truth - so a test cannot
accidentally pass because an algorithm peeked at the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd

ESTIMATE_COLUMNS = ["t", "x", "y", "heading", "v"]


@dataclass
class SensorData:
    """Sensor streams in the local ENU frame. ``imu``/``odom`` may be absent (recordings)."""

    gnss: pd.DataFrame                 # t, x, y, sigma
    imu: pd.DataFrame | None = None    # t, accel_long, accel_lat, yaw_rate
    odom: pd.DataFrame | None = None   # t, speed

    @property
    def t_end(self) -> float:
        """Last timestamp of any available stream."""
        frames = [f for f in (self.gnss, self.imu, self.odom) if f is not None and len(f)]
        return max(float(f["t"].iloc[-1]) for f in frames)


class Algorithm(Protocol):
    """What every registered algorithm provides."""

    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    config: dict[str, Any]

    def run(self, data: SensorData) -> pd.DataFrame:
        """Return an estimate DataFrame with columns ``ESTIMATE_COLUMNS``."""
        ...


def merge_params(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Flat config merge that rejects unknown keys (catches typos in tuning)."""
    unknown = set(overrides or {}) - set(defaults)
    if unknown:
        raise KeyError(f"unknown algorithm parameter(s): {sorted(unknown)}")
    return {**defaults, **(overrides or {})}


def output_times(data: SensorData, rate_hz: float) -> np.ndarray:
    """Regular output grid from the first GNSS fix to the end of the data.

    Estimates are produced on this grid also during GNSS outages, so drift can be measured.
    """
    if data.gnss.empty:
        raise ValueError("no GNSS data: cannot initialise the filter")
    t0 = float(data.gnss["t"].iloc[0])
    n = int(np.floor((data.t_end - t0) * rate_hz + 1e-9)) + 1
    return t0 + np.arange(n) / rate_hz
