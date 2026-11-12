"""Baseline algorithm: GNSS-only 2D constant-velocity Kalman filter.

State  s = [x, y, vx, vy]  (ENU, metres and m/s)
Model  constant velocity, driven by white-noise acceleration with spectral density q:

    F = [[I, dt I], [0, I]]
    Q = q * [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]

Update with GNSS position z = [x, y], H = [I 0], R = sigma^2 I where sigma is the
receiver-reported accuracy (floored at ``sigma_floor``).

Outlier gate: the normalised innovation squared d2 = y^T S^-1 y is chi-square with 2 degrees
of freedom for a consistent filter; fixes with d2 > ``gate`` (13.8 = 99.9 %) are rejected.
After ``max_rejects`` consecutive rejections the filter re-initialises its position at the
next fix (velocity kept, uncertainty inflated), so it cannot lock itself out after a real
jump.

This is the "old" algorithm: it knows nothing about IMU or odometry, so during an outage it
can only extrapolate a straight line at the last velocity.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from poslab.fusion.base import ESTIMATE_COLUMNS, SensorData, merge_params, output_times

DEFAULTS: dict[str, Any] = {
    "q": 0.05,              # [m^2/s^3] white-noise acceleration spectral density
    "sigma_floor": 0.01,    # [m] lower bound for the reported GNSS sigma
    "init_vel_sigma": 3.0,  # [m/s] initial velocity uncertainty (velocity starts at 0)
    "gate": 13.8,           # chi-square(2) threshold, None/0 disables gating
    "max_rejects": 5,
    "rate_hz": 10.0,        # output rate
}


def transition(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State transition F and process noise Q for a time step ``dt``."""
    i2 = np.eye(2)
    f = np.block([[i2, dt * i2], [np.zeros((2, 2)), i2]])
    qm = q * np.block([[dt**3 / 3 * i2, dt**2 / 2 * i2], [dt**2 / 2 * i2, dt * i2]])
    return f, qm


class GnssKF:
    """GNSS-only constant-velocity Kalman filter (see module docstring)."""

    NAME: ClassVar[str] = "gnss_kf"
    VERSION: ClassVar[str] = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = merge_params(DEFAULTS, config)
        self.rejected = 0   # number of gated fixes in the last run (diagnostics)

    def run(self, data: SensorData) -> pd.DataFrame:
        """Filter the GNSS fixes; return estimates on a regular output grid."""
        c = self.config
        gnss = data.gnss.sort_values("t")
        tz = gnss["t"].to_numpy()
        z = gnss[["x", "y"]].to_numpy()
        sig = np.maximum(gnss["sigma"].to_numpy(), c["sigma_floor"])
        t_out = output_times(data, c["rate_hz"])

        # event list: (time, kind, index); kind 0 = measurement before output at equal time
        events = sorted([(t, 0, i) for i, t in enumerate(tz)] +
                        [(t, 1, i) for i, t in enumerate(t_out)])
        s = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
        p = np.diag([sig[0] ** 2, sig[0] ** 2, c["init_vel_sigma"] ** 2, c["init_vel_sigma"] ** 2])
        h = np.hstack([np.eye(2), np.zeros((2, 2))])
        t_now, streak, self.rejected = float(tz[0]), 0, 0
        out = np.zeros((len(t_out), 4))
        for t, kind, i in events:
            if t > t_now:
                f, qm = transition(t - t_now, c["q"])
                s, p = f @ s, f @ p @ f.T + qm
                t_now = t
            if kind == 1:
                out[i] = s
                continue
            if i == 0:
                continue                                  # first fix initialised the state
            innov = z[i] - h @ s
            sm = h @ p @ h.T + sig[i] ** 2 * np.eye(2)
            d2 = float(innov @ np.linalg.solve(sm, innov))
            if c["gate"] and d2 > c["gate"]:
                if streak < c["max_rejects"]:
                    streak += 1
                    self.rejected += 1
                    continue
                # GNSS keeps disagreeing: trust it - re-initialise the position at the fix
                # (a single forced update would barely move a confident filter)
                s[:2] = z[i]
                p[:2, :] = 0.0
                p[:, :2] = 0.0
                p[:2, :2] = sig[i] ** 2 * np.eye(2)
                p[2:, 2:] += c["init_vel_sigma"] ** 2 * np.eye(2)
                streak = 0
                continue
            streak = 0
            k = p @ h.T @ np.linalg.inv(sm)
            s = s + k @ innov
            p = (np.eye(4) - k @ h) @ p
        return pd.DataFrame({
            "t": t_out, "x": out[:, 0], "y": out[:, 1],
            "heading": np.arctan2(out[:, 3], out[:, 2]),
            "v": np.hypot(out[:, 2], out[:, 3]),
        })[ESTIMATE_COLUMNS]
