"""Algorithm under test: GNSS + wheel odometry + IMU extended Kalman filter.

State  s = [x, y, heading, v]   (ENU metres, heading from east CCW [rad], speed [m/s])

Predict (unicycle model, at every IMU sample, inputs held constant until the next one):
    x += v cos(h) dt          y += v sin(h) dt
    h += w dt                 w = gyro yaw rate
    v += a dt                 a = longitudinal accel after a CAUSAL low-pass filter

Jacobian of the prediction:
    F = [[1, 0, -v sin(h) dt, cos(h) dt],
         [0, 1,  v cos(h) dt, sin(h) dt],
         [0, 0,  1,           0        ],
         [0, 0,  0,           1        ]]
    P <- F P F^T + diag(q_xy, q_xy, q_heading, q_v) * dt

Updates
    GNSS position  z = [x, y], H = [I 0], R = sigma^2 I (receiver-reported sigma),
                   with the same chi-square gate as gnss_kf (``base.Gate``)
    odometry speed z = v,      H = [0 0 0 1], R = odom_sigma^2

Why the low-pass: engine vibration (27 Hz by default) is large on the accelerometer but
carries no information about vehicle motion. The filter must be causal because the EKF
runs forward in time; ``dsp.lowpass(causal=True)`` over the whole signal gives exactly the
samples a streaming filter would produce, so pre-filtering the array is equivalent to
filtering inside the loop. The price is group delay (~83 ms at 5 Hz, 4th order): the speed
prediction reacts that much later to real acceleration changes, which odometry corrects.

Initialisation: the heading is not observable from one fix, so the filter waits until the
GNSS position has moved ``max(init_dist, 5 sigma)`` from the first fix and takes the heading
of that displacement (assumes the vehicle starts on a roughly straight path). Until then it
outputs the latest GNSS fix with heading NaN.

During GNSS outages the filter coasts on gyro + odometry (+ accel): heading error grows with
the unestimated gyro bias, along-track error with the odometry scale error.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from poslab.dsp import lowpass
from poslab.fusion.base import ESTIMATE_COLUMNS, Gate, SensorData, merge_params, output_times

DEFAULTS: dict[str, Any] = {
    "accel_lowpass_hz": 5.0,    # None/0 = no filtering (to demonstrate the effect)
    "accel_lowpass_order": 4,
    "q_xy": 1e-4,               # [m^2/s] position random walk (slip, model error)
    "q_heading": 1e-5,          # [rad^2/s] heading random walk (gyro noise + bias)
    "q_v": 0.01,                # [m^2/s^3] speed random walk (accel noise + bias)
    "odom_sigma": 0.05,         # [m/s] odometry speed noise
    "sigma_floor": 0.01,        # [m] lower bound for the reported GNSS sigma
    "gate": 13.8,               # chi-square(2) 99.9 %, None/0 disables
    "max_rejects": 5,
    "init_dist": 1.0,           # [m] minimum travel before the heading is initialised
    "rate_hz": 10.0,            # output rate
}

# event kinds, processed in this order at equal timestamps
_IMU, _GNSS, _ODOM, _OUT = 0, 1, 2, 3


class FusionEKF:
    """GNSS + odometry + IMU EKF (see module docstring)."""

    NAME: ClassVar[str] = "ekf"
    VERSION: ClassVar[str] = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = merge_params(DEFAULTS, config)
        self.rejected = 0
        self.init_time: float | None = None

    # -------------------------------------------------------------- filter steps

    def _predict(self, s: np.ndarray, p: np.ndarray, w: float, a: float, dt: float
                 ) -> tuple[np.ndarray, np.ndarray]:
        x, y, h, v = s
        c, sn = np.cos(h), np.sin(h)
        s_new = np.array([x + v * c * dt, y + v * sn * dt, h + w * dt, v + a * dt])
        f = np.eye(4)
        f[0, 2], f[0, 3] = -v * sn * dt, c * dt
        f[1, 2], f[1, 3] = v * c * dt, sn * dt
        q = self._q * dt
        return s_new, f @ p @ f.T + q

    @staticmethod
    def _update_position(s: np.ndarray, p: np.ndarray, z: np.ndarray, r: float
                         ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (state, covariance, d2) after a position update; d2 = NIS before it."""
        innov = z - s[:2]
        sm = p[:2, :2] + r * np.eye(2)
        sm_inv = np.linalg.inv(sm)
        d2 = float(innov @ sm_inv @ innov)
        k = p[:, :2] @ sm_inv
        return s + k @ innov, p - k @ p[:2, :], d2

    @staticmethod
    def _update_speed(s: np.ndarray, p: np.ndarray, z: float, r: float
                      ) -> tuple[np.ndarray, np.ndarray]:
        sm = p[3, 3] + r
        k = p[:, 3] / sm
        return s + k * (z - s[3]), p - np.outer(k, p[3, :])

    # -------------------------------------------------------------- main loop

    def run(self, data: SensorData) -> pd.DataFrame:
        """Fuse GNSS, odometry and IMU; return estimates on a regular output grid."""
        if data.imu is None or data.odom is None:
            raise ValueError("ekf needs IMU and odometry data")
        c = self.config
        self._q = np.diag([c["q_xy"], c["q_xy"], c["q_heading"], c["q_v"]])

        imu_t = data.imu["t"].to_numpy()
        gyro = data.imu["yaw_rate"].to_numpy()
        acc = data.imu["accel_long"].to_numpy()
        if c["accel_lowpass_hz"]:
            fs = 1.0 / float(np.median(np.diff(imu_t)))
            acc = lowpass(acc, fs, c["accel_lowpass_hz"], c["accel_lowpass_order"], causal=True)
        g_t = data.gnss["t"].to_numpy()
        g_z = data.gnss[["x", "y"]].to_numpy()
        g_r = np.maximum(data.gnss["sigma"].to_numpy(), c["sigma_floor"]) ** 2
        o_t, o_v = data.odom["t"].to_numpy(), data.odom["speed"].to_numpy()
        t_out = output_times(data, c["rate_hz"])

        events = sorted([(t, _IMU, i) for i, t in enumerate(imu_t)]
                        + [(t, _GNSS, i) for i, t in enumerate(g_t)]
                        + [(t, _ODOM, i) for i, t in enumerate(o_t)]
                        + [(t, _OUT, i) for i, t in enumerate(t_out)])

        gate = Gate(c["gate"], c["max_rejects"])
        out = np.full((len(t_out), 4), np.nan)
        s, p = np.zeros(4), np.eye(4)
        ready = False
        t_now, w, a = None, 0.0, 0.0              # held IMU inputs
        last_fix, last_speed = None, np.nan
        first_fix: tuple[np.ndarray, float] | None = None
        self.init_time = None

        for t, kind, i in events:
            if ready and t > t_now:
                s, p = self._predict(s, p, w, a, t - t_now)
                t_now = t
            if kind == _IMU:
                w, a = gyro[i], acc[i]
            elif kind == _ODOM:
                last_speed = o_v[i]
                if ready:
                    s, p = self._update_speed(s, p, o_v[i], c["odom_sigma"] ** 2)
            elif kind == _GNSS:
                last_fix = g_z[i]
                if not ready:
                    if first_fix is None:
                        first_fix = (g_z[i], g_r[i])
                        continue
                    d = g_z[i] - first_fix[0]
                    sig = np.sqrt(max(g_r[i], first_fix[1]))
                    if np.hypot(*d) < max(c["init_dist"], 5.0 * sig) or np.isnan(last_speed):
                        continue
                    s = np.array([g_z[i, 0], g_z[i, 1], np.arctan2(d[1], d[0]), last_speed])
                    h_sig = 2.0 * sig / np.hypot(*d) + 0.02
                    p = np.diag([g_r[i], g_r[i], h_sig**2, c["odom_sigma"] ** 2])
                    ready, t_now, self.init_time = True, t, t
                    continue
                s_upd, p_upd, d2 = self._update_position(s, p, g_z[i], g_r[i])
                verdict = gate.check(d2)
                if verdict == "accept":
                    s, p = s_upd, p_upd
                elif verdict == "reset":                    # re-initialise position at the fix
                    s[:2] = g_z[i]
                    p[:2, :] = 0.0
                    p[:, :2] = 0.0
                    p[:2, :2] = g_r[i] * np.eye(2)
            else:  # _OUT
                if ready:
                    out[i] = s
                elif last_fix is not None:
                    out[i] = [last_fix[0], last_fix[1], np.nan, last_speed]

        self.rejected = gate.rejected
        heading = np.pi - np.mod(np.pi - out[:, 2], 2 * np.pi)   # wrap to (-pi, pi]
        return pd.DataFrame({"t": t_out, "x": out[:, 0], "y": out[:, 1],
                             "heading": heading, "v": out[:, 3]})[ESTIMATE_COLUMNS]
