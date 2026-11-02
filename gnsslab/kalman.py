"""A 2-D constant-velocity Kalman filter on GNSS positions (local ENU metres).

State x = [e, n, ve, vn]. The motion model assumes constant velocity plus
random (white-noise) acceleration; GNSS gives position measurements only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Android reports accuracy as the radius of 68 % confidence. For a circular 2-D
# Gaussian with per-axis sigma s, P(r < k*s) = 1 - exp(-k^2/2) = 0.68  ->  k = 1.51
K68 = float(np.sqrt(-2.0 * np.log(1.0 - 0.68)))

CHI2_2DOF_999 = 13.82   # 99.9 % gate for a 2-D innovation


def sigma_from_accuracy(acc, default=5.0, floor=1.0) -> np.ndarray:
    """Per-axis measurement sigma [m] from Android's reported accuracy."""
    s = np.asarray(acc, float) / K68
    s = np.where(np.isfinite(s), s, default)
    return np.maximum(s, floor)


def transition(dt: float, q: float):
    """State transition F and process noise Q for one time step.

    q is the spectral density of the white-noise acceleration [m^2/s^3];
    larger q = the filter trusts the constant-velocity model less.
    """
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    a, b, c = dt ** 3 / 3, dt ** 2 / 2, dt
    Q = q * np.array([[a, 0, b, 0],
                      [0, a, 0, b],
                      [b, 0, c, 0],
                      [0, b, 0, c]])
    return F, Q


@dataclass
class FilterRun:
    t: np.ndarray         # epoch times [s]
    x: np.ndarray         # (N, 4) posterior state at each epoch
    P: np.ndarray         # (N, 4, 4) posterior covariance
    updated: np.ndarray   # (N,) True where a GNSS fix was used
    rejected: np.ndarray  # (N,) True where a fix failed the innovation gate
    q: float


def run_filter(t, e, n, sigma, q=3.0, gate=CHI2_2DOF_999, use=None, max_rejects=3,
               init_vel_sigma=10.0) -> FilterRun:
    """Filter a whole track. use[k] == False withholds fix k (simulated outage).

    Fixes whose innovation fails the chi-square gate are skipped as outliers
    (multipath jumps). After max_rejects consecutive rejections the next fix is
    accepted anyway, so the filter cannot lock itself out after a real jump.
    """
    t = np.asarray(t, float)
    N = len(t)
    use = np.ones(N, bool) if use is None else np.asarray(use, bool)
    if not use[0]:
        raise ValueError("the first fix must be usable to initialise the filter")

    X = np.zeros((N, 4))
    P = np.zeros((N, 4, 4))
    updated = np.zeros(N, bool)
    rejected = np.zeros(N, bool)
    I4 = np.eye(4)

    x = np.array([e[0], n[0], 0.0, 0.0])
    p = np.diag([sigma[0] ** 2, sigma[0] ** 2, init_vel_sigma ** 2, init_vel_sigma ** 2])
    updated[0] = True
    rejects = 0

    for k in range(N):
        if k > 0:
            F, Q = transition(t[k] - t[k - 1], q)
            x = F @ x                       # predict
            p = F @ p @ F.T + Q
            if use[k]:
                R = np.eye(2) * sigma[k] ** 2
                y = np.array([e[k], n[k]]) - x[:2]       # innovation (H selects position)
                S = p[:2, :2] + R
                d2 = float(y @ np.linalg.solve(S, y))     # squared Mahalanobis distance
                if gate is not None and d2 > gate and rejects < max_rejects:
                    rejected[k] = True
                    rejects += 1
                else:
                    K = p[:, :2] @ np.linalg.inv(S)       # Kalman gain
                    x = x + K @ y                          # update
                    IKH = I4.copy()
                    IKH[:, :2] -= K
                    p = IKH @ p @ IKH.T + K @ R @ K.T      # Joseph form: stays symmetric/PSD
                    updated[k] = True
                    rejects = 0
        X[k], P[k] = x, p
    return FilterRun(t, X, P, updated, rejected, q)


def predict_ahead(x0, P0, t0, ts, q):
    """Propagate a state with no measurements (GNSS outage).

    Returns predicted positions (M, 2), their 2x2 covariances (M, 2, 2) and the
    horizontal drms = sqrt(var_e + var_n) at times ts.
    """
    x, p, tp = np.array(x0, float), np.array(P0, float), float(t0)
    pos = np.zeros((len(ts), 2))
    cov = np.zeros((len(ts), 2, 2))
    for i, tk in enumerate(ts):
        F, Q = transition(tk - tp, q)
        x = F @ x
        p = F @ p @ F.T + Q
        tp = tk
        pos[i] = x[:2]
        cov[i] = p[:2, :2]
    drms = np.sqrt(cov[:, 0, 0] + cov[:, 1, 1])
    return pos, cov, drms
