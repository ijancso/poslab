"""GNSS + odometry + IMU EKF (poslab.fusion.ekf) and the shared gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poslab.fusion.base import ESTIMATE_COLUMNS, Gate, SensorData
from poslab.fusion.ekf import FusionEKF


def straight_drive(duration: float = 60.0, v: float = 2.5, heading: float = 0.3,
                   gap: tuple[float, float] | None = None, gyro_bias: float = 0.0,
                   odom_scale: float = 1.0, vib_amp: float = 0.0) -> SensorData:
    """Noiseless sensors for a constant-speed straight drive from (10, -5)."""
    t_imu = np.arange(0, duration + 1e-9, 0.01)
    imu = pd.DataFrame({"t": t_imu,
                        "accel_long": vib_amp * np.sin(2 * np.pi * 27.0 * t_imu),
                        "accel_lat": 0.0, "yaw_rate": gyro_bias})
    t_g = t_imu[::10]
    gnss = pd.DataFrame({"t": t_g, "x": 10.0 + v * np.cos(heading) * t_g,
                         "y": -5.0 + v * np.sin(heading) * t_g, "sigma": 0.02})
    if gap:
        gnss = gnss[(gnss.t < gap[0]) | (gnss.t >= gap[1])].reset_index(drop=True)
    odom = pd.DataFrame({"t": t_imu[::2], "speed": v * odom_scale})
    return SensorData(gnss=gnss, imu=imu, odom=odom)


def truth_xy(t: np.ndarray, v: float = 2.5, heading: float = 0.3) -> tuple[np.ndarray, ...]:
    return 10.0 + v * np.cos(heading) * t, -5.0 + v * np.sin(heading) * t


def test_converges_on_noiseless_straight_line():
    ekf = FusionEKF()
    est = ekf.run(straight_drive())
    assert list(est.columns) == ESTIMATE_COLUMNS
    assert ekf.init_time is not None and ekf.init_time < 1.0     # 1 m travelled
    late = est[est.t > 5]
    x, y = truth_xy(late.t.to_numpy())
    np.testing.assert_allclose(late.x, x, atol=2e-3)
    np.testing.assert_allclose(late.y, y, atol=2e-3)
    np.testing.assert_allclose(late.heading, 0.3, atol=1e-3)
    np.testing.assert_allclose(late.v, 2.5, atol=1e-3)


def test_before_initialisation_outputs_latest_fix_without_heading():
    est = FusionEKF().run(straight_drive())
    first = est.iloc[0]
    assert np.isnan(first.heading) and first.x == pytest.approx(10.0)


def test_coasts_through_straight_outage():
    est = FusionEKF().run(straight_drive(gap=(20.0, 40.0)))
    inside = est[(est.t >= 20.0) & (est.t < 40.0)]
    x, y = truth_xy(inside.t.to_numpy())
    assert np.hypot(inside.x - x, inside.y - y).max() < 0.01      # 20 s, < 1 cm


def test_gyro_bias_turns_into_cross_track_drift_during_outage():
    """Unestimated gyro bias b: heading error b*tau, lateral drift ~ v b tau^2 / 2."""
    est = FusionEKF().run(straight_drive(gap=(20.0, 40.0), gyro_bias=0.001))
    end = est[(est.t >= 39.85) & (est.t < 40.0)].iloc[-1]
    x, y = truth_xy(np.array([end.t]))
    drift = float(np.hypot(end.x - x[0], end.y - y[0]))
    assert drift == pytest.approx(0.5 * 2.5 * 0.001 * 20.0**2, rel=0.35)


def test_odometry_scale_error_turns_into_along_track_drift_during_outage():
    est = FusionEKF().run(straight_drive(gap=(20.0, 30.0), odom_scale=1.05))
    end = est[est.t < 30.0].iloc[-1]
    x, y = truth_xy(np.array([end.t]))
    along = (end.x - x[0]) * np.cos(0.3) + (end.y - y[0]) * np.sin(0.3)
    assert along == pytest.approx(0.05 * 2.5 * 10.0, rel=0.25)    # ~1.25 m ahead


def test_lowpass_removes_vibration_from_speed_without_odometry():
    data = straight_drive(vib_amp=3.0)
    data.odom = data.odom.iloc[:1]                                 # only the first sample
    filt = FusionEKF().run(data)
    raw = FusionEKF({"accel_lowpass_hz": None}).run(data)
    late = filt.t > 5
    assert np.std(filt.v[late]) < 0.2 * np.std(raw.v[late])


def test_gate_rejects_spike_and_resets_after_persistent_jump():
    data = straight_drive()
    g = data.gnss
    g.loc[(g.t >= 30.0) & (g.t < 30.3), "x"] += 5.0              # 3-epoch spike
    ekf = FusionEKF()
    est = ekf.run(data)
    assert ekf.rejected == 3
    near = est[(est.t >= 30.0) & (est.t < 31.0)]
    x, _ = truth_xy(near.t.to_numpy())
    assert np.abs(near.x - x).max() < 0.05

    data = straight_drive()
    data.gnss.loc[data.gnss.t >= 30.0, "x"] += 5.0                # permanent jump
    ekf = FusionEKF()
    est = ekf.run(data)
    assert ekf.rejected == 5
    x, _ = truth_xy(est.t.to_numpy()[-1:])
    assert abs(est.x.iloc[-1] - (x[0] + 5.0)) < 0.05


def test_requires_imu_and_odometry():
    data = straight_drive()
    with pytest.raises(ValueError, match="IMU and odometry"):
        FusionEKF().run(SensorData(gnss=data.gnss))


def test_gate_logic():
    gate = Gate(10.0, max_rejects=2)
    assert [gate.check(d) for d in (1.0, 50.0, 50.0, 50.0, 1.0)] == \
        ["accept", "reject", "reject", "reset", "accept"]
    assert gate.rejected == 2
    assert Gate(None, 2).check(1e9) == "accept"
