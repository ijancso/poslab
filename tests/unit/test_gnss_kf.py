"""GNSS-only constant-velocity Kalman filter and the algorithm registry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poslab.fusion.base import ESTIMATE_COLUMNS, SensorData, output_times
from poslab.fusion.gnss_kf import GnssKF, transition
from poslab.fusion.registry import ALGORITHMS, get_algorithm


def straight_line(duration: float = 60.0, v: tuple[float, float] = (0.0, 2.5),
                  sigma: float = 0.0, seed: int = 0, gap: tuple[float, float] | None = None
                  ) -> pd.DataFrame:
    """10 Hz GNSS fixes of a constant-velocity drive, optional noise and outage."""
    t = np.arange(0, duration + 1e-9, 0.1)
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"t": t, "x": 5.0 + v[0] * t + rng.normal(0, sigma, t.size),
                      "y": -3.0 + v[1] * t + rng.normal(0, sigma, t.size),
                      "sigma": max(sigma, 0.01)})
    if gap:
        g = g[(g.t < gap[0]) | (g.t >= gap[1])].reset_index(drop=True)
    return g


def test_transition_matrices():
    f, q = transition(0.5, 2.0)
    s = f @ np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(s, [2.5, 4.0, 3.0, 4.0])
    np.testing.assert_allclose(q[0, 0], 2.0 * 0.5**3 / 3)
    np.testing.assert_allclose(q, q.T)


def test_converges_on_noiseless_straight_line():
    est = GnssKF().run(SensorData(gnss=straight_line()))
    assert list(est.columns) == ESTIMATE_COLUMNS
    late = est[est.t > 10]
    np.testing.assert_allclose(late.x, 5.0, atol=1e-3)
    np.testing.assert_allclose(late.y, -3.0 + 2.5 * late.t, atol=1e-3)
    np.testing.assert_allclose(late.v, 2.5, atol=1e-3)
    np.testing.assert_allclose(late.heading, np.pi / 2, atol=1e-3)


def test_reduces_noise_on_straight_line():
    g = straight_line(sigma=0.5, seed=1)
    est = GnssKF().run(SensorData(gnss=g))
    late = est.t > 10
    err_kf = np.hypot(est.x[late] - 5.0, est.y[late] - (-3.0 + 2.5 * est.t[late]))
    err_raw = np.hypot(g.x[g.t > 10] - 5.0, g.y[g.t > 10] - (-3.0 + 2.5 * g.t[g.t > 10]))
    assert np.sqrt(np.mean(err_kf**2)) < 0.5 * np.sqrt(np.mean(err_raw**2))


def test_outputs_through_outage_and_extrapolates_straight_line():
    g = straight_line(v=(1.0, 2.0), gap=(30.0, 40.0))
    est = GnssKF().run(SensorData(gnss=g))
    np.testing.assert_allclose(np.diff(est.t), 0.1, atol=1e-9)
    inside = est[(est.t >= 30.0) & (est.t < 40.0)]
    assert len(inside) == 100
    np.testing.assert_allclose(inside.x, 5.0 + inside.t, atol=1e-3)
    np.testing.assert_allclose(inside.y, -3.0 + 2.0 * inside.t, atol=1e-3)


def test_gate_rejects_multipath_spike_but_not_forever():
    g = straight_line(sigma=0.05, seed=2)
    spike = (g.t >= 20.0) & (g.t < 20.3)                     # 3 epochs, +10 m east
    g.loc[spike, "x"] += 10.0
    kf = GnssKF()
    est = kf.run(SensorData(gnss=g))
    assert kf.rejected == 3
    near = est[(est.t >= 20.0) & (est.t < 21.0)]
    assert np.abs(near.x - 5.0).max() < 0.2

    jump = straight_line(sigma=0.05, seed=2)
    jump.loc[jump.t >= 20.0, "x"] += 10.0                    # permanent jump
    kf = GnssKF({"max_rejects": 5})
    est = kf.run(SensorData(gnss=jump))
    assert kf.rejected == 5
    assert abs(est.x.iloc[-1] - 15.0) < 0.2                  # filter followed the jump


def test_gate_can_be_disabled():
    g = straight_line(sigma=0.05, seed=2)
    g.loc[(g.t >= 20.0) & (g.t < 20.3), "x"] += 10.0
    kf = GnssKF({"gate": None})
    kf.run(SensorData(gnss=g))
    assert kf.rejected == 0


def test_output_grid_extends_to_end_of_other_sensors():
    g = straight_line(duration=10.0)
    odom = pd.DataFrame({"t": [0.0, 12.0], "speed": [2.5, 2.5]})
    t = output_times(SensorData(gnss=g, odom=odom), 10.0)
    assert t[0] == 0.0 and t[-1] == pytest.approx(12.0)
    with pytest.raises(ValueError, match="no GNSS"):
        output_times(SensorData(gnss=g.iloc[:0]), 10.0)


def test_registry():
    assert "gnss_kf" in ALGORITHMS
    alg = get_algorithm("gnss_kf", {"q": 1.0})
    assert alg.VERSION and alg.config["q"] == 1.0
    with pytest.raises(KeyError, match="unknown algorithm parameter"):
        get_algorithm("gnss_kf", {"qq": 1.0})
    with pytest.raises(KeyError, match="available"):
        get_algorithm("nope")
