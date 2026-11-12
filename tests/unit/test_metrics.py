"""Metrics on hand-built error cases with known answers (poslab.metrics)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poslab.metrics import compute_metrics, error_components, outage_drift_curve, static_metrics


def north_truth(duration: float = 20.0, v: float = 2.0) -> pd.DataFrame:
    """Truth: driving north along x = 0 at constant speed, 100 Hz."""
    t = np.arange(0, duration + 1e-9, 0.01)
    return pd.DataFrame({"t": t, "x": 0.0, "y": v * t, "heading": np.pi / 2,
                         "speed": v, "yaw_rate": 0.0, "accel_long": 0.0})


def estimate(truth: pd.DataFrame, dx: np.ndarray | float, dy: np.ndarray | float,
             rate: float = 10.0) -> pd.DataFrame:
    """Estimate at ``rate`` Hz = truth + (dx, dy) offsets."""
    t = np.arange(0, truth.t.iloc[-1] + 1e-9, 1 / rate)
    return pd.DataFrame({"t": t, "x": np.interp(t, truth.t, truth.x) + dx,
                         "y": np.interp(t, truth.t, truth.y) + dy,
                         "heading": np.pi / 2, "v": 2.0})


def test_constant_offset_east_is_pure_cross_track():
    """Driving north, 0.3 m east of the path = 0.3 m to the RIGHT -> cross-track -0.3."""
    tr = north_truth()
    err = error_components(estimate(tr, 0.3, 0.0), tr)
    np.testing.assert_allclose(err.horizontal, 0.3)
    np.testing.assert_allclose(err.cross, -0.3, atol=1e-12)
    np.testing.assert_allclose(err.along, 0.0, atol=1e-12)
    m = compute_metrics(estimate(tr, 0.3, 0.0), tr, warmup_s=0.0)
    for key in ("h_rms_m", "h_cep50_m", "h_cep95_m", "h_max_m", "xt_rms_m", "xt_max_m"):
        assert m[key] == pytest.approx(0.3)
    assert m["at_rms_m"] == pytest.approx(0.0, abs=1e-12)
    assert m["n_samples"] == 201


def test_speed_rms_ignores_nan_before_initialisation():
    tr = north_truth()
    est = estimate(tr, 0.0, 0.0)
    est["v"] = 2.1                                              # truth speed is 2.0
    est.loc[est.t < 6.0, "v"] = np.nan
    m = compute_metrics(est, tr, warmup_s=0.0)
    assert m["v_rms_mps"] == pytest.approx(0.1)
    assert "v_rms_mps" not in compute_metrics(est.drop(columns="v"), tr, warmup_s=0.0)


def test_offset_behind_is_pure_along_track():
    tr = north_truth()
    m = compute_metrics(estimate(tr, 0.0, -1.0), tr, warmup_s=0.0)
    assert m["at_rms_m"] == pytest.approx(1.0) and m["xt_rms_m"] == pytest.approx(0.0, abs=1e-12)


def test_cross_track_follows_heading_in_a_turn():
    """Truth heading east: an offset to the north is to the LEFT -> cross-track +0.5."""
    t = np.arange(0, 10.01, 0.01)
    tr = pd.DataFrame({"t": t, "x": 2.0 * t, "y": 0.0, "heading": 0.0})
    est = pd.DataFrame({"t": t[::10], "x": 2.0 * t[::10], "y": 0.5})
    np.testing.assert_allclose(error_components(est, tr).cross, 0.5)


def test_rms_cep_and_max_on_known_error_set():
    """Errors 0, 0.1, ..., 2.0 m (21 values, uniform over time)."""
    tr = north_truth(duration=2.0)
    est = estimate(tr, np.arange(21) * 0.1, 0.0)
    m = compute_metrics(est, tr, warmup_s=0.0)
    e = np.arange(21) * 0.1
    assert m["h_rms_m"] == pytest.approx(np.sqrt(np.mean(e**2)))
    assert m["h_cep50_m"] == pytest.approx(1.0)
    assert m["h_cep95_m"] == pytest.approx(1.9)
    assert m["h_max_m"] == pytest.approx(2.0)


def test_warmup_is_excluded():
    tr = north_truth()
    dx = np.where(np.arange(0, 20.01, 0.1) < 5.0, 10.0, 0.1)    # large error while converging
    m = compute_metrics(estimate(tr, dx, 0.0), tr, warmup_s=5.0)
    assert m["h_max_m"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="warm-up"):
        compute_metrics(estimate(tr, 0.0, 0.0), tr, warmup_s=100.0)


def test_outage_drift_end_error_and_curve():
    """Error grows linearly 0.2 m/s during the outage [8, 13) s, 0 elsewhere."""
    tr = north_truth()
    t = np.arange(0, 20.01, 0.1)
    drift = np.where((t >= 8.0) & (t < 13.0), 0.2 * (t - 8.0), 0.0)
    est = estimate(tr, drift, 0.0)
    m = compute_metrics(est, tr, outages=[[8.0, 13.0], [15.0, 16.0]], warmup_s=0.0)
    assert m["outage1_end_err_m"] == pytest.approx(0.2 * 4.9)    # last estimate at 12.9 s
    assert m["outage2_end_err_m"] == pytest.approx(0.0)
    assert m["outage_end_err_max_m"] == pytest.approx(0.98)
    curve = outage_drift_curve(error_components(est, tr), [[8.0, 13.0]])
    assert len(curve) == 50
    np.testing.assert_allclose(curve.error, 0.2 * curve.tau, atol=1e-9)
    assert outage_drift_curve(error_components(est, tr), []).empty


def test_outage_without_estimates_and_out_of_range_estimates_raise():
    tr = north_truth()
    with pytest.raises(ValueError, match="no estimates inside outage"):
        compute_metrics(estimate(tr, 0.0, 0.0), tr, outages=[[30.0, 31.0]], warmup_s=0.0)
    est = estimate(tr, 0.0, 0.0)
    est.loc[len(est) - 1, "t"] = 25.0
    with pytest.raises(ValueError, match="beyond the truth"):
        compute_metrics(est, tr)


def test_static_metrics_known_points():
    # 4 points on a 1 m square around (10, 20): every distance to the mean is sqrt(0.5)
    x = np.array([9.5, 10.5, 10.5, 9.5])
    y = np.array([19.5, 19.5, 20.5, 20.5])
    m = static_metrics(x, y)
    assert m["cep50_m"] == pytest.approx(np.sqrt(0.5))
    assert m["cep95_m"] == pytest.approx(np.sqrt(0.5))
    assert m["std_east_m"] == pytest.approx(np.std(x, ddof=1))
    assert m["n_samples"] == 4
    with pytest.raises(ValueError):
        static_metrics(np.array([1.0]), np.array([1.0]))


def test_static_metrics_gaussian_theory():
    """2D Gaussian with per-axis sigma s: CEP50 = 1.177 s, CEP95 = 2.448 s."""
    rng = np.random.default_rng(0)
    m = static_metrics(rng.normal(0, 2.0, 50000), rng.normal(0, 2.0, 50000))
    assert m["cep50_m"] == pytest.approx(1.177 * 2.0, rel=0.02)
    assert m["cep95_m"] == pytest.approx(2.448 * 2.0, rel=0.02)
