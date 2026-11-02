import numpy as np
import pytest

from gnsslab.geo import enu_to_geodetic
from gnsslab.io import make_track
from gnsslab.kalman import K68, predict_ahead, run_filter, sigma_from_accuracy
from gnsslab.outage import outage_study, summarize
from gnsslab.static import analyze_static


def static_track(sigma, n=20000, seed=0, acc=None):
    rng = np.random.default_rng(seed)
    e, nn = rng.normal(0, sigma, n), rng.normal(0, sigma, n)
    lat, lon, h = enu_to_geodetic(e, nn, np.zeros(n), 47.5, 19.0, 0.0)
    return make_track(np.arange(n, dtype=float), lat, lon, h,
                      acc=None if acc is None else np.full(n, acc))


def test_static_cep_matches_gaussian_theory():
    r = analyze_static(static_track(2.0), ref=(47.5, 19.0, 0.0))
    assert r.cep50 == pytest.approx(1.1774 * 2.0, rel=0.03)   # Rayleigh median
    assert r.cep95 == pytest.approx(2.4477 * 2.0, rel=0.03)
    assert abs(r.mean_e) < 0.1 and abs(r.mean_n) < 0.1


def test_accuracy_68_percent_calibration():
    r = analyze_static(static_track(2.0, acc=K68 * 2.0))
    assert r.within_acc == pytest.approx(0.68, abs=0.02)


def test_white_noise_has_short_correlation_time():
    r = analyze_static(static_track(2.0, n=3000))
    assert r.corr_time_s <= 1.0


def test_filter_tracks_straight_line_and_rejects_outlier():
    rng = np.random.default_rng(1)
    t = np.arange(200.0)
    e = 10.0 * t + rng.normal(0, 2, 200)
    n = 5.0 * t + rng.normal(0, 2, 200)
    e[100] += 80                                   # multipath jump
    run = run_filter(t, e, n, np.full(200, 2.0), q=0.1)
    assert run.rejected[100]
    assert np.allclose(run.x[-1, 2:], [10, 5], atol=0.5)


def test_prediction_without_noise_is_exact_for_constant_velocity():
    pos, cov, drms = predict_ahead([0, 0, 3, 4], np.eye(4) * 0.01, 0.0, [1.0, 2.0, 10.0], q=1.0)
    assert np.allclose(pos[-1], [30, 40])
    assert np.all(np.diff(drms) > 0)              # uncertainty grows during the outage
    assert cov.shape == (3, 2, 2)


def test_filter_is_consistent_when_model_is_right():
    # constant velocity truth + white noise: the 95 % ellipse should hold ~95 % of drifts
    rng = np.random.default_rng(3)
    t = np.arange(3000.0)
    e, n = 12.0 * t + rng.normal(0, 2, 3000), 5.0 * t + rng.normal(0, 2, 3000)
    run = run_filter(t, e, n, np.full(3000, 2.0), q=0.01)
    row = summarize(outage_study(run, [10.0], gap=0.0))[0]
    assert row["inside_95_ellipse"] > 0.85


def test_outage_study_on_straight_drive_gives_small_drift():
    rng = np.random.default_rng(2)
    t = np.arange(600.0)
    e, n = 15.0 * t + rng.normal(0, 1.5, 600), rng.normal(0, 1.5, 600)
    run = run_filter(t, e, n, sigma_from_accuracy(np.full(600, 1.5 * K68)), q=0.05)
    rows = summarize(outage_study(run, [10.0]))
    assert rows[0]["outages"] > 10
    assert rows[0]["kf_end_median_m"] < 5.0
    assert rows[0]["hold_end_median_m"] > 100.0


def test_sigma_from_accuracy_defaults_and_floor():
    s = sigma_from_accuracy([np.nan, 0.1, K68 * 4])
    assert s.tolist() == pytest.approx([5.0, 1.0, 4.0])
