"""Unit tests for the synthetic scenario generator (poslab.sim.scenario)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from poslab.sim.scenario import load_config, merge_config, simulate

# a small field keeps the tests fast: 3 rows x 40 m
SMALL: dict[str, Any] = {"seed": 7, "field": {"rows": 3, "row_length": 40.0}}


def small(**overrides: Any) -> dict[str, Any]:
    """SMALL config with extra overrides merged on top."""
    return merge_config(overrides, merge_config(SMALL))


@pytest.fixture(scope="module")
def scn():
    return simulate(small())


def test_merge_rejects_unknown_keys_and_bad_types():
    with pytest.raises(KeyError, match="gnss.sigmaa"):
        merge_config({"gnss": {"sigmaa": 1.0}})
    with pytest.raises(TypeError):
        merge_config({"gnss": 1.0})


def test_load_config_uses_file_stem_as_name(tmp_path):
    f = tmp_path / "my_case.yaml"
    f.write_text("seed: 3\ngnss:\n  sigma: 1.5\n")
    cfg = load_config(f)
    assert cfg["name"] == "my_case" and cfg["seed"] == 3 and cfg["gnss"]["sigma"] == 1.5
    assert cfg["imu"]["vib_hz"] == 27.0          # default kept


def test_same_seed_same_data_different_seed_different_noise():
    a, b, c = simulate(small()), simulate(small()), simulate(small(seed=8))
    assert a.gnss.equals(b.gnss) and a.imu.equals(b.imu) and a.odom.equals(b.odom)
    assert a.truth.equals(c.truth)               # truth does not depend on the seed
    assert not np.allclose(a.gnss.x, c.gnss.x)


def test_sensor_streams_are_independent():
    """Changing odometry noise must not change the GNSS noise realisation."""
    a, b = simulate(small()), simulate(small(odom={"noise": 0.5}))
    assert a.gnss.equals(b.gnss)


def test_rates_and_columns(scn):
    assert list(scn.truth.columns) == ["t", "x", "y", "heading", "speed", "yaw_rate", "accel_long"]
    assert np.allclose(np.diff(scn.truth.t), 0.01)
    assert np.allclose(np.diff(scn.gnss.t), 0.1)
    assert np.allclose(np.diff(scn.imu.t), 0.01)
    assert np.allclose(np.diff(scn.odom.t), 0.02)


def test_field_geometry(scn):
    tr = scn.truth
    # starts at the origin heading north, ends after 2 turns at x = 2 * spacing, heading north
    assert tr.x.iloc[0] == pytest.approx(0.0) and tr.y.iloc[0] == pytest.approx(0.0)
    assert tr.heading.iloc[0] == pytest.approx(np.pi / 2)
    assert tr.x.iloc[-1] == pytest.approx(12.0, abs=0.03)
    assert tr.y.iloc[-1] == pytest.approx(40.0, abs=0.03)
    # headland turns are semicircles of radius 3 m beyond the row ends
    assert tr.y.max() == pytest.approx(43.0, abs=0.01)
    assert tr.y.min() == pytest.approx(-3.0, abs=0.01)
    kinds = [iv["kind"] for iv in scn.meta["intervals"]]
    assert kinds == ["row", "turn", "row", "turn", "row"]


def test_wider_spacing_adds_straight_headland_segment():
    s = simulate(small(field={"row_spacing": 10.0}))
    assert s.truth.x.iloc[-1] == pytest.approx(20.0, abs=0.03)
    assert len(s.meta["intervals"]) == 5          # straight part merged into the turn


def test_speed_profile_limits_and_turn_speed(scn):
    tr, m = scn.truth, scn.meta
    assert tr.speed.max() == pytest.approx(2.5) and tr.speed.min() == pytest.approx(1.2)
    assert np.abs(tr.accel_long).max() <= 0.3 + 1e-3
    turn = m["intervals"][1]
    inside = (tr.t > turn["t0"] + 0.05) & (tr.t < turn["t1"] - 0.05)
    assert np.allclose(tr.speed[inside], 1.2)
    # turn radius r = v / |yaw_rate|
    assert np.allclose(tr.speed[inside] / np.abs(tr.yaw_rate[inside]), 3.0)


def test_truth_is_kinematically_consistent(scn):
    """Integrating speed/heading reproduces x/y; accel_long is the derivative of speed."""
    tr, dt = scn.truth, 0.01
    vx, vy = tr.speed * np.cos(tr.heading), tr.speed * np.sin(tr.heading)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:].to_numpy() + vx[:-1].to_numpy()) * dt)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:].to_numpy() + vy[:-1].to_numpy()) * dt)])
    assert np.max(np.hypot(x - tr.x, y - tr.y)) < 0.02
    dv = np.gradient(tr.speed.to_numpy(), dt)
    assert np.median(np.abs(dv - tr.accel_long)) < 1e-3
    assert tr.heading.between(-np.pi, np.pi).all()


def test_gnss_noise_level_and_outages():
    s = simulate(small(gnss={"sigma": 0.5, "outages": [[10.0, 15.0], [30.0, 31.0]]}))
    t = s.gnss.t.to_numpy()
    assert not ((t >= 10.0) & (t < 15.0)).any() and not ((t >= 30.0) & (t < 31.0)).any()
    assert ((t >= 15.0) & (t < 15.05)).any()
    ex = s.gnss.x - np.interp(t, s.truth.t, s.truth.x)
    assert ex.mean() == pytest.approx(0.0, abs=0.05)
    assert ex.std() == pytest.approx(0.5, rel=0.1)


def test_gnss_bias_walk_and_multipath_spikes():
    base = simulate(small(gnss={"sigma": 0.0}))
    walk = simulate(small(gnss={"sigma": 0.0, "bias_rw": 0.1}))
    err = np.hypot(walk.gnss.x - base.gnss.x, walk.gnss.y - base.gnss.y)
    assert err.iloc[-1] > 0.05                    # bias has wandered away from zero
    mp = simulate(small(gnss={"sigma": 0.0, "multipath": {"prob": 0.02, "amp": 5.0, "epochs": 3}}))
    jump = np.hypot(mp.gnss.x - base.gnss.x, mp.gnss.y - base.gnss.y)
    assert (jump > 4.99).sum() >= 3 and jump.max() <= 5.0 + 1e-6


def test_imu_vibration_bias_and_lateral_accel():
    s = simulate(small(imu={"accel_noise": 0.0, "gyro_noise": 0.0, "vib_amp": 1.0}))
    tr, imu = s.truth, s.imu
    resid = imu.accel_long - tr.accel_long - 0.02
    assert resid.abs().max() == pytest.approx(1.0, abs=0.01)   # pure 1.0 m/s^2 sinusoid
    spec = np.abs(np.fft.rfft(resid.to_numpy()))
    f = np.fft.rfftfreq(len(resid), 0.01)
    assert f[np.argmax(spec)] == pytest.approx(27.0, abs=0.05)
    lat = imu.accel_lat - tr.speed * tr.yaw_rate + 0.01
    assert lat.abs().max() == pytest.approx(1.0, abs=0.01)
    assert np.allclose(imu.yaw_rate - tr.yaw_rate, 0.001)


def test_odometry_scale_error():
    s = simulate(small(odom={"scale_error": 0.05, "noise": 0.0}))
    truth_v = np.interp(s.odom.t, s.truth.t, s.truth.speed)
    assert np.allclose(s.odom.speed / truth_v, 1.05)


@pytest.mark.parametrize("overrides, error", [
    ({"field": {"row_spacing": 4.0}}, "row_spacing"),
    ({"field": {"rows": 0}}, "rows"),
    ({"motion": {"turn_speed": 0.0}}, "positive"),
    ({"gnss": {"rate_hz": 7}}, "divide"),
    ({"imu": {"vib_hz": 60.0}}, "Nyquist"),
    ({"gnss": {"outages": [[5.0, 5.0]]}}, "t1 > t0"),
])
def test_invalid_configs_are_rejected(overrides, error):
    with pytest.raises(ValueError, match=error):
        simulate(small(**overrides))
