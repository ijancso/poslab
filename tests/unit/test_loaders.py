"""GnssLogger recording loader (poslab.io.loaders)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poslab.io.loaders import latlon_to_local, load_gnsslogger, load_recording

FIXTURE = Path(__file__).parent / "fixtures" / "gnsslogger_small.txt"


def test_parses_gps_fixes_only_and_cleans_rows():
    rec = load_gnsslogger(FIXTURE)
    # 7 Fix lines: 1 FLP, 1 duplicate time, 1 missing latitude -> 4 GPS fixes remain;
    # the lower-case 'gps' row with missing trailing columns is still accepted
    assert list(rec.gnss.columns) == ["t", "x", "y", "sigma"]
    np.testing.assert_allclose(rec.gnss.t, [0.0, 1.0, 2.0, 4.0])
    assert rec.meta["lat0"] == 47.0 and rec.meta["t0_unix_ms"] == 1700000000000
    assert rec.name == "gnsslogger_small"


def test_local_enu_conversion_and_sigma():
    g = load_gnsslogger(FIXTURE).gnss
    # reference values from an exact WGS84 ECEF -> ENU conversion (legacy gnsslab.geo)
    np.testing.assert_allclose(g.x, [0.0, 0.0, 0.76056, 0.76056], atol=1e-5)
    np.testing.assert_allclose(g.y, [0.0, 1.111708, 1.111708, 0.0], atol=1e-5)
    # Android accuracy is a 68 % radius; per-axis sigma = accuracy / 1.51
    np.testing.assert_allclose(g.sigma, [2.0, 2.0, 2.0, 1.0], rtol=1e-6)


def test_equirectangular_error_is_small_near_origin():
    x, y = latlon_to_local(np.array([47.0009]), np.array([19.0013]), 47.0, 19.0)
    # exact ENU of this point (about 140 m away) is (98.87114, 100.05459) m
    assert np.hypot(x[0] - 98.87114, y[0] - 100.05459) < 0.003


def test_errors(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("Fix,GPS,47,19\n")
    with pytest.raises(ValueError, match="header"):
        load_gnsslogger(f)
    f.write_text("# Fix,Provider,LatitudeDegrees,LongitudeDegrees,UnixTimeMillis\n")
    with pytest.raises(ValueError, match="accuracymeters"):
        load_gnsslogger(f)
    with pytest.raises(ValueError, match="no 'NLP' fixes"):
        load_gnsslogger(FIXTURE, provider="NLP")
    f.write_text(FIXTURE.read_text().split("Fix,GPS,47.0000100,19.0000000")[0])
    with pytest.raises(ValueError, match="fewer than 2"):
        load_gnsslogger(f)


def test_recording_directory_with_meta(tmp_path):
    d = tmp_path / "balcony_static"
    d.mkdir()
    (d / "gnss_log.txt").write_text(FIXTURE.read_text())
    (d / "meta.yaml").write_text("description: phone on the balcony\nstatic: true\n")
    rec = load_recording(d)
    assert rec.name == "balcony_static" and rec.meta["static"] is True
    assert len(rec.gnss) == 4
    (d / "second.txt").write_text("")
    with pytest.raises(ValueError, match="exactly one"):
        load_recording(d)
