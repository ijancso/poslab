import numpy as np

from gnsslab.geo import enu_to_geodetic, geodetic_to_enu


def test_roundtrip_enu_geodetic():
    e = np.array([0.0, 100.0, -2500.0, 12.3])
    n = np.array([0.0, -50.0, 4000.0, 7.7])
    u = np.array([0.0, 10.0, -3.0, 1.0])
    lat, lon, h = enu_to_geodetic(e, n, u, 47.5, 19.0, 120.0)
    e2, n2, u2 = geodetic_to_enu(lat, lon, h, 47.5, 19.0, 120.0)
    assert np.allclose([e2, n2, u2], [e, n, u], atol=1e-4)


def test_one_microdegree_of_latitude_is_about_11_cm():
    _, n, _ = geodetic_to_enu([47.500001], [19.0], [0.0], 47.5, 19.0, 0.0)
    assert abs(n[0] - 0.1112) < 0.001


def test_missing_height_gives_nan_up_but_valid_horizontal():
    e, n, u = geodetic_to_enu([47.5001], [19.0001], [np.nan], 47.5, 19.0, 100.0)
    assert np.isfinite(e[0]) and np.isfinite(n[0]) and np.isnan(u[0])
