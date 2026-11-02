"""Coordinate conversions: WGS84 geodetic <-> ECEF <-> local ENU (east, north, up).

All analysis is done in a local ENU frame in metres, because errors and
distances are only meaningful in metres, not in degrees.
"""
from __future__ import annotations

import numpy as np

# WGS84 ellipsoid
A = 6378137.0                 # semi-major axis [m]
F = 1 / 298.257223563         # flattening
E2 = F * (2 - F)              # first eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = np.radians(np.asarray(lat_deg, float))
    lon = np.radians(np.asarray(lon_deg, float))
    h = np.asarray(h, float)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)   # prime vertical radius of curvature
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + h) * np.sin(lat)
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """Iterative inverse; converges to sub-millimetre in a few steps away from the poles."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    h = np.zeros_like(p)
    for _ in range(8):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


def enu_rotation(lat0_deg, lon0_deg):
    """Rotation matrix R so that [e, n, u] = R @ (ecef - ecef_ref)."""
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    sp, cp = np.sin(lat0), np.cos(lat0)
    sl, cl = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-sl,        cl,       0.0],
        [-sp * cl,  -sp * sl,  cp],
        [cp * cl,    cp * sl,  sp],
    ])


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0=0.0):
    """Convert positions to ENU metres relative to (lat0, lon0, h0).

    Missing heights (NaN) are replaced by h0 for the horizontal computation,
    and the returned 'up' is NaN for those points.
    """
    h = np.asarray(h, float)
    missing = ~np.isfinite(h)
    h = np.where(missing, h0, h)
    x, y, z = geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    enu = enu_rotation(lat0, lon0) @ np.vstack([x - x0, y - y0, z - z0])
    e, n, u = enu[0], enu[1], enu[2].copy()
    u[missing] = np.nan
    return e, n, u


def enu_to_geodetic(e, n, u, lat0, lon0, h0=0.0):
    d = enu_rotation(lat0, lon0).T @ np.vstack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)])
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    return ecef_to_geodetic(x0 + d[0], y0 + d[1], z0 + d[2])
