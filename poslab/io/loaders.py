"""Load generated scenarios and GnssLogger phone recordings.

Recordings are converted to the same local ENU frame as the synthetic data, relative to the
first GNSS fix, with an equirectangular (local tangent plane) approximation:

    x = (lon - lon0) * N(lat0) * cos(lat0)       N = prime-vertical radius of curvature
    y = (lat - lat0) * M(lat0)                   M = meridian radius of curvature

(WGS84 radii at lat0, angles in radians). The approximation error grows roughly with
d^2 tan(lat0) / R for a distance d from the origin (the cos(lat) scale changes as you move
north). Checked against an exact ECEF -> ENU conversion at 47 deg N: ~2 mm at 100 m, ~0.2 m
at 1 km. Fine for a field or a static session, not for long drives (the legacy
``gnsslab.geo`` does the exact conversion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from poslab.sim.scenario import Scenario

SCENARIO_COLUMNS: dict[str, list[str]] = {
    "truth": ["t", "x", "y", "heading", "speed", "yaw_rate", "accel_long"],
    "gnss": ["t", "x", "y", "sigma"],
    "imu": ["t", "accel_long", "accel_lat", "yaw_rate"],
    "odom": ["t", "speed"],
}

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
# Android reports accuracy as the radius of 68 % confidence. For a circular 2D Gaussian
# with per-axis sigma s, P(r < k s) = 1 - exp(-k^2 / 2) = 0.68  ->  k = 1.51.
ANDROID_ACC_TO_SIGMA = 1.0 / 1.51


def load_scenario(path: str | Path) -> Scenario:
    """Load a generated scenario directory (``data/generated/<name>/``)."""
    path = Path(path)
    if not (path / "meta.yaml").exists():
        raise FileNotFoundError(f"{path}: no meta.yaml - run 'python -m poslab generate' first")
    with open(path / "meta.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    frames: dict[str, pd.DataFrame] = {}
    for name, cols in SCENARIO_COLUMNS.items():
        df = pd.read_csv(path / f"{name}.csv")
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{path / name}.csv: missing columns {sorted(missing)}")
        frames[name] = df[cols]
    return Scenario(name=meta.get("name", path.name), meta=meta, **frames)


# ------------------------------------------------------------------ recordings

@dataclass
class Recording:
    """A phone recording: GNSS fixes in local ENU (t in seconds since the first fix)."""

    name: str
    gnss: pd.DataFrame                      # t, x, y, sigma
    meta: dict[str, Any] = field(default_factory=dict)


def latlon_to_local(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to local east/north metres around (lat0, lon0) [deg]."""
    phi0 = np.radians(lat0)
    w = np.sqrt(1.0 - WGS84_E2 * np.sin(phi0) ** 2)
    n_radius = WGS84_A / w
    m_radius = WGS84_A * (1.0 - WGS84_E2) / w**3
    x = np.radians(np.asarray(lon, float) - lon0) * n_radius * np.cos(phi0)
    y = np.radians(np.asarray(lat, float) - lat0) * m_radius
    return x, y


def load_gnsslogger(path: str | Path, provider: str = "GPS") -> Recording:
    """Parse the ``Fix`` lines of a Google GnssLogger .txt file.

    Columns are found by name from the ``# Fix,...`` header line (the column set differs
    between app versions). Only fixes of ``provider`` are kept (GPS = the GNSS chipset;
    FLP/NLP are fused/network locations and would contaminate the statistics).
    """
    path = Path(path)
    header: list[str] | None = None
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if body.startswith("Fix,"):
                    header = [h.strip().lower() for h in body.split(",")]
            elif line.startswith("Fix,"):
                rows.append([v.strip() for v in line.split(",")])
    if header is None:
        raise ValueError(f"{path}: no '# Fix,...' header line (is this a GnssLogger file?)")
    need = ["provider", "latitudedegrees", "longitudedegrees", "accuracymeters", "unixtimemillis"]
    missing = [c for c in need if c not in header]
    if missing:
        raise ValueError(f"{path}: Fix header lacks columns {missing}")
    df = pd.DataFrame([r[:len(header)] + [""] * (len(header) - len(r)) for r in rows],
                      columns=header)
    df = df[df["provider"].str.upper() == provider.upper()]
    if df.empty:
        raise ValueError(f"{path}: no '{provider}' fixes")
    num = df[need[1:]].apply(pd.to_numeric, errors="coerce").dropna()
    num = num.sort_values("unixtimemillis").drop_duplicates("unixtimemillis")
    if len(num) < 2:
        raise ValueError(f"{path}: fewer than 2 valid '{provider}' fixes")
    lat, lon = num["latitudedegrees"].to_numpy(), num["longitudedegrees"].to_numpy()
    x, y = latlon_to_local(lat, lon, lat[0], lon[0])
    t_ms = num["unixtimemillis"].to_numpy()
    gnss = pd.DataFrame({"t": (t_ms - t_ms[0]) / 1000.0, "x": x, "y": y,
                         "sigma": num["accuracymeters"].to_numpy() * ANDROID_ACC_TO_SIGMA})
    meta = {"source": str(path), "provider": provider, "lat0": float(lat[0]),
            "lon0": float(lon[0]), "t0_unix_ms": int(t_ms[0])}
    return Recording(name=path.stem, gnss=gnss.reset_index(drop=True), meta=meta)


def load_recording(path: str | Path) -> Recording:
    """Load ``data/recordings/<name>/``: one GnssLogger .txt plus an optional meta.yaml
    (e.g. ``description``, ``static: true``, ``phone``). The directory name is the name."""
    path = Path(path)
    txts = sorted(path.glob("*.txt"))
    if len(txts) != 1:
        raise ValueError(f"{path}: expected exactly one GnssLogger .txt, found {len(txts)}")
    rec = load_gnsslogger(txts[0])
    if (path / "meta.yaml").exists():
        with open(path / "meta.yaml", encoding="utf-8") as f:
            rec.meta.update(yaml.safe_load(f) or {})
    rec.name = path.name
    return rec
