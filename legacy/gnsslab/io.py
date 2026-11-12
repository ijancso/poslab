"""Load position logs from Android logging apps into one common Track structure.

Supported inputs:
  * Google GnssLogger .txt  - the 'Fix' records (raw measurements are ignored)
  * GPX                     - e.g. GPSLogger, most tracking apps
  * CSV                     - GPSLogger CSV or any CSV with time/lat/lon columns
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class Track:
    t: np.ndarray       # unix time [s]
    lat: np.ndarray     # [deg]
    lon: np.ndarray     # [deg]
    alt: np.ndarray     # [m], NaN if unknown
    acc: np.ndarray     # reported horizontal accuracy [m] (Android: 68% radius), NaN if unknown
    speed: np.ndarray   # reported speed [m/s], NaN if unknown
    source: str = ""

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def select(self, idx) -> "Track":
        return Track(self.t[idx], self.lat[idx], self.lon[idx], self.alt[idx],
                     self.acc[idx], self.speed[idx], self.source)


def make_track(t, lat, lon, alt=None, acc=None, speed=None, source="") -> Track:
    """Build a clean Track: drop invalid rows, sort by time, drop duplicate timestamps."""
    t = np.asarray(t, float)
    n = len(t)

    def col(v):
        return np.full(n, np.nan) if v is None else np.asarray(v, float)

    tr = Track(t, col(lat), col(lon), col(alt), col(acc), col(speed), source)
    tr = tr.select(np.isfinite(tr.t) & np.isfinite(tr.lat) & np.isfinite(tr.lon))
    tr = tr.select(np.argsort(tr.t, kind="stable"))
    keep = np.ones(len(tr), bool)
    keep[1:] = np.diff(tr.t) > 0
    tr = tr.select(keep)
    if len(tr) < 2:
        raise ValueError(f"{source}: fewer than 2 valid fixes")
    return tr


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_time(v: str) -> float:
    """Unix seconds from: unix seconds, unix milliseconds, or an ISO 8601 string."""
    v = v.strip()
    try:
        x = float(v)
        return x / 1000.0 if x > 1e11 else x   # > 1e11 can only be milliseconds
    except ValueError:
        pass
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------- GnssLogger

GNSSLOGGER_DEFAULT_FIX = ("Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,"
                          "SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis").split(",")


def load_gnsslogger(path, provider: str | None = "gps") -> Track:
    header, rows = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if body.startswith("Fix,"):
                    header = [h.strip() for h in body.split(",")]
                continue
            if line.startswith("Fix,"):
                rows.append(line.split(","))

    idx = {c.lower(): i for i, c in enumerate(header or GNSSLOGGER_DEFAULT_FIX)}
    for need in ("latitudedegrees", "longitudedegrees", "unixtimemillis"):
        if need not in idx:
            raise ValueError(f"{path}: Fix header has no '{need}' column")

    def get(row, name):
        i = idx.get(name)
        return row[i] if i is not None and i < len(row) else ""

    if provider and "provider" in idx:
        found = sorted({get(r, "provider") for r in rows})
        rows = [r for r in rows if get(r, "provider").lower() == provider.lower()]
        if not rows:
            raise ValueError(f"{path}: no '{provider}' fixes; providers in file: {found}")

    return make_track(
        t=[_float(get(r, "unixtimemillis")) / 1000.0 for r in rows],
        lat=[_float(get(r, "latitudedegrees")) for r in rows],
        lon=[_float(get(r, "longitudedegrees")) for r in rows],
        alt=[_float(get(r, "altitudemeters")) for r in rows],
        acc=[_float(get(r, "accuracymeters")) for r in rows],
        speed=[_float(get(r, "speedmps")) for r in rows],
        source=str(path),
    )


# ---------------------------------------------------------------------- GPX

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()   # strip XML namespace


def load_gpx(path) -> Track:
    root = ET.parse(path).getroot()
    t, lat, lon, alt, acc, speed = [], [], [], [], [], []
    for el in root.iter():
        if _local(el.tag) != "trkpt":
            continue
        vals = {}
        for child in el.iter():          # includes <extensions> children
            if child is el or not (child.text and child.text.strip()):
                continue
            vals.setdefault(_local(child.tag), child.text.strip())
        if "time" not in vals:
            continue
        t.append(parse_time(vals["time"]))
        lat.append(_float(el.get("lat")))
        lon.append(_float(el.get("lon")))
        alt.append(_float(vals.get("ele")))
        acc.append(_float(vals.get("accuracy")))
        speed.append(_float(vals.get("speed")))
    return make_track(t, lat, lon, alt, acc, speed, source=str(path))


# ---------------------------------------------------------------------- CSV

CSV_ALIASES = {
    "t": ["unix_time_ms", "unixtimemillis", "timestamp_ms", "time", "timestamp", "datetime", "date_time"],
    "lat": ["lat", "latitude", "latitudedegrees"],
    "lon": ["lon", "lng", "long", "longitude", "longitudedegrees"],
    "alt": ["alt", "altitude", "elevation", "ele", "altitudemeters"],
    "acc": ["accuracy", "acc", "horizontal_accuracy", "accuracymeters", "hacc"],
    "speed": ["speed", "speedmps", "speed_mps"],
}


def load_csv(path, provider: str | None = "gps") -> Track:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError(f"{path}: no data rows")

    cols = {}
    for key, names in CSV_ALIASES.items():
        cols[key] = next((n for n in names if n in rows[0]), None)
    if not (cols["t"] and cols["lat"] and cols["lon"]):
        raise ValueError(f"{path}: need time, lat and lon columns; got {list(rows[0])}")

    # GPSLogger mixes providers; keep only GNSS fixes if any row is marked as such
    if provider and "provider" in rows[0]:
        matching = [r for r in rows if r["provider"].lower() == provider.lower()]
        if matching:
            rows = matching

    def column(key):
        name = cols[key]
        return None if name is None else [_float(r[name]) for r in rows]

    return make_track(
        t=[parse_time(r[cols["t"]]) if r[cols["t"]] else float("nan") for r in rows],
        lat=column("lat"), lon=column("lon"), alt=column("alt"),
        acc=column("acc"), speed=column("speed"), source=str(path),
    )


# ------------------------------------------------------------------ dispatch

def load_track(path, provider: str | None = "gps") -> Track:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".gpx":
        return load_gpx(p)
    if ext == ".csv":
        return load_csv(p, provider)
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if "Fix," in line:
                return load_gnsslogger(p, provider)
            if "<gpx" in line:
                return load_gpx(p)
            if i > 2000:
                break
    return load_csv(p, provider)
