"""Synthetic tractor scenario generator.

A scenario is a YAML file of parameter overrides on top of ``DEFAULTS``. The generator
builds a field pattern (parallel rows joined by headland U-turns), a smooth speed profile,
100 Hz ground truth, and three sensor streams (GNSS, IMU, wheel odometry) with the
configured errors.

Conventions (used everywhere in poslab):
  * local 2D ENU frame in metres: x = east, y = north
  * heading = yaw angle from +x (east), counter-clockwise positive, radians in (-pi, pi]
    -> unicycle model: dx/dt = v cos(heading), dy/dt = v sin(heading)
  * yaw_rate positive counter-clockwise (left turn); lateral acceleration positive to the left

Determinism: all randomness comes from the scenario ``seed``. Each sensor gets its own
child generator (``SeedSequence.spawn``), so changing e.g. the odometry noise does not
change the GNSS noise realisation of the same scenario.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

GENERATOR_VERSION = "1.1"

DEFAULTS: dict[str, Any] = {
    "description": "",
    "seed": 0,
    "smoke": False,
    "field": {
        "rows": 8,              # number of parallel rows
        "row_length": 200.0,    # [m]
        "row_spacing": 6.0,     # [m] distance between neighbouring rows
        "turn_radius": 3.0,     # [m] headland turn arc radius
    },
    "motion": {
        "rate_hz": 100,         # truth sample rate
        "row_speed": 2.5,       # [m/s]
        "turn_speed": 1.2,      # [m/s]
        "max_accel": 0.3,       # [m/s^2] limit for speed changes between row and turn
    },
    "gnss": {
        "rate_hz": 10,
        "sigma": 0.02,          # [m] white position noise per axis (0.02 RTK-like, 1.5 standalone)
        "bias_rw": 0.0,         # [m/sqrt(s)] slow bias random walk per axis, 0 = off
        "multipath": {
            "prob": 0.0,        # probability per epoch that a spike starts
            "amp": 0.0,         # [m] spike offset (random direction)
            "epochs": 3,        # spike length in epochs
        },
        "outages": [],          # [[t0, t1], ...] seconds, no GNSS for t0 <= t < t1
    },
    "imu": {
        "rate_hz": 100,
        "accel_noise": 0.05,    # [m/s^2] white noise per axis
        "accel_bias_long": 0.02,  # [m/s^2]
        "accel_bias_lat": -0.01,  # [m/s^2]
        "gyro_noise": 0.002,    # [rad/s] white noise
        "gyro_bias": 0.001,     # [rad/s]
        "vib_hz": 27.0,         # [Hz] engine vibration frequency
        "vib_amp": 0.3,         # [m/s^2] engine vibration amplitude on both accel axes
    },
    "odom": {
        "rate_hz": 50,
        "scale_error": 0.0,     # wheel slip / tyre radius error, +0.02 = reads 2 % high
        "noise": 0.02,          # [m/s] white noise
    },
}

_DS = 0.01  # [m] arc-length step used to build the path before resampling in time


@dataclass
class Scenario:
    """One generated scenario: truth, sensor streams and resolved metadata."""

    name: str
    truth: pd.DataFrame     # t, x, y, heading, speed, yaw_rate, accel_long
    gnss: pd.DataFrame      # t, x, y, sigma (receiver-reported 1-sigma per axis)
    imu: pd.DataFrame       # t, accel_long, accel_lat, yaw_rate
    odom: pd.DataFrame      # t, speed
    meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ configuration

def merge_config(overrides: dict[str, Any], base: dict[str, Any] | None = None,
                 path: str = "") -> dict[str, Any]:
    """Deep-merge ``overrides`` into a copy of ``base`` (default: DEFAULTS).

    Unknown keys raise ``KeyError`` so that typos in scenario files are caught early.
    """
    out = copy.deepcopy(DEFAULTS if base is None else base)
    for key, value in (overrides or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in out:
            raise KeyError(f"unknown scenario parameter '{where}'")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"scenario parameter '{where}' must be a mapping")
            out[key] = merge_config(value, out[key], where)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a scenario YAML file and merge it with the defaults; adds ``name``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    cfg = merge_config(overrides)
    cfg["name"] = path.stem
    return cfg


# ------------------------------------------------------------------ path and motion

def _segments(fcfg: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Field pattern as (kind, length [m], curvature [1/m]) segments.

    Rows run north/south, row i at x = i * row_spacing, starting north-bound at the origin.
    A headland turn is a quarter arc, an optional straight (row_spacing - 2 r) and a
    quarter arc; with the default spacing = 2 r it is a plain semicircle. Turns alternate
    right (clockwise, at the north end) and left (at the south end).
    """
    rows, length = int(fcfg["rows"]), float(fcfg["row_length"])
    spacing, r = float(fcfg["row_spacing"]), float(fcfg["turn_radius"])
    if rows < 1 or length <= 0 or r <= 0:
        raise ValueError("field needs rows >= 1, row_length > 0, turn_radius > 0")
    if spacing < 2 * r - 1e-9:
        raise ValueError("row_spacing must be >= 2 * turn_radius")
    segs: list[tuple[str, float, float]] = []
    for i in range(rows):
        segs.append(("row", length, 0.0))
        if i == rows - 1:
            break
        k = (-1.0 if i % 2 == 0 else 1.0) / r
        segs.append(("turn", np.pi * r / 2, k))
        if spacing - 2 * r > 1e-9:
            segs.append(("turn", spacing - 2 * r, 0.0))
        segs.append(("turn", np.pi * r / 2, k))
    return segs


def _path(segs: list[tuple[str, float, float]]) -> dict[str, np.ndarray]:
    """Sample the path in arc length. Each segment gets its own exact grid so that the
    heading change of every arc is exact (no accumulated rounding at segment borders)."""
    s_parts, k_parts, turn_parts = [np.zeros(1)], [], []
    s0 = 0.0
    for kind, length, k in segs:
        n = max(1, int(np.ceil(length / _DS)))
        s_parts.append(s0 + np.linspace(0.0, length, n + 1)[1:])
        k_parts.append(np.full(n, k))            # curvature of each interval
        turn_parts.append(np.full(n, kind == "turn"))
        s0 += length
    s = np.concatenate(s_parts)
    k_int = np.concatenate(k_parts)
    in_turn_int = np.concatenate(turn_parts)
    ds = np.diff(s)
    heading = np.pi / 2 + np.concatenate([[0.0], np.cumsum(k_int * ds)])
    hmid = 0.5 * (heading[1:] + heading[:-1])   # exact for constant curvature (chord)
    x = np.concatenate([[0.0], np.cumsum(np.cos(hmid) * ds * _chord(k_int, ds))])
    y = np.concatenate([[0.0], np.cumsum(np.sin(hmid) * ds * _chord(k_int, ds))])
    # point values: an endpoint between a row and a turn belongs to the turn
    k_pt = np.concatenate([k_int[:1], k_int])
    turn_pt = np.concatenate([in_turn_int[:1], in_turn_int])
    turn_pt[:-1] |= in_turn_int
    return {"s": s, "x": x, "y": y, "heading": heading, "k": k_pt, "turn": turn_pt}


def _chord(k: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Chord / arc length ratio sin(k ds / 2) / (k ds / 2), equal to 1 on straights."""
    half = 0.5 * k * ds
    return np.where(np.abs(half) < 1e-12, 1.0, np.sin(half) / np.where(half == 0, 1, half))


def _speed_profile(s: np.ndarray, turn: np.ndarray, mcfg: dict[str, Any]) -> np.ndarray:
    """Speed along the path: row_speed on rows, turn_speed in turns, and constant
    acceleration ramps (|a| <= max_accel) in between, so the vehicle has already slowed
    down when it enters a turn. Two passes over arc length: v^2 <= v_prev^2 + 2 a ds."""
    v_lim = np.where(turn, mcfg["turn_speed"], mcfg["row_speed"]).astype(float)
    a2 = 2.0 * float(mcfg["max_accel"])
    ds = np.diff(s)
    v = v_lim.copy()
    for i in range(1, len(v)):                       # accelerate no faster than a_max
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + a2 * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):              # brake no harder than a_max
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + a2 * ds[i]))
    return v


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - angle, 2 * np.pi)


def make_truth(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Ground truth at ``motion.rate_hz`` plus the row/turn time intervals."""
    mcfg = cfg["motion"]
    if min(mcfg["row_speed"], mcfg["turn_speed"], mcfg["max_accel"]) <= 0:
        raise ValueError("speeds and max_accel must be positive")
    segs = _segments(cfg["field"])
    p = _path(segs)
    s = p["s"]
    v = _speed_profile(s, p["turn"], mcfg)
    t_of_s = np.concatenate([[0.0], np.cumsum(2.0 * np.diff(s) / (v[1:] + v[:-1]))])
    accel_s = v * np.gradient(v, s)                  # dv/dt = v dv/ds

    fs = float(mcfg["rate_hz"])
    t = np.arange(int(np.floor(t_of_s[-1] * fs)) + 1) / fs
    s_t = np.interp(t, t_of_s, s)
    heading = np.interp(s_t, s, p["heading"])
    speed = np.interp(s_t, s, v)
    # curvature is piecewise constant: take the value of the interval the sample falls in
    idx = np.clip(np.searchsorted(s, s_t, side="right") - 1, 0, len(s) - 2)
    k = p["k"][1:][idx]
    truth = pd.DataFrame({
        "t": t,
        "x": np.interp(s_t, s, p["x"]),
        "y": np.interp(s_t, s, p["y"]),
        "heading": _wrap(heading),
        "speed": speed,
        "yaw_rate": k * speed,
        "accel_long": np.interp(s_t, s, accel_s),
    })

    intervals, s0 = [], 0.0
    for kind, length, _ in segs:
        t0, t1 = np.interp([s0, s0 + length], s, t_of_s)
        if intervals and intervals[-1]["kind"] == kind:
            intervals[-1]["t1"] = float(t1)          # merge the parts of one turn
        else:
            intervals.append({"kind": kind, "t0": float(t0), "t1": float(t1)})
        s0 += length
    return truth, intervals


# ------------------------------------------------------------------ sensors

def _decimate(truth: pd.DataFrame, truth_hz: float, rate_hz: float) -> pd.DataFrame:
    """Every n-th truth sample; the sensor rate must divide the truth rate."""
    step = truth_hz / rate_hz
    if rate_hz <= 0 or abs(step - round(step)) > 1e-9:
        raise ValueError(f"sensor rate {rate_hz} Hz must divide the truth rate {truth_hz} Hz")
    return truth.iloc[:: int(round(step))].reset_index(drop=True)


def make_gnss(truth: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """GNSS positions: truth + white noise + bias random walk + multipath spikes,
    with the outage windows removed.

    ``sigma`` is what the receiver *reports* (like Android's accuracy field): the nominal
    white-noise level. It does not include bias or multipath - receivers are optimistic.
    """
    g = cfg["gnss"]
    tr = _decimate(truth, cfg["motion"]["rate_hz"], g["rate_hz"])
    n, dt = len(tr), 1.0 / g["rate_hz"]
    err = rng.normal(0.0, g["sigma"], (n, 2))
    bias = np.cumsum(rng.normal(0.0, g["bias_rw"] * np.sqrt(dt), (n, 2)), axis=0)
    mp = g["multipath"]
    spike = np.zeros((n, 2))
    starts = np.flatnonzero(rng.random(n) < mp["prob"])
    angles = rng.uniform(0.0, 2 * np.pi, len(starts))
    for k0, ang in zip(starts, angles, strict=True):
        spike[k0:k0 + int(mp["epochs"])] = mp["amp"] * np.array([np.cos(ang), np.sin(ang)])
    pos = tr[["x", "y"]].to_numpy() + err + bias + spike
    keep = np.ones(n, bool)
    for t0, t1 in g["outages"]:
        if t1 <= t0:
            raise ValueError(f"outage window [{t0}, {t1}] must have t1 > t0")
        keep &= ~((tr["t"].to_numpy() >= t0) & (tr["t"].to_numpy() < t1))
    out = pd.DataFrame({"t": tr["t"], "x": pos[:, 0], "y": pos[:, 1],
                        "sigma": np.full(n, float(g["sigma"]))})
    return out[keep].reset_index(drop=True)


def make_imu(truth: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """IMU: longitudinal and lateral (centripetal = v * yaw_rate) acceleration and yaw rate,
    each with white noise and a constant bias; engine vibration added to both accel axes."""
    c = cfg["imu"]
    tr = _decimate(truth, cfg["motion"]["rate_hz"], c["rate_hz"])
    n, t, gyro = len(tr), tr["t"].to_numpy(), tr["yaw_rate"].to_numpy()
    if c["vib_hz"] >= c["rate_hz"] / 2:
        raise ValueError("vib_hz must be below the IMU Nyquist frequency")
    phase = rng.uniform(0.0, 2 * np.pi, 2)
    vib = c["vib_amp"] * np.sin(2 * np.pi * c["vib_hz"] * t[:, None] + phase)
    a_long = tr["accel_long"].to_numpy() + c["accel_bias_long"] + vib[:, 0]
    a_lat = (tr["speed"] * tr["yaw_rate"]).to_numpy() + c["accel_bias_lat"] + vib[:, 1]
    return pd.DataFrame({
        "t": t,
        "accel_long": a_long + rng.normal(0.0, c["accel_noise"], n),
        "accel_lat": a_lat + rng.normal(0.0, c["accel_noise"], n),
        "yaw_rate": gyro + c["gyro_bias"] + rng.normal(0.0, c["gyro_noise"], n),
    })


def make_odom(truth: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Wheel speed with a scale-factor error (slip) and white noise."""
    c = cfg["odom"]
    tr = _decimate(truth, cfg["motion"]["rate_hz"], c["rate_hz"])
    speed = tr["speed"].to_numpy() * (1.0 + c["scale_error"]) + rng.normal(0.0, c["noise"], len(tr))
    return pd.DataFrame({"t": tr["t"], "speed": speed})


# ------------------------------------------------------------------ top level

def simulate(cfg: dict[str, Any]) -> Scenario:
    """Generate truth and all sensor streams for a resolved config (see ``merge_config``)."""
    truth, intervals = make_truth(cfg)
    rng_gnss, rng_imu, rng_odom = (np.random.default_rng(s)
                                   for s in np.random.SeedSequence(int(cfg["seed"])).spawn(3))
    meta = copy.deepcopy(cfg)
    meta.setdefault("name", "unnamed")
    meta["generator_version"] = GENERATOR_VERSION
    meta["duration_s"] = float(truth["t"].iloc[-1])
    meta["intervals"] = intervals
    return Scenario(
        name=meta["name"],
        truth=truth,
        gnss=make_gnss(truth, cfg, rng_gnss),
        imu=make_imu(truth, cfg, rng_imu),
        odom=make_odom(truth, cfg, rng_odom),
        meta=meta,
    )


def write_scenario(scn: Scenario, out_dir: str | Path) -> Path:
    """Write truth.csv, gnss.csv, imu.csv, odom.csv and meta.yaml to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("truth", "gnss", "imu", "odom"):
        getattr(scn, name).to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
    with open(out / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(scn.meta, f, sort_keys=False)
    return out


def generate_file(yaml_path: str | Path, out_root: str | Path) -> Path:
    """Load a scenario YAML, simulate it and write it to ``out_root/<name>/``."""
    cfg = load_config(yaml_path)
    return write_scenario(simulate(cfg), Path(out_root) / cfg["name"])
