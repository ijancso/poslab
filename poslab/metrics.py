"""Accuracy metrics of an estimate against ground truth (or, for recordings, static scatter).

All distances are in metres. Metric keys carry their unit (``_m``) so that the JSON files,
the baseline and the report are self-explanatory.

Definitions
-----------
* horizontal error e = |p_est - p_true| at the estimate timestamps (truth interpolated).
* CEP50 / CEP95: radius of the circle (centred on the truth) holding 50 % / 95 % of the
  errors, i.e. the 50th / 95th percentile of e.
* cross-track error: the component of the error perpendicular to the truth heading at the
  same time, xt = -ex sin(h) + ey cos(h) (positive = left of the path). On straight rows it
  is the distance to the truth path - the number that decides whether the implement hits
  the crop row. Along-track error (timing) is reported separately and matters less.
* outage drift: the horizontal error at the last estimate inside each GNSS outage, and the
  error as a function of time into the outage (``outage_drift_curve``).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _interp_truth(est: pd.DataFrame, truth: pd.DataFrame) -> dict[str, np.ndarray]:
    """Truth x, y and heading at the estimate times (heading interpolated unwrapped)."""
    t = est["t"].to_numpy()
    if t.min() < truth["t"].iloc[0] - 1e-9 or t.max() > truth["t"].iloc[-1] + 1e-9:
        raise ValueError("estimate times extend beyond the truth time range")
    h = np.interp(t, truth["t"], np.unwrap(truth["heading"].to_numpy()))
    return {"x": np.interp(t, truth["t"], truth["x"]),
            "y": np.interp(t, truth["t"], truth["y"]), "heading": h}


def error_components(est: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-sample errors: t, ex, ey, horizontal, along-track, cross-track [m]."""
    tr = _interp_truth(est, truth)
    ex = est["x"].to_numpy() - tr["x"]
    ey = est["y"].to_numpy() - tr["y"]
    c, s = np.cos(tr["heading"]), np.sin(tr["heading"])
    return pd.DataFrame({
        "t": est["t"].to_numpy(), "ex": ex, "ey": ey, "horizontal": np.hypot(ex, ey),
        "along": ex * c + ey * s, "cross": -ex * s + ey * c,
    })


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def outage_drift_curve(err: pd.DataFrame, outages: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Horizontal error vs time into each outage: columns outage (1-based), tau, error."""
    parts = []
    for i, (t0, t1) in enumerate(outages, start=1):
        m = (err["t"] >= t0) & (err["t"] < t1)
        parts.append(pd.DataFrame({"outage": i, "tau": err["t"][m] - t0,
                                   "error": err["horizontal"][m]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["outage", "tau", "error"])


def compute_metrics(est: pd.DataFrame, truth: pd.DataFrame,
                    outages: Sequence[Sequence[float]] = (), warmup_s: float = 5.0
                    ) -> dict[str, Any]:
    """Flat metric dict for one (scenario, algorithm) run.

    The first ``warmup_s`` seconds are excluded (filter convergence from a cold start); the
    outage metrics always use the full estimate.
    """
    err_all = error_components(est, truth)
    err = err_all[err_all["t"] >= err_all["t"].iloc[0] + warmup_s]
    if err.empty:
        raise ValueError("no estimates left after the warm-up period")
    e, xt = err["horizontal"].to_numpy(), err["cross"].to_numpy()
    m: dict[str, Any] = {
        "n_samples": int(len(err)),
        "h_rms_m": _rms(e),
        "h_cep50_m": float(np.percentile(e, 50)),
        "h_cep95_m": float(np.percentile(e, 95)),
        "h_max_m": float(e.max()),
        "xt_rms_m": _rms(xt),
        "xt_max_m": float(np.abs(xt).max()),
        "at_rms_m": _rms(err["along"].to_numpy()),
    }
    if len(outages):
        curve = outage_drift_curve(err_all, outages)
        ends = []
        for i in range(1, len(outages) + 1):
            c = curve[curve["outage"] == i]
            if c.empty:
                raise ValueError(f"no estimates inside outage {i} {list(outages[i - 1])}")
            ends.append(float(c["error"].iloc[-1]))
            m[f"outage{i}_end_err_m"] = ends[-1]
        m["outage_end_err_max_m"] = max(ends)
    return m


def static_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Scatter of a stationary recording around its own mean (no truth needed).

    std east/north and CEP50/95 of the distance to the mean position (precision, not
    absolute accuracy: a constant offset of the whole session is invisible here).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        raise ValueError("need at least 2 positions")
    d = np.hypot(x - x.mean(), y - y.mean())
    return {
        "n_samples": int(len(x)),
        "std_east_m": float(np.std(x, ddof=1)),
        "std_north_m": float(np.std(y, ddof=1)),
        "cep50_m": float(np.percentile(d, 50)),
        "cep95_m": float(np.percentile(d, 95)),
    }
