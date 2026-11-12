"""Drive test: cut GNSS out for N seconds and measure how far the filter drifts.

Method
  1. Run the Kalman filter once over the whole drive with all GNSS fixes
     (the "reference" solution).
  2. Pick outage start epochs k0 (vehicle moving, filter settled).
  3. For each outage, start from the filter state at k0 and predict forward
     with no measurements. Because the data before k0 is identical, this is
     exactly what a full re-run with those fixes removed would produce.
  4. Drift = distance between the outage prediction and the reference solution.

Baseline "hold": freeze the last position. Anything useful must beat it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman import FilterRun, predict_ahead


@dataclass
class Outage:
    duration: float
    k0: int                  # last epoch with GNSS before the outage
    idx: np.ndarray          # epochs inside the outage
    elapsed: np.ndarray      # seconds since outage start
    ref: np.ndarray          # (M, 2) reference positions (filter with GNSS)
    kf: np.ndarray           # (M, 2) filter prediction without GNSS
    err_kf: np.ndarray       # drift of the filter [m]
    err_hold: np.ndarray     # drift of 'hold last position' [m]
    sigma_kf: np.ndarray     # filter's own predicted horizontal drms [m]
    nees: np.ndarray         # normalised error: drift^T P^-1 drift (chi-square, 2 dof, if honest)

CHI2_2DOF_95 = 5.991


def plan_outages(run: FilterRun, duration, warmup=30.0, gap=10.0, min_speed=3.0) -> list[int]:
    """Non-overlapping outage starts: after warmup, moving, separated by gap seconds."""
    t = run.t
    speed = np.hypot(run.x[:, 2], run.x[:, 3])
    starts = []
    k = int(np.searchsorted(t, t[0] + warmup))
    while k < len(t):
        if t[k] + duration > t[-1]:
            break
        if speed[k] >= min_speed and run.updated[k]:
            starts.append(k)
            k = int(np.searchsorted(t, t[k] + duration + gap))
        else:
            k += 1
    return starts


def simulate_outage(run: FilterRun, k0: int, duration: float) -> Outage:
    t = run.t
    idx = np.nonzero((t > t[k0]) & (t <= t[k0] + duration))[0]
    kf, cov, drms = predict_ahead(run.x[k0], run.P[k0], t[k0], t[idx], run.q)
    ref = run.x[idx, :2]
    hold = run.x[k0, :2]
    d = kf - ref
    nees = np.einsum("mi,mij,mj->m", d, np.linalg.inv(cov), d)
    return Outage(
        duration=duration, k0=k0, idx=idx, elapsed=t[idx] - t[k0], ref=ref, kf=kf,
        err_kf=np.hypot(*d.T), err_hold=np.hypot(*(hold - ref).T), sigma_kf=drms, nees=nees,
    )


def outage_study(run: FilterRun, durations, **plan_kw) -> dict[float, list[Outage]]:
    return {d: [simulate_outage(run, k0, d) for k0 in plan_outages(run, d, **plan_kw)]
            for d in durations}


def summarize(study: dict[float, list[Outage]]) -> list[dict]:
    rows = []
    for d, outs in study.items():
        outs = [o for o in outs if len(o.idx)]
        if not outs:
            rows.append({"duration_s": d, "outages": 0})
            continue
        end_kf = np.array([o.err_kf[-1] for o in outs])
        end_hold = np.array([o.err_hold[-1] for o in outs])
        end_sig = np.array([o.sigma_kf[-1] for o in outs])
        end_nees = np.array([o.nees[-1] for o in outs])
        rows.append({
            "duration_s": d,
            "outages": len(outs),
            "kf_end_median_m": float(np.median(end_kf)),
            "kf_end_p95_m": float(np.percentile(end_kf, 95)),
            "kf_max_m": float(np.max(end_kf)),
            "hold_end_median_m": float(np.median(end_hold)),
            "kf_worse_than_hold": float(np.mean(end_kf > end_hold)),
            "kf_drms_end_median_m": float(np.median(end_sig)),
            # if the filter is honest about its uncertainty, ~95 % of drifts fall
            # inside its own 95 % error ellipse
            "inside_95_ellipse": float(np.mean(end_nees <= CHI2_2DOF_95)),
        })
    return rows


def drift_curve(outs: list[Outage], step=1.0):
    """Median and 95th percentile drift as a function of time since GNSS loss."""
    if not outs:
        return np.array([]), {}
    el = np.concatenate([o.elapsed for o in outs])
    bins = np.round(el / step) * step
    curves = {}
    for name in ("err_kf", "err_hold"):
        vals = np.concatenate([getattr(o, name) for o in outs])
        centers = np.unique(bins)
        curves[name] = (
            np.array([np.median(vals[bins == c]) for c in centers]),
            np.array([np.percentile(vals[bins == c], 95) for c in centers]),
        )
    return np.unique(bins), curves
