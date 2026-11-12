"""Static test: the phone does not move, so every change in position is error."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import geodetic_to_enu
from .io import Track


@dataclass
class StaticResult:
    label: str
    source: str
    ref_kind: str            # "mean" (precision only) or "reference" (true accuracy)
    ref: tuple               # (lat, lon, h) used as origin
    count: int
    duration_s: float
    mean_e: float            # bias east  [m] (0 by construction when ref_kind == "mean")
    mean_n: float            # bias north [m]
    std_e: float
    std_n: float
    h_rms: float             # horizontal RMS error [m]
    cep50: float             # radius containing 50 % of fixes [m]
    cep95: float             # radius containing 95 % of fixes [m]
    h_max: float
    v_std: float             # vertical spread [m], NaN if no altitude
    acc_median: float        # median reported accuracy [m], NaN if not logged
    within_acc: float        # share of fixes whose real error <= reported accuracy (ideal ~0.68)
    corr_time_s: float       # how long errors stay correlated (autocorrelation < 1/e)
    t: np.ndarray = field(repr=False, default=None)
    e: np.ndarray = field(repr=False, default=None)
    n: np.ndarray = field(repr=False, default=None)
    u: np.ndarray = field(repr=False, default=None)
    r: np.ndarray = field(repr=False, default=None)

    def summary(self) -> dict:
        skip = {"t", "e", "n", "u", "r"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}


def correlation_time(t, series) -> float:
    """Lag where the (averaged) autocorrelation of the error first drops below 1/e.

    Assumes roughly regular sampling (true for 1 Hz phone logs).
    """
    dt = float(np.median(np.diff(t)))
    acfs = []
    for x in series:
        x = np.asarray(x, float) - np.mean(x)
        m = len(x)
        spec = np.fft.rfft(x, 2 * m)                   # zero padding -> linear, not circular
        acf = np.fft.irfft(spec * np.conj(spec))[:m]
        if acf[0] > 0:
            acfs.append(acf / acf[0])
    if not acfs:
        return float("nan")
    acf = np.mean(acfs, axis=0)
    below = np.nonzero(acf < 1 / np.e)[0]
    return float(below[0] * dt) if below.size else float("nan")


def analyze_static(track: Track, label: str | None = None, ref=None) -> StaticResult:
    has_alt = np.isfinite(track.alt).any()
    if ref is None:
        lat0, lon0 = float(np.mean(track.lat)), float(np.mean(track.lon))
        h0 = float(np.nanmean(track.alt)) if has_alt else 0.0
        kind = "mean"
    else:
        lat0, lon0 = float(ref[0]), float(ref[1])
        h0 = float(ref[2]) if len(ref) > 2 else (float(np.nanmean(track.alt)) if has_alt else 0.0)
        kind = "reference"

    e, n, u = geodetic_to_enu(track.lat, track.lon, track.alt, lat0, lon0, h0)
    r = np.hypot(e, n)
    acc_ok = np.isfinite(track.acc)

    return StaticResult(
        label=label or track.source, source=track.source, ref_kind=kind, ref=(lat0, lon0, h0),
        count=len(track), duration_s=track.duration,
        mean_e=float(np.mean(e)), mean_n=float(np.mean(n)),
        std_e=float(np.std(e)), std_n=float(np.std(n)),
        h_rms=float(np.sqrt(np.mean(r ** 2))),
        cep50=float(np.percentile(r, 50)), cep95=float(np.percentile(r, 95)),
        h_max=float(np.max(r)),
        v_std=float(np.nanstd(u)) if has_alt else float("nan"),
        acc_median=float(np.median(track.acc[acc_ok])) if acc_ok.any() else float("nan"),
        within_acc=float(np.mean(r[acc_ok] <= track.acc[acc_ok])) if acc_ok.any() else float("nan"),
        corr_time_s=correlation_time(track.t, [e, n]),
        t=track.t - track.t[0], e=e, n=n, u=u, r=r,
    )
