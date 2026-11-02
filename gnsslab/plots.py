"""Static PNG figures (matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # fixed order: blue, orange, aqua
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 2, "font.size": 10,
})


def _color(i):
    return SERIES[i % len(SERIES)]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def static_plots(results, out_dir) -> list[Path]:
    out = Path(out_dir)
    paths = []

    # 1) scatter per session with CEP50 / CEP95 circles, shared scale
    lim = max(np.max(np.abs(np.concatenate([r.e, r.n]))) for r in results) * 1.1
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.4), squeeze=False)
    ang = np.linspace(0, 2 * np.pi, 200)
    for i, (ax, r) in enumerate(zip(axes[0], results)):
        ax.scatter(r.e, r.n, s=8, color=_color(i), alpha=0.35, linewidths=0)
        for rad, style, name in ((r.cep50, "-", "CEP50"), (r.cep95, "--", "CEP95")):
            ax.plot(r.mean_e + rad * np.cos(ang), r.mean_n + rad * np.sin(ang),
                    style, color=INK2, linewidth=1.2, label=f"{name} {rad:.1f} m")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(r.label)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
    paths.append(_save(fig, out / "static_scatter.png"))

    # 2) horizontal error over time
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, r in enumerate(results):
        ax.plot(r.t / 60, r.r, color=_color(i), linewidth=1.2, label=r.label)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Horizontal error [m]")
    ax.set_title("Horizontal error over time")
    if len(results) > 1:
        ax.legend()
    paths.append(_save(fig, out / "static_error_time.png"))

    # 3) CDF of horizontal error
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(results):
        x = np.sort(r.r)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=_color(i), label=r.label)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.axhline(0.95, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Horizontal error [m]")
    ax.set_ylabel("Share of fixes")
    ax.set_title("Error distribution (CDF)")
    if len(results) > 1:
        ax.legend(loc="lower right")
    paths.append(_save(fig, out / "static_cdf.png"))
    return paths


def drive_plots(run, study, out_dir) -> list[Path]:
    from .outage import drift_curve

    out = Path(out_dir)
    paths = []
    longest = max(study)
    outs = study[longest]

    # 1) track with the outage predictions of the longest outage length
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(run.x[:, 0], run.x[:, 1], color=AXIS, linewidth=1.5, label="Filter with GNSS (reference)")
    for j, o in enumerate(outs):
        seg = np.vstack([run.x[o.k0, :2], o.kf])
        ax.plot(seg[:, 0], seg[:, 1], color=SERIES[0], linewidth=2,
                label=f"Prediction during {longest:g} s outage" if j == 0 else None)
        ax.plot(*run.x[o.k0, :2], "o", color=SERIES[0], markersize=4)
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Drive track and simulated GNSS outages")
    ax.legend(loc="best")
    paths.append(_save(fig, out / "drive_track.png"))

    # 2) drift vs time since GNSS loss
    centers, curves = drift_curve(outs)
    if len(centers):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (name, label) in enumerate((("err_kf", "Kalman (constant velocity)"),
                                           ("err_hold", "Hold last position"))):
            med, p95 = curves[name]
            ax.plot(centers, med, color=_color(i), label=f"{label}, median")
            ax.plot(centers, p95, "--", color=_color(i), linewidth=1.2, label=f"{label}, 95th pct")
        ax.set_xlabel("Time since GNSS loss [s]")
        ax.set_ylabel("Drift [m]")
        ax.set_title(f"Position drift during GNSS outage ({len(outs)} outages)")
        ax.legend(loc="upper left")
        paths.append(_save(fig, out / "drive_drift.png"))
    return paths
