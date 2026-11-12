"""Plot truth and raw sensor data of a generated scenario (a visual sanity check).

Usage (from the repo root, after ``python -m poslab generate``):
    python tools/plot_scenario.py open_sky_rtk [more names ...] [--out out/plots]

Writes ``<out>/<name>.png`` with six panels: trajectory, GNSS error, speed vs odometry,
yaw rate vs gyro, longitudinal accel (raw IMU vs truth) and a 1 s zoom showing the
engine vibration. GNSS outages are shaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from poslab.io.loaders import load_scenario  # noqa: E402
from poslab.sim.scenario import Scenario  # noqa: E402


def _shade_outages(ax: plt.Axes, scn: Scenario) -> None:
    """Grey bands for the GNSS outage windows."""
    for i, (t0, t1) in enumerate(scn.meta["gnss"]["outages"]):
        ax.axvspan(t0, t1, color="0.85", zorder=0, label="GNSS outage" if i == 0 else None)


def _turn_inset(ax: plt.Axes, scn: Scenario) -> None:
    """Equal-aspect zoom on the first turn inside a GNSS outage (else the first turn)."""
    turns = [iv for iv in scn.meta["intervals"] if iv["kind"] == "turn"]
    if not turns:
        return
    outages = scn.meta["gnss"]["outages"]
    turn = next((iv for iv in turns for t0, t1 in outages if t0 < iv["t1"] and iv["t0"] < t1),
                turns[0])
    tr, gn = scn.truth, scn.gnss
    m = (tr.t >= turn["t0"] - 4) & (tr.t <= turn["t1"] + 4)
    mg = (gn.t >= turn["t0"] - 4) & (gn.t <= turn["t1"] + 4)
    ins = ax.inset_axes((0.62, 0.35, 0.36, 0.6))
    ins.plot(gn.x[mg], gn.y[mg], ".", ms=3, color="tab:orange")
    ins.plot(tr.x[m], tr.y[m], "-", lw=1, color="k")
    ins.set_aspect("equal", adjustable="datalim")
    ins.set_title(f"turn {turn['t0']:.0f}-{turn['t1']:.0f} s", fontsize=8)
    ins.tick_params(labelsize=7)


def plot_scenario(scn: Scenario, out_dir: Path) -> Path:
    """Draw the six-panel overview figure for one scenario and save it as PNG."""
    tr, gn, imu, od = scn.truth, scn.gnss, scn.imu, scn.odom
    fig, axs = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(f"{scn.name} (synthetic) - {scn.meta.get('description', '').strip()}",
                 fontsize=10, wrap=True)

    ax = axs[0, 0]
    ax.plot(gn.x, gn.y, ".", ms=2, color="tab:orange", label="GNSS (10 Hz)")
    ax.plot(tr.x, tr.y, "-", lw=1, color="k", label="truth")
    ax.set(xlabel="east x [m]", ylabel="north y [m]", title="Trajectory (inset: one headland turn)")
    ax.set_xlim(tr.x.min() - 5, tr.x.max() + 60)
    ax.legend(loc="lower right")
    _turn_inset(ax, scn)

    ax = axs[0, 1]
    tx = np.interp(gn.t, tr.t, tr.x)
    ty = np.interp(gn.t, tr.t, tr.y)
    ax.plot(gn.t, np.hypot(gn.x - tx, gn.y - ty), ".", ms=2, color="tab:orange")
    _shade_outages(ax, scn)
    ax.set(xlabel="time [s]", ylabel="horizontal error [m]", title="GNSS position error vs truth")

    ax = axs[1, 0]
    ax.plot(od.t, od.speed, lw=0.5, color="tab:blue", label="odometry (50 Hz)")
    ax.plot(tr.t, tr.speed, lw=1, color="k", label="truth")
    ax.set(xlabel="time [s]", ylabel="speed [m/s]", title="Speed: truth vs wheel odometry")
    ax.legend()

    ax = axs[1, 1]
    ax.plot(imu.t, imu.yaw_rate, lw=0.5, color="tab:green", label="gyro (100 Hz)")
    ax.plot(tr.t, tr.yaw_rate, lw=1, color="k", label="truth")
    ax.set(xlabel="time [s]", ylabel="yaw rate [rad/s]", title="Yaw rate: truth vs gyro")
    ax.legend()

    ax = axs[2, 0]
    ax.plot(imu.t, imu.accel_long, lw=0.3, color="tab:red", label="IMU raw (100 Hz)")
    ax.plot(tr.t, tr.accel_long, lw=1, color="k", label="truth")
    ax.set(xlabel="time [s]", ylabel="accel long [m/s²]", title="Longitudinal acceleration")
    ax.legend()

    ax = axs[2, 1]
    t0 = float(scn.meta["intervals"][0]["t1"]) - 5.0     # braking ramp before the first turn
    m = (imu.t >= t0) & (imu.t < t0 + 1.0)
    ax.plot(imu.t[m], imu.accel_long[m], ".-", lw=0.8, ms=3, color="tab:red", label="IMU raw")
    mt = (tr.t >= t0) & (tr.t < t0 + 1.0)
    ax.plot(tr.t[mt], tr.accel_long[mt], lw=1.5, color="k", label="truth")
    ax.set(xlabel="time [s]", ylabel="accel long [m/s²]",
           title=f"1 s zoom: {scn.meta['imu']['vib_hz']:g} Hz engine vibration on accel")
    ax.legend()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scn.name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("names", nargs="+", help="scenario names in --generated")
    p.add_argument("--generated", default="data/generated", help="generated scenario root")
    p.add_argument("--out", default="out/plots", help="output directory for PNGs")
    a = p.parse_args(argv)
    for name in a.names:
        print(plot_scenario(load_scenario(Path(a.generated) / name), Path(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
