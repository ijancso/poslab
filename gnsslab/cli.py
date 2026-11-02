"""Command line: gnsslab static | drive | simulate"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _parse_input(arg: str):
    """'LABEL=path' or just 'path'."""
    if not os.path.exists(arg) and "=" in arg:
        label, path = arg.split("=", 1)
        return label, path
    return Path(arg).stem, arg


def cmd_static(a) -> int:
    from .io import load_track
    from .plots import static_plots
    from .report import static_table, write_static_report
    from .static import analyze_static

    ref = tuple(float(v) for v in a.ref.split(",")) if a.ref else None
    results = []
    for arg in a.files:
        label, path = _parse_input(arg)
        results.append(analyze_static(load_track(path, a.provider), label, ref))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = write_static_report(results, out, static_plots(results, out))
    print(static_table(results))
    print(f"\nReport: {report}")
    return 0


def cmd_drive(a) -> int:
    from .geo import geodetic_to_enu
    from .io import load_track
    from .kalman import run_filter, sigma_from_accuracy
    from .outage import outage_study, summarize
    from .plots import drive_plots
    from .report import drive_table, write_drive_report

    tr = load_track(a.file, a.provider)
    e, n, _ = geodetic_to_enu(tr.lat, tr.lon, np.zeros(len(tr)), tr.lat[0], tr.lon[0], 0.0)
    run = run_filter(tr.t, e, n, sigma_from_accuracy(tr.acc), q=a.q)

    durations = [float(d) for d in a.durations.split(",")]
    study = outage_study(run, durations, warmup=a.warmup, gap=a.gap, min_speed=a.min_speed)
    rows = summarize(study)

    step = np.hypot(np.diff(run.x[:, 0]), np.diff(run.x[:, 1]))
    info = {
        "file": str(a.file), "fixes": len(tr), "duration_s": tr.duration,
        "distance_m": float(step.sum()), "mean_speed_kmh": float(step.sum() / tr.duration * 3.6),
        "q": a.q, "rejected": int(run.rejected.sum()),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = write_drive_report(info, rows, out, drive_plots(run, study, out))
    print(drive_table(rows))
    print(f"\nReport: {report}")
    return 0


def cmd_simulate(a) -> int:
    from .simulate import write_samples

    for p in write_samples(a.out, a.seed):
        print(p)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gnsslab", description="Phone GNSS accuracy and outage-drift lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static", help="accuracy statistics of stationary recordings")
    s.add_argument("files", nargs="+", help="log files, optionally as LABEL=path")
    s.add_argument("--ref", help="known position 'lat,lon[,height]' for absolute accuracy")
    s.add_argument("--provider", default="gps", help="provider to keep (default: gps)")
    s.add_argument("--out", default="results/static")
    s.set_defaults(func=cmd_static)

    d = sub.add_parser("drive", help="simulate GNSS outages on a drive and measure drift")
    d.add_argument("file")
    d.add_argument("--durations", default="5,10,20,30", help="outage lengths in seconds")
    d.add_argument("--q", type=float, default=3.0, help="process noise [m^2/s^3]")
    d.add_argument("--warmup", type=float, default=30.0, help="seconds before the first outage")
    d.add_argument("--gap", type=float, default=10.0, help="seconds between outages")
    d.add_argument("--min-speed", type=float, default=3.0, help="only start outages above this speed [m/s]")
    d.add_argument("--provider", default="gps")
    d.add_argument("--out", default="results/drive")
    d.set_defaults(func=cmd_drive)

    m = sub.add_parser("simulate", help="write synthetic demo logs")
    m.add_argument("--out", default="data/sample")
    m.add_argument("--seed", type=int, default=1)
    m.set_defaults(func=cmd_simulate)

    a = p.parse_args(argv)
    return a.func(a)
