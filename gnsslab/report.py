"""Write results as Markdown (for humans) and JSON (for later comparison)."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _fmt(v, nd=2):
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _json(obj, path):
    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    Path(path).write_text(json.dumps(clean(obj), indent=2))


def static_table(results) -> str:
    headers = ["Session", "Fixes", "Min", "CEP50 [m]", "CEP95 [m]", "H-RMS [m]", "Max [m]",
               "V-std [m]", "Reported acc. [m]", "Inside reported acc.", "Error corr. time [s]"]
    rows = [[r.label, r.count, r.duration_s / 60, r.cep50, r.cep95, r.h_rms, r.h_max,
             r.v_std, r.acc_median,
             "–" if math.isnan(r.within_acc) else f"{r.within_acc:.0%}", r.corr_time_s]
            for r in results]
    return _table(headers, rows)


def write_static_report(results, out_dir, figures) -> Path:
    out = Path(out_dir)
    kind = results[0].ref_kind
    note = ("Errors are measured against the **mean position** of each session, so they show "
            "precision (scatter), not absolute accuracy. Use `--ref lat,lon` with a surveyed point "
            "to measure accuracy." if kind == "mean" else
            "Errors are measured against the given **reference point** (absolute accuracy).")
    bias = ""
    if kind == "reference":
        bias = "\n\nBias (mean offset from reference):\n\n" + _table(
            ["Session", "East [m]", "North [m]"], [[r.label, r.mean_e, r.mean_n] for r in results])
    md = [
        "# Static GNSS test", "", note, "", static_table(results), bias, "",
        "*Inside reported acc.*: Android reports a 68 % radius, so a well-calibrated phone "
        "is close to 68 %. Much lower means the phone is over-confident.", "",
        "*Error corr. time*: how long errors stay correlated. Fixes closer together than this "
        "are not independent samples.", "",
    ]
    md += [f"![{p.stem}]({p.name})" for p in figures]
    path = out / "report.md"
    path.write_text("\n".join(md) + "\n")
    _json([r.summary() for r in results], out / "metrics.json")
    return path


def drive_table(rows) -> str:
    headers = ["Outage [s]", "Outages", "KF drift median [m]", "KF drift p95 [m]",
               "KF drift max [m]", "Hold drift median [m]", "KF worse than hold",
               "KF predicted drms [m]", "Inside KF 95% ellipse"]
    body = []
    for r in rows:
        d = f"{r['duration_s']:g}"
        if r["outages"] == 0:
            body.append([d, 0] + ["–"] * 7)
            continue
        body.append([d, r["outages"], r["kf_end_median_m"], r["kf_end_p95_m"],
                     r["kf_max_m"], r["hold_end_median_m"], f"{r['kf_worse_than_hold']:.0%}",
                     r["kf_drms_end_median_m"], f"{r['inside_95_ellipse']:.0%}"])
    return _table(headers, body)


def write_drive_report(info, rows, out_dir, figures) -> Path:
    out = Path(out_dir)
    md = [
        "# Drive test: GNSS outages", "",
        f"Track: {info['fixes']} fixes, {info['duration_s'] / 60:.1f} min, "
        f"{info['distance_m'] / 1000:.2f} km, mean speed {info['mean_speed_kmh']:.0f} km/h. "
        f"Filter q = {info['q']}, fixes rejected as outliers: {info['rejected']}.", "",
        "Drift = distance between the filter running without GNSS and the same filter with "
        "GNSS, measured at the end of each outage.", "",
        drive_table(rows), "",
        "*Inside KF 95% ellipse*: share of outages where the real drift stayed inside the "
        "filter's own 95 % error ellipse. An honest filter scores ~95 %; much lower means it "
        "is over-confident (process noise q too small for this driving).", "",
    ]
    md += [f"![{p.stem}]({p.name})" for p in figures]
    path = out / "report.md"
    path.write_text("\n".join(md) + "\n")
    _json({"track": info, "outages": rows}, out / "metrics.json")
    return path
