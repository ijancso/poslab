"""Static HTML performance report: current metrics vs requirements and baseline, with plots.

One self-contained file (``out/report/index.html``): PNG figures are embedded as base64,
no web framework, no external assets. Written for a developer who did not write the
algorithm: every figure has a short caption, every number a unit.

Inputs (all produced by ``python -m poslab run``): out/metrics.json, out/estimates/,
data/generated/<scenario>/, plus baseline/metrics.json and thresholds.yaml.
"""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from poslab.baseline import (  # noqa: E402
    Delta,
    Results,
    compare,
    load_regression_config,
    load_results,
)
from poslab.dsp import dominant_frequency, lowpass, psd  # noqa: E402
from poslab.fusion.ekf import DEFAULTS as EKF_DEFAULTS  # noqa: E402
from poslab.io.loaders import load_scenario  # noqa: E402
from poslab.metrics import error_components  # noqa: E402
from poslab.sim.scenario import Scenario  # noqa: E402
from poslab.thresholds import check_limits, load_thresholds  # noqa: E402

# Reference data-viz palette: categorical slots in fixed order, ink for truth/text,
# status colours reserved for pass/fail (always with an icon + label).
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SURFACE = "#fcfcfb"
ALGO_COLORS = {"gnss_kf": "#2a78d6", "ekf": "#eb6834"}     # slots 1, 2
EXTRA_COLORS = ["#6250d6", "#e34948"]                       # slots 7, 8 for new algorithms
GNSS_COLOR = "#1baf7a"                                      # slot 3
GOOD, WARN, BAD = "#0ca30c", "#b07800", "#d03b3b"           # text-safe status inks

KEY_METRICS = [("h_cep95_m", "CEP95", "m"), ("xt_max_m", "cross-track max", "m"),
               ("outage_end_err_max_m", "outage end error", "m"),
               ("v_rms_mps", "speed RMS", "m/s")]


def algo_color(algo: str, order: list[str]) -> str:
    """Colour follows the algorithm, never its rank in a particular run."""
    if algo in ALGO_COLORS:
        return ALGO_COLORS[algo]
    extra = [a for a in order if a not in ALGO_COLORS]
    return EXTRA_COLORS[extra.index(algo) % len(EXTRA_COLORS)]


# ------------------------------------------------------------------ figures

def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _shade_outages(ax: plt.Axes, outages: list[list[float]]) -> None:
    for i, (t0, t1) in enumerate(outages):
        ax.axvspan(t0, t1, color="#d9d8d3", zorder=0, lw=0,
                   label="GNSS outage" if i == 0 else None)


def _focus_window(scn: Scenario) -> tuple[float, float, str]:
    """Time window worth zooming into: the first outage, else the first headland turn."""
    outages = scn.meta["gnss"]["outages"]
    if outages:
        t0, t1 = outages[0]
        return t0 - 8.0, t1 + 8.0, f"GNSS outage {t0:g}-{t1:g} s (±8 s)"
    turns = [iv for iv in scn.meta["intervals"] if iv["kind"] == "turn"]
    if turns:
        return turns[0]["t0"] - 8.0, turns[0]["t1"] + 8.0, "first headland turn (±8 s)"
    return scn.truth.t.iloc[0], scn.truth.t.iloc[-1], "whole run"


def fig_trajectory(scn: Scenario, est: dict[str, pd.DataFrame]) -> tuple[str, str]:
    t0, t1, what = _focus_window(scn)
    tr = scn.truth[(scn.truth.t >= t0) & (scn.truth.t <= t1)]
    gn = scn.gnss[(scn.gnss.t >= t0) & (scn.gnss.t <= t1)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _style(ax)
    ax.plot(gn.x, gn.y, "o", ms=2.5, color=GNSS_COLOR, label="GNSS fixes", zorder=2)
    ax.plot(tr.x, tr.y, "-", lw=2, color=INK, label="truth", zorder=3)
    order = list(est)
    for algo, e in est.items():
        e = e[(e.t >= t0) & (e.t <= t1)]
        ax.plot(e.x, e.y, "-", lw=1.5, color=algo_color(algo, order), label=algo, zorder=4)
    ax.set(xlabel="east x [m]", ylabel="north y [m]")
    # frame on the true path; a long thin window (straight row) gets a stretched narrow axis
    xr, yr = (tr.x.min(), tr.x.max()), (tr.y.min(), tr.y.max())
    span_x, span_y = xr[1] - xr[0], yr[1] - yr[0]
    stretched = max(span_x, span_y) > 3 * max(min(span_x, span_y), 1.0)
    if stretched:
        narrow_pad, long_pad = 1.5, 3.0
        px, py = (narrow_pad, long_pad) if span_x < span_y else (long_pad, narrow_pad)
        ax.set_xlim(xr[0] - px, xr[1] + px)
        ax.set_ylim(yr[0] - py, yr[1] + py)
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, frameon=False, loc="best")
    note = (" Axes not to scale: the narrow axis is stretched (±1.5 m around the true path) "
            "to make lateral deviations visible; estimates outside it are cut off."
            if stretched else "")
    return _png(fig), (f"Trajectory, zoomed on the {what}. Black: true path; dots: GNSS fixes "
                       f"(none during an outage); coloured lines: algorithm estimates.{note}")


def fig_error_vs_time(scn: Scenario, est: dict[str, pd.DataFrame]) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    _style(ax)
    _shade_outages(ax, scn.meta["gnss"]["outages"])
    order = list(est)
    for algo, e in est.items():
        err = error_components(e, scn.truth)
        ax.plot(err.t, np.maximum(err.horizontal, 1e-4), lw=1.2,
                color=algo_color(algo, order), label=algo)
    ax.set_yscale("log")
    ax.set(xlabel="time [s]", ylabel="horizontal error [m] (log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
    return _png(fig), ("Horizontal position error vs time (log scale: 1 cm and 10 m on one "
                       "axis). Grey bands: GNSS outages - the error growing inside them is "
                       "the drift.")


def fig_crosstrack_hist(scn: Scenario, est: dict[str, pd.DataFrame]) -> tuple[str, str]:
    order = list(est)
    fig, axs = plt.subplots(1, len(est), figsize=(4.2 * len(est), 2.9), squeeze=False)
    for ax, (algo, e) in zip(axs[0], est.items(), strict=True):
        _style(ax)
        err = error_components(e, scn.truth)
        xt = 100 * err.cross[err.t >= err.t.iloc[0] + 5.0]          # cm, after warm-up
        lim = max(float(np.percentile(np.abs(xt), 99.5)), 1.0)
        ax.hist(np.clip(xt, -lim, lim), bins=60, range=(-lim, lim),
                color=algo_color(algo, order), edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(algo, fontsize=9, color=INK)
        ax.set(xlabel="cross-track error [cm]", ylabel="samples")
    return _png(fig), ("Cross-track error distribution per algorithm (perpendicular to the "
                       "true heading; positive = left of the path). Each panel has its own "
                       "scale; the outermost 0.5 % are clipped into the edge bins.")


def fig_accel_spectrum(scn: Scenario) -> tuple[str, str]:
    fs = 1.0 / float(np.median(np.diff(scn.imu.t)))
    raw = scn.imu.accel_long.to_numpy()
    cutoff = EKF_DEFAULTS["accel_lowpass_hz"]
    filt = lowpass(raw, fs, cutoff, EKF_DEFAULTS["accel_lowpass_order"], causal=True)
    f, p_raw = psd(raw, fs)
    _, p_filt = psd(filt, fs)
    f_dom = dominant_frequency(raw, fs)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _style(ax)
    ax.semilogy(f, p_raw, lw=1.5, color=INK_MUTED, label="raw accelerometer")
    ax.semilogy(f, p_filt, lw=1.5, color=INK, label=f"after {cutoff:g} Hz causal low-pass")
    ax.axvline(cutoff, color=INK_2, lw=1, ls="--")
    ax.annotate(f"vibration {f_dom:.1f} Hz", (f_dom, p_raw.max()), xytext=(6, -4),
                textcoords="offset points", fontsize=8, color=INK_2, va="top")
    ax.set(xlabel="frequency [Hz]", ylabel="PSD [(m/s²)²/Hz] (log)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return _png(fig), (f"Longitudinal acceleration spectrum (Welch PSD) before and after the "
                       f"EKF's {cutoff:g} Hz Butterworth low-pass (dashed line). The "
                       f"vibration peak at {f_dom:.1f} Hz is suppressed; vehicle motion lives "
                       "below ~1 Hz.")


def fig_algo_comparison(current: Results, algos: list[str]) -> tuple[str, str]:
    metrics = [m for m in KEY_METRICS[:3]]
    scns = sorted(current)
    fig, axs = plt.subplots(1, len(metrics), figsize=(11, 0.42 * len(scns) + 1.4), sharey=True)
    y = np.arange(len(scns))
    for ax, (key, label, unit) in zip(axs, metrics, strict=True):
        _style(ax)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.grid(True, axis="y", color=GRID, lw=0.4, ls=":")
        vals = np.array([[current[s].get(a, {}).get(key, np.nan) for a in algos] for s in scns])
        # thin connector shows the gap between the algorithms on each scenario
        has = ~np.all(np.isnan(vals), axis=1)
        ax.hlines(y[has], np.nanmin(vals[has], axis=1), np.nanmax(vals[has], axis=1),
                  color=INK_MUTED, lw=1)
        for j, algo in enumerate(algos):
            ax.plot(vals[:, j], y, "o", ms=8, color=algo_color(algo, algos), label=algo,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"{label} [{unit}]", fontsize=9, color=INK)
        ax.set_xlabel("log scale", fontsize=8)
    axs[0].set_yticks(y, scns, fontsize=8)
    axs[0].set_ylim(len(scns) - 0.5, -0.5)
    axs[0].legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3)
    return _png(fig), ("Algorithms side by side on every scenario (log scale - values span "
                       "centimetres to tens of metres). Further left is better. Scenarios "
                       "without an outage have no outage-error dots.")


# ------------------------------------------------------------------ status logic

@dataclass
class EntryStatus:
    """Requirement + regression status of one (scenario, algorithm)."""

    req: str            # pass | fail | known | fixed (known failure that now passes) | none
    req_detail: str
    regressions: int
    deltas: list[Delta]


def entry_status(scn: str, algo: str, metrics: dict[str, Any], thresholds: dict[str, Any],
                 deltas: list[Delta]) -> EntryStatus:
    th = thresholds.get(scn)
    mine = [d for d in deltas if d.scenario == scn and d.algo == algo]
    if not th:
        return EntryStatus("none", "no requirements defined", 0, mine)
    checks = check_limits(metrics, th["limits"])
    failed = [c for c in checks if not c.passed]
    known = th["known_failures"].get(algo)
    if failed:
        detail = ", ".join(f"{c.metric} {c.value if c.value is None else f'{c.value:.3g}'}"
                           f" > {c.limit:g}" for c in failed)
        req = "known" if known else "fail"
        if known:
            detail = f"{detail}. Known: {known.strip()}"
    else:
        req = "fixed" if known else "pass"
        detail = ("passes although listed as a known failure - update thresholds.yaml"
                  if known else "all limits met")
    return EntryStatus(req, detail, sum(d.failed for d in mine), mine)


_REQ_BADGE = {
    "pass": ("✓", "pass", GOOD), "fail": ("✗", "FAIL", BAD),
    "known": ("!", "known failure", WARN), "fixed": ("!", "unexpected pass", WARN),
    "none": ("–", "no limits", INK_MUTED),
}


def _badge(icon: str, label: str, color: str, title: str = "") -> str:
    t = f' title="{html.escape(title)}"' if title else ""
    return (f'<span class="badge" style="color:{color};border-color:{color}"{t}>'
            f'<b>{icon}</b> {html.escape(label)}</span>')


def _delta_cell(d: Delta | None) -> str:
    if d is None or d.rel is None:
        return '<td class="num muted">–</td>'
    pct = 100 * d.rel
    if d.status == "regression":
        return f'<td class="num" style="color:{BAD}"><b>▲ {pct:+.1f}%</b></td>'
    if d.status == "same":
        return '<td class="num muted">0%</td>'
    color = GOOD if d.status == "better" else INK_2
    arrow = "▼" if d.status == "better" else "▲"
    return f'<td class="num" style="color:{color}">{arrow} {pct:+.1f}%</td>'


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


# ------------------------------------------------------------------ page

CSS = f"""
:root {{ color-scheme: light; --surface:{SURFACE}; --ink:{INK}; --ink2:{INK_2};
  --muted:{INK_MUTED}; --grid:{GRID}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 20px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--grid); }}
h3 {{ font-size: 16px; margin: 24px 0 6px; }}
p, li {{ color: var(--ink2); max-width: 75ch; }}
.note {{ border-left: 3px solid var(--muted); padding: 4px 12px; color: var(--ink2); }}
.tiles {{ display:flex; flex-wrap:wrap; gap:12px; margin: 16px 0; }}
.tile {{ border:1px solid var(--grid); border-radius:8px; padding:10px 14px; min-width:150px; }}
.tile .v {{ font-size: 26px; font-weight: 650; }}
.tile .l {{ font-size: 12px; color: var(--ink2); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
th, td {{ padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: left;
  vertical-align: top; }}
th {{ font-weight: 600; color: var(--ink2); white-space: nowrap; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.muted {{ color: var(--muted); }}
.badge {{ display:inline-block; border:1px solid; border-radius: 10px; padding: 0 8px;
  font-size: 12px; white-space: nowrap; }}
figure {{ margin: 12px 0 20px; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 6px; }}
figcaption {{ font-size: 13px; color: var(--ink2); max-width: 90ch; }}
.grid2 {{ display:grid; gap: 8px 20px;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
details summary {{ cursor: pointer; color: var(--ink2); font-size: 14px; }}
code {{ font-size: 13px; }}
"""


def _figure(png_caption: tuple[str, str], alt: str) -> str:
    png, caption = png_caption
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{png}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def build_report(out_dir: str | Path = "out", generated: str | Path = "data/generated",
                 baseline_path: str | Path = "baseline/metrics.json",
                 thresholds_path: str | Path = "thresholds.yaml") -> Path:
    """Render ``<out_dir>/report/index.html`` and return its path."""
    out_dir = Path(out_dir)
    current = load_results(out_dir / "metrics.json")
    if not current:
        raise FileNotFoundError(f"{out_dir / 'metrics.json'} missing - run "
                                "'python -m poslab run' first")
    baseline = load_results(baseline_path)
    thresholds = load_thresholds(thresholds_path) if Path(thresholds_path).exists() else {}
    deltas = compare(current, baseline, load_regression_config(thresholds_path))
    known = list(ALGO_COLORS)                   # baseline algorithm first, then the others
    algos = sorted({a for s in current.values() for a in s},
                   key=lambda a: (known.index(a) if a in known else len(known), a))

    status = {(s, a): entry_status(s, a, current[s][a], thresholds, deltas)
              for s in sorted(current) for a in algos if a in current[s]}
    n = len(status)
    n_pass = sum(st.req == "pass" for st in status.values())
    n_known = sum(st.req == "known" for st in status.values())
    n_fail = sum(st.req in ("fail", "fixed") for st in status.values())
    n_reg = sum(st.regressions for st in status.values())
    n_better = sum(d.status == "better" for d in deltas)
    versions = {a: next(current[s][a]["algo_version"] for s in current if a in current[s])
                for a in algos}

    parts: list[str] = []
    w = parts.append
    w(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>poslab performance report</title><style>{CSS}</style></head><body><main>")
    w("<h1>poslab performance report</h1>")
    w(f"<p class='muted'>Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · "
      + " · ".join(f"<code>{html.escape(a)}</code> v{html.escape(str(v))}"
                   for a, v in versions.items()) + "</p>")
    w("<p class='note'><b>Synthetic data.</b> All scenarios are simulated (see "
      "<code>data/scenarios/</code>); this is a learning project, not field-test evidence.</p>")

    w("<div class='tiles'>")
    for v, label, color in ((f"{n_pass}/{n}", "runs meet requirements", GOOD if n_fail == 0
                             else INK), (str(n_known), "known failures", WARN if n_known else INK),
                            (str(n_fail), "unexpected failures", BAD if n_fail else INK),
                            (str(n_reg), "regressions vs baseline", BAD if n_reg else INK),
                            (str(n_better), "metrics better than baseline", INK)):
        w(f"<div class='tile'><div class='v' style='color:{color}'>{v}</div>"
          f"<div class='l'>{label}</div></div>")
    w("</div>")
    if not baseline:
        w("<p class='note'>No baseline found - run <code>python -m poslab baseline-update"
          "</code> to store one.</p>")

    # summary table
    w("<h2>Summary</h2><p>Requirements come from <code>thresholds.yaml</code>; the change "
      "column compares with <code>baseline/metrics.json</code> (▼ better, ▲ worse; "
      "<b>bold red</b> = regression beyond tolerance). All metrics: lower is better.</p>")
    w("<div class='scroll'><table><tr><th>scenario</th><th>algorithm</th>"
      "<th>requirements</th><th>vs baseline</th>")
    for _, label, unit in KEY_METRICS:
        w(f"<th class='num'>{label} [{unit}]</th><th class='num'>change</th>")
    w("</tr>")
    for (scn, algo), st in status.items():
        icon, label, color = _REQ_BADGE[st.req]
        reg = (_badge("✗", f"{st.regressions} regression(s)", BAD) if st.regressions
               else _badge("✓", "no regression", GOOD) if st.deltas and
               any(d.baseline is not None for d in st.deltas)
               else _badge("–", "no baseline", INK_MUTED))
        w(f"<tr><td>{scn}</td><td>{algo}</td><td>{_badge(icon, label, color, st.req_detail)}"
          f"</td><td>{reg}</td>")
        by_metric = {d.metric: d for d in st.deltas}
        for key, _, _ in KEY_METRICS:
            w(f"<td class='num'>{_fmt(current[scn][algo].get(key))}</td>"
              f"{_delta_cell(by_metric.get(key))}")
        w("</tr>")
    w("</table></div>")

    problems = [(k, st) for k, st in status.items() if st.req in ("fail", "fixed", "known")]
    if problems:
        w("<h3>Requirement findings</h3><ul>")
        for (scn, algo), st in problems:
            icon, label, color = _REQ_BADGE[st.req]
            w(f"<li>{_badge(icon, label, color)} <b>{algo}</b> on <b>{scn}</b>: "
              f"{html.escape(st.req_detail)}</li>")
        w("</ul>")
    regs = [d for d in deltas if d.failed]
    if regs:
        w("<h3>Regressions</h3><div class='scroll'><table><tr><th>scenario</th>"
          "<th>algorithm</th><th>metric</th><th class='num'>baseline</th>"
          "<th class='num'>current</th><th class='num'>change</th><th class='num'>tolerance"
          "</th></tr>")
        for d in regs:
            w(f"<tr><td>{d.scenario}</td><td>{d.algo}</td><td>{d.metric}</td>"
              f"<td class='num'>{_fmt(d.baseline)}</td><td class='num'>{_fmt(d.current)}</td>"
              f"{_delta_cell(d)}<td class='num'>{100 * d.tolerance:.0f}%</td></tr>")
        w("</table></div>")

    if len(algos) > 1:
        w("<h2>Algorithm comparison</h2>")
        w(_figure(fig_algo_comparison(current, algos), "bar charts comparing algorithms"))

    # per scenario
    w("<h2>Scenarios</h2>")
    for scn_name in sorted(current):
        scn = load_scenario(Path(generated) / scn_name)
        est = {a: pd.read_csv(out_dir / "estimates" / scn_name / f"{a}.csv")
               for a in algos if a in current[scn_name]}
        w(f"<h3 id='{scn_name}'>{scn_name}</h3>")
        w(f"<p>{html.escape(str(scn.meta.get('description', '')).strip())}</p>")
        w("<div class='grid2'>")
        w(_figure(fig_trajectory(scn, est), f"trajectory zoom for {scn_name}"))
        w(_figure(fig_accel_spectrum(scn), f"accelerometer spectrum for {scn_name}"))
        w("</div>")
        w(_figure(fig_error_vs_time(scn, est), f"error vs time for {scn_name}"))
        w(_figure(fig_crosstrack_hist(scn, est), f"cross-track histogram for {scn_name}"))
        w("<details><summary>All metrics for this scenario</summary><div class='scroll'>"
          "<table><tr><th>metric</th>")
        for a in est:
            w(f"<th class='num'>{a}</th><th class='num'>baseline</th><th class='num'>change"
              "</th>")
        limits = thresholds.get(scn_name, {}).get("limits", {})
        w("<th class='num'>limit</th></tr>")
        keys = sorted({k for a in est for k, v in current[scn_name][a].items()
                       if isinstance(v, (int, float)) and k.endswith(("_m", "_mps"))})
        for k in keys:
            w(f"<tr><td>{k}</td>")
            for a in est:
                d = next((d for d in status[(scn_name, a)].deltas if d.metric == k), None)
                w(f"<td class='num'>{_fmt(current[scn_name][a].get(k))}</td>"
                  f"<td class='num muted'>{_fmt(d.baseline if d else None)}</td>"
                  f"{_delta_cell(d)}")
            w(f"<td class='num'>{_fmt(limits.get(k))}</td></tr>")
        w("</table></div></details>")

    w("<h2>How to read this</h2><ul>"
      "<li><b>CEP95</b>: radius around the true position containing 95 % of the estimates.</li>"
      "<li><b>Cross-track</b>: error perpendicular to the direction of travel - decides "
      "whether the implement stays on the crop row.</li>"
      "<li><b>Outage end error</b>: position error at the end of a GNSS outage, i.e. how far "
      "the estimate drifted while coasting on IMU and odometry.</li>"
      "<li><b>Speed RMS</b>: speed estimate error; accelerometer vibration shows up here.</li>"
      "<li>Metrics exclude the first 5 s (filter start-up).</li></ul>")
    w("</main></body></html>")

    path = out_dir / "report" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path
