"""Baseline metrics: load, compare (regression check) and promote.

``baseline/metrics.json`` has the same layout as ``out/metrics.json``:
``{scenario: {algorithm: {metric: value, "algo_version": ..., ...}}}``. It is committed, so
every change to it shows up in a pull request diff.

All compared metrics are "lower is better" (errors, drift). A metric *regresses* when

    current > baseline * (1 + tolerance)   and   current - baseline > min_abs_delta

The relative tolerance (default 5 %) comes from the ``regression`` section of
thresholds.yaml, optionally per metric; the absolute floor keeps sub-millimetre changes of
centimetre-level metrics from counting as regressions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BASELINE = Path("baseline/metrics.json")
DEFAULT_TOLERANCE = 0.05
DEFAULT_MIN_ABS_DELTA = 0.0005

Results = dict[str, dict[str, dict[str, Any]]]


@dataclass
class RegressionConfig:
    """Tolerances for the baseline comparison."""

    default_tolerance: float = DEFAULT_TOLERANCE
    min_abs_delta: float = DEFAULT_MIN_ABS_DELTA
    tolerances: dict[str, float] | None = None

    def tolerance(self, metric: str) -> float:
        return (self.tolerances or {}).get(metric, self.default_tolerance)


@dataclass
class Delta:
    """Comparison of one metric of one (scenario, algorithm) against the baseline."""

    scenario: str
    algo: str
    metric: str
    baseline: float | None
    current: float | None
    tolerance: float
    status: str             # same | better | worse | regression | new | missing

    @property
    def rel(self) -> float | None:
        """Relative change (current - baseline) / baseline, None if undefined."""
        if self.baseline is None or self.current is None or self.baseline == 0:
            return None
        return (self.current - self.baseline) / self.baseline

    @property
    def failed(self) -> bool:
        return self.status in ("regression", "missing")


def is_compared(metric: str, value: Any) -> bool:
    """Only numeric error metrics are compared (units _m or _mps), not counts/versions."""
    return metric.endswith(("_m", "_mps")) and isinstance(value, (int, float))


def load_results(path: str | Path) -> Results:
    """Read a metrics JSON file ({} if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_regression_config(thresholds_path: str | Path) -> RegressionConfig:
    """``regression:`` section of thresholds.yaml (defaults if absent)."""
    path = Path(thresholds_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    reg = (data or {}).get("regression") or {}
    unknown = set(reg) - {"default_tolerance", "min_abs_delta", "tolerances"}
    if unknown:
        raise KeyError(f"{path}: unknown regression keys {sorted(unknown)}")
    return RegressionConfig(
        default_tolerance=float(reg.get("default_tolerance", DEFAULT_TOLERANCE)),
        min_abs_delta=float(reg.get("min_abs_delta", DEFAULT_MIN_ABS_DELTA)),
        tolerances={k: float(v) for k, v in (reg.get("tolerances") or {}).items()},
    )


def compare_entry(scenario: str, algo: str, current: dict[str, Any],
                  baseline: dict[str, Any] | None, cfg: RegressionConfig) -> list[Delta]:
    """Compare the metrics of one (scenario, algorithm)."""
    baseline = baseline or {}
    names = sorted({k for k, v in current.items() if is_compared(k, v)}
                   | {k for k, v in baseline.items() if is_compared(k, v)})
    out = []
    for m in names:
        cur, base, tol = current.get(m), baseline.get(m), cfg.tolerance(m)
        if base is None:
            status = "new"
        elif cur is None:
            status = "missing"
        elif cur > base * (1 + tol) and cur - base > cfg.min_abs_delta:
            status = "regression"
        elif abs(cur - base) <= 1e-9 * max(1.0, abs(base)):
            status = "same"
        else:
            status = "worse" if cur > base else "better"
        out.append(Delta(scenario, algo, m, base, cur, tol, status))
    return out


def compare(current: Results, baseline: Results, cfg: RegressionConfig | None = None
            ) -> list[Delta]:
    """Compare every (scenario, algorithm) of ``current``; baseline-only entries are
    ignored (a smoke run covers only part of the baseline)."""
    cfg = cfg or RegressionConfig()
    out: list[Delta] = []
    for scn in sorted(current):
        for algo in sorted(current[scn]):
            out += compare_entry(scn, algo, current[scn][algo],
                                 baseline.get(scn, {}).get(algo), cfg)
    return out


def format_deltas(deltas: list[Delta], only_changed: bool = True) -> str:
    """Readable diff table (used by baseline-update and test failure messages)."""
    rows = [d for d in deltas if not only_changed or d.status != "same"]
    if not rows:
        return "  (no changes)"
    lines = [f"  {'scenario':18s} {'algo':8s} {'metric':22s} {'baseline':>10s} "
             f"{'current':>10s} {'change':>8s}  status"]
    for d in rows:
        b = "-" if d.baseline is None else f"{d.baseline:10.4f}"
        c = "-" if d.current is None else f"{d.current:10.4f}"
        r = "" if d.rel is None else f"{100 * d.rel:+7.1f}%"
        flag = f"REGRESSION (> {100 * d.tolerance:.0f}%)" if d.status == "regression" \
            else d.status.upper() if d.status == "missing" else d.status
        lines.append(f"  {d.scenario:18s} {d.algo:8s} {d.metric:22s} {b:>10s} {c:>10s} "
                     f"{r:>8s}  {flag}")
    return "\n".join(lines)


def promote(current: Results, path: str | Path = DEFAULT_BASELINE) -> Path:
    """Merge ``current`` into the baseline file: entries in ``current`` replace their
    baseline counterparts, other baseline entries are kept."""
    path = Path(path)
    merged = load_results(path)
    for scn, algos in current.items():
        merged.setdefault(scn, {}).update(algos)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
