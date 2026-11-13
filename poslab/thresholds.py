"""Pass/fail limits per scenario (``thresholds.yaml``) and how metrics are checked.

File format::

    scenarios:
      <scenario>:
        limits: {<metric>: <max value>, ...}     # metric must be <= limit
        known_failures: {<algorithm>: <reason>}  # optional, see below

``known_failures`` documents an algorithm that is *expected* not to meet the scenario's
requirements (e.g. the GNSS-only baseline cannot coast through an outage). The scenario
tests mark those cases as strict xfail: they must keep failing, and if one starts to pass
the test run says so, so the file gets updated instead of silently going stale.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_THRESHOLDS = Path("thresholds.yaml")


@dataclass
class LimitCheck:
    """Result of checking one metric against its limit."""

    metric: str
    value: float | None
    limit: float

    @property
    def passed(self) -> bool:
        return self.value is not None and self.value <= self.limit


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS) -> dict[str, dict[str, Any]]:
    """Read thresholds.yaml -> {scenario: {"limits": {...}, "known_failures": {...}}}."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, dict[str, Any]] = {}
    for name, entry in (data.get("scenarios") or {}).items():
        unknown = set(entry) - {"limits", "known_failures"}
        if unknown:
            raise KeyError(f"{path}: scenario '{name}' has unknown keys {sorted(unknown)}")
        out[name] = {"limits": dict(entry.get("limits") or {}),
                     "known_failures": dict(entry.get("known_failures") or {})}
    return out


def check_limits(metrics: dict[str, Any], limits: dict[str, float]) -> list[LimitCheck]:
    """Check every limited metric; a metric missing from ``metrics`` counts as failed."""
    return [LimitCheck(k, metrics.get(k), float(v)) for k, v in sorted(limits.items())]


def format_checks(checks: list[LimitCheck], title: str = "") -> str:
    """Readable table of limit checks (used in test failure messages)."""
    lines = [title] if title else []
    lines.append(f"  {'metric':24s} {'value':>10s} {'limit':>10s}  result")
    for c in checks:
        val = "missing" if c.value is None else f"{c.value:10.4f}"
        lines.append(f"  {c.metric:24s} {val:>10s} {c.limit:10.4f}  "
                     f"{'ok' if c.passed else 'FAIL'}")
    return "\n".join(lines)
