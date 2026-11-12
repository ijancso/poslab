"""Algorithm registry: name -> class. Every class has ``NAME`` and ``VERSION``.

Bump an algorithm's ``VERSION`` whenever its behaviour changes; the version is stored with
the metrics, so a baseline comparison shows which algorithm version produced which numbers.
"""
from __future__ import annotations

from typing import Any

from poslab.fusion.base import Algorithm
from poslab.fusion.ekf import FusionEKF
from poslab.fusion.gnss_kf import GnssKF

ALGORITHMS: dict[str, type[Algorithm]] = {
    GnssKF.NAME: GnssKF,
    FusionEKF.NAME: FusionEKF,
}


def get_algorithm(name: str, config: dict[str, Any] | None = None) -> Algorithm:
    """Instantiate a registered algorithm with optional config overrides."""
    try:
        cls = ALGORITHMS[name]
    except KeyError:
        raise KeyError(f"unknown algorithm '{name}'; available: {sorted(ALGORITHMS)}") from None
    return cls(config)
