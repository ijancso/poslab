"""Scenario x algorithm tests: every algorithm on every scenario against thresholds.yaml.

Run all:          pytest -m scenario
Run smoke subset: pytest -m smoke
Generated data is cached in data/generated/ and regenerated when a scenario YAML changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from poslab.fusion.registry import ALGORITHMS
from poslab.metrics import compute_metrics
from poslab.runner import ensure_generated, run_one
from poslab.sim.scenario import Scenario
from poslab.thresholds import check_limits, format_checks, load_thresholds

REPO = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO / "data" / "scenarios"
GENERATED = REPO / "data" / "generated"
THRESHOLDS = load_thresholds(REPO / "thresholds.yaml")
SCENARIOS = sorted(f.stem for f in SCENARIO_DIR.glob("*.yaml"))


def _is_smoke(name: str) -> bool:
    return bool(yaml.safe_load((SCENARIO_DIR / f"{name}.yaml").read_text()).get("smoke"))


def _cases() -> list[Any]:
    """(scenario, algorithm) pairs with scenario/smoke markers and strict xfail."""
    cases = []
    for scn in SCENARIOS:
        for algo in sorted(ALGORITHMS):
            marks = [pytest.mark.scenario]
            if _is_smoke(scn):
                marks.append(pytest.mark.smoke)
            reason = THRESHOLDS.get(scn, {}).get("known_failures", {}).get(algo)
            if reason:
                marks.append(pytest.mark.xfail(reason=reason.strip(), strict=True))
            cases.append(pytest.param(scn, algo, marks=marks, id=f"{scn}-{algo}"))
    return cases


_scenario_cache: dict[str, Scenario] = {}
_metric_cache: dict[tuple[str, str], dict[str, Any]] = {}


def scenario(name: str) -> Scenario:
    """Generated scenario (cached for the test session)."""
    if name not in _scenario_cache:
        _scenario_cache[name] = ensure_generated(SCENARIO_DIR / f"{name}.yaml", GENERATED)
    return _scenario_cache[name]


def metrics(name: str, algo: str) -> dict[str, Any]:
    """Metrics of ``algo`` on scenario ``name`` (cached for the test session)."""
    if (name, algo) not in _metric_cache:
        _metric_cache[(name, algo)] = run_one(scenario(name), algo)[1]
    return _metric_cache[(name, algo)]


@pytest.mark.parametrize("name, algo", _cases())
def test_metrics_within_thresholds(name: str, algo: str) -> None:
    checks = check_limits(metrics(name, algo), THRESHOLDS[name]["limits"])
    failed = [c for c in checks if not c.passed]
    assert not failed, format_checks(checks, f"{algo} on {name}: thresholds.yaml violated")


def test_every_scenario_has_thresholds_with_known_metric_names() -> None:
    assert set(SCENARIOS) == set(THRESHOLDS), "scenarios and thresholds.yaml disagree"
    sample = metrics("open_sky_rtk", "ekf") | {"outage_end_err_max_m": 0.0}
    for name, entry in THRESHOLDS.items():
        assert entry["limits"], f"{name}: no limits"
        assert set(entry["limits"]) <= set(sample), f"{name}: unknown metric in limits"
        assert set(entry["known_failures"]) <= set(ALGORITHMS), f"{name}: unknown algorithm"


@pytest.mark.scenario
def test_accel_lowpass_is_what_keeps_vibration_out_of_the_speed() -> None:
    """Same scenario, EKF without the low-pass filter: the speed requirement fails."""
    scn = scenario("heavy_vibration")
    limit = THRESHOLDS["heavy_vibration"]["limits"]["v_rms_mps"]
    _, m_filtered = run_one(scn, "ekf")
    _, m_raw = run_one(scn, "ekf", {"accel_lowpass_hz": None})
    assert m_filtered["v_rms_mps"] <= limit < m_raw["v_rms_mps"]


@pytest.mark.scenario
def test_ekf_beats_gnss_kf_through_the_turn_outage() -> None:
    """The reason the EKF exists: at least 50x less drift through a 30 s turn outage."""
    ekf = metrics("outage_30s_turn", "ekf")["outage_end_err_max_m"]
    kf = metrics("outage_30s_turn", "gnss_kf")["outage_end_err_max_m"]
    assert ekf * 50 < kf


@pytest.mark.scenario
def test_ekf_run_is_deterministic() -> None:
    """Running the same algorithm twice gives identical estimates and metrics."""
    from poslab.fusion.registry import get_algorithm
    from poslab.runner import sensor_data

    scn = scenario("outage_10s_row")
    a = get_algorithm("ekf").run(sensor_data(scn))
    b = get_algorithm("ekf").run(sensor_data(scn))
    assert a.equals(b)
    m = compute_metrics(a, scn.truth, scn.meta["gnss"]["outages"])
    assert m.items() <= metrics("outage_10s_row", "ekf").items()
