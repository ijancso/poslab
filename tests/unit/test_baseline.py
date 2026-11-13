"""Baseline comparison and promotion (poslab.baseline)."""
from __future__ import annotations

import json

import pytest

from poslab.baseline import (
    RegressionConfig,
    compare,
    format_deltas,
    load_regression_config,
    load_results,
    promote,
)

BASE = {"s1": {"ekf": {"h_rms_m": 0.100, "xt_max_m": 0.200, "v_rms_mps": 0.010,
                       "n_samples": 100, "algo_version": "1.0"}}}


def cur(**changes: float) -> dict:
    entry = {**BASE["s1"]["ekf"], **changes}
    return {"s1": {"ekf": entry}}


def statuses(current: dict, cfg: RegressionConfig | None = None) -> dict[str, str]:
    return {d.metric: d.status for d in compare(current, BASE, cfg)}


def test_identical_is_same_and_only_error_metrics_are_compared():
    assert statuses(cur()) == {"h_rms_m": "same", "xt_max_m": "same", "v_rms_mps": "same"}


def test_regression_needs_relative_and_absolute_excess():
    s = statuses(cur(h_rms_m=0.104, xt_max_m=0.220, v_rms_mps=0.0106))
    assert s["h_rms_m"] == "worse"          # +4 %: within the 5 % tolerance
    assert s["xt_max_m"] == "regression"    # +10 % and +2 cm
    assert s["v_rms_mps"] == "regression"   # +6 % and +0.0006 > default floor 0.0005
    cfg = RegressionConfig(min_abs_delta=0.001)
    assert statuses(cur(v_rms_mps=0.0106), cfg)["v_rms_mps"] == "worse"       # below floor
    assert statuses(cur(v_rms_mps=0.0125), cfg)["v_rms_mps"] == "regression"


def test_better_new_missing_and_per_metric_tolerance():
    c = cur(h_rms_m=0.05, at_rms_m=0.3)
    del c["s1"]["ekf"]["xt_max_m"]
    s = statuses(c)
    assert s["h_rms_m"] == "better" and s["at_rms_m"] == "new" and s["xt_max_m"] == "missing"
    cfg = RegressionConfig(tolerances={"xt_max_m": 0.20})
    assert statuses(cur(xt_max_m=0.230), cfg)["xt_max_m"] == "worse"   # +15 % < 20 %


def test_new_algorithm_and_baseline_only_entries():
    deltas = compare({"s1": {"new_algo": {"h_rms_m": 1.0}}, "s9": {"ekf": {"h_rms_m": 1.0}}},
                     BASE)
    assert {d.status for d in deltas} == {"new"} and not any(d.failed for d in deltas)


def test_relative_change_and_table():
    deltas = compare(cur(xt_max_m=0.3), BASE)
    d = next(d for d in deltas if d.metric == "xt_max_m")
    assert d.rel == pytest.approx(0.5) and d.failed
    table = format_deltas(deltas)
    assert "REGRESSION (> 5%)" in table and "+50.0%" in table and "h_rms_m" not in table
    assert format_deltas(compare(cur(), BASE)) == "  (no changes)"
    assert "MISSING" in format_deltas(compare({"s1": {"ekf": {"h_rms_m": 0.1}}}, BASE))


def test_promote_merges_and_load_results(tmp_path):
    path = tmp_path / "baseline" / "metrics.json"
    assert load_results(path) == {}
    promote(BASE, path)
    promote({"s2": {"gnss_kf": {"h_rms_m": 1.0}}, "s1": {"ekf": {"h_rms_m": 0.05}}}, path)
    data = json.loads(path.read_text())
    assert data["s2"]["gnss_kf"]["h_rms_m"] == 1.0
    assert data["s1"]["ekf"] == {"h_rms_m": 0.05}             # entry replaced as a whole


def test_regression_config_from_thresholds(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("regression:\n  default_tolerance: 0.1\n  tolerances: {h_max_m: 0.2}\n")
    cfg = load_regression_config(f)
    assert cfg.tolerance("h_rms_m") == 0.1 and cfg.tolerance("h_max_m") == 0.2
    assert load_regression_config(tmp_path / "none.yaml").default_tolerance == 0.05
    f.write_text("regression:\n  tolerance: 0.1\n")
    with pytest.raises(KeyError):
        load_regression_config(f)
