"""thresholds.yaml loading and limit checks (poslab.thresholds)."""
from __future__ import annotations

import pytest

from poslab.thresholds import check_limits, format_checks, load_thresholds


def test_load_and_check(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("scenarios:\n  a:\n    limits: {h_rms_m: 0.1, xt_max_m: 0.2}\n"
                 "    known_failures: {gnss_kf: too simple}\n  b:\n    limits: {}\n")
    th = load_thresholds(f)
    assert th["a"]["known_failures"] == {"gnss_kf": "too simple"}
    assert th["b"] == {"limits": {}, "known_failures": {}}
    checks = check_limits({"h_rms_m": 0.05, "xt_max_m": 0.3}, th["a"]["limits"])
    assert [(c.metric, c.passed) for c in checks] == [("h_rms_m", True), ("xt_max_m", False)]
    table = format_checks(checks, "title")
    assert table.splitlines()[0] == "title" and "FAIL" in table and "ok" in table


def test_missing_metric_fails_and_limit_is_inclusive():
    checks = check_limits({"h_rms_m": 0.1}, {"h_rms_m": 0.1, "v_rms_mps": 0.01})
    assert [c.passed for c in checks] == [True, False]
    assert "missing" in format_checks(checks)


def test_unknown_keys_rejected(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("scenarios:\n  a:\n    limit: {h_rms_m: 0.1}\n")
    with pytest.raises(KeyError, match="unknown keys"):
        load_thresholds(f)
