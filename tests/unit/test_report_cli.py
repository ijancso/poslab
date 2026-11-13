"""HTML report and the report / baseline-update commands on a tiny scenario."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from poslab.__main__ import main
from poslab.report import algo_color, build_report


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict[str, Path]:
    """Two tiny scenarios (one with an outage, one rowless), run with both algorithms."""
    root = tmp_path_factory.mktemp("ws")
    scn = root / "scenarios"
    scn.mkdir()
    (scn / "tiny.yaml").write_text(
        "description: tiny test field\nseed: 1\nfield: {rows: 2, row_length: 30.0}\n"
        "gnss: {outages: [[16.0, 19.0]]}\n")
    (scn / "single_row.yaml").write_text("seed: 2\nfield: {rows: 1, row_length: 40.0}\n")
    th = root / "thresholds.yaml"
    th.write_text("scenarios:\n  tiny:\n    limits: {xt_max_m: 0.5, outage_end_err_max_m: 0.3}\n"
                  "    known_failures: {gnss_kf: drifts}\n")
    ws = {"root": root, "scenarios": scn, "gen": root / "gen", "out": root / "out",
          "baseline": root / "baseline.json", "thresholds": th}
    assert main(["run", "--scenarios", str(scn), "--generated", str(ws["gen"]),
                 "--out", str(ws["out"])]) == 0
    return ws


def _report(ws: dict[str, Path]) -> str:
    assert main(["report", "--out", str(ws["out"]), "--generated", str(ws["gen"]),
                 "--baseline", str(ws["baseline"]), "--thresholds", str(ws["thresholds"])]) == 0
    return (ws["out"] / "report" / "index.html").read_text()


def test_report_without_baseline(workspace):
    page = _report(workspace)
    assert page.startswith("<!doctype html>") and "</html>" in page
    assert page.count("data:image/png;base64,") == 2 * 4 + 1      # 4 per scenario + comparison
    assert "No baseline found" in page and "Synthetic data" in page
    assert "tiny test field" in page and "no limits" in page          # single_row has none


def test_baseline_update_then_report_shows_regression(workspace, capsys, monkeypatch):
    ws = workspace
    args = ["baseline-update", "--out", str(ws["out"]), "--baseline", str(ws["baseline"]),
            "--thresholds", str(ws["thresholds"])]
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(args) == 1 and not ws["baseline"].exists()            # declined
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert main(args) == 0 and ws["baseline"].exists()
    assert main(args + ["--yes"]) == 0
    assert "already up to date" in capsys.readouterr().out

    # make the stored baseline look 50 % better than the current ekf -> regression
    base = json.loads(ws["baseline"].read_text())
    base["tiny"]["ekf"]["xt_max_m"] *= 0.5
    ws["baseline"].write_text(json.dumps(base))
    page = _report(ws)
    assert "1 regression(s)" in page and "Regressions</h3>" in page and "▲ +100.0%" in page


def test_requirement_badges_follow_thresholds(workspace):
    from poslab.report import entry_status
    from poslab.thresholds import load_thresholds

    metrics = json.loads((workspace["out"] / "metrics.json").read_text())
    th = load_thresholds(workspace["thresholds"])
    st_kf = entry_status("tiny", "gnss_kf", metrics["tiny"]["gnss_kf"], th, [])
    st_ekf = entry_status("tiny", "ekf", metrics["tiny"]["ekf"], th, [])
    kf_ok = (metrics["tiny"]["gnss_kf"]["xt_max_m"] <= 0.5
             and metrics["tiny"]["gnss_kf"]["outage_end_err_max_m"] <= 0.3)
    ekf_ok = (metrics["tiny"]["ekf"]["xt_max_m"] <= 0.5
              and metrics["tiny"]["ekf"]["outage_end_err_max_m"] <= 0.3)
    assert st_kf.req == ("fixed" if kf_ok else "known")
    assert st_ekf.req == ("pass" if ekf_ok else "fail")
    page = _report(workspace)
    assert ("unexpected pass" if kf_ok else "known failure") in page
    assert entry_status("x", "ekf", {}, th, []).req == "none"


def test_report_needs_metrics(tmp_path):
    with pytest.raises(FileNotFoundError, match="poslab run"):
        build_report(tmp_path)
    with pytest.raises(SystemExit, match="poslab run"):
        main(["baseline-update", "--out", str(tmp_path)])


def test_colour_follows_algorithm_not_rank():
    assert algo_color("ekf", ["ekf"]) == algo_color("ekf", ["gnss_kf", "ekf"])
    assert algo_color("new_a", ["ekf", "new_a", "new_b"]) != algo_color(
        "new_b", ["ekf", "new_a", "new_b"])
