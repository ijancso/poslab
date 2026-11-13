"""``python -m poslab run`` and the runner (auto-regeneration, metrics JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from poslab.__main__ import main
from poslab.runner import ensure_generated, is_up_to_date, run_one


@pytest.fixture
def scenario_dir(tmp_path: Path) -> Path:
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "tiny.yaml").write_text(
        "seed: 1\nsmoke: true\nfield: {rows: 2, row_length: 30.0}\n"
        "gnss: {outages: [[10.0, 14.0]]}\n")
    return d


def test_run_writes_metrics_and_estimates(scenario_dir, tmp_path, capsys):
    out, gen = tmp_path / "out", tmp_path / "gen"
    assert main(["run", "--smoke", "--algo", "gnss_kf", "--scenarios", str(scenario_dir),
                 "--generated", str(gen), "--out", str(out)]) == 0
    res = json.loads((out / "metrics.json").read_text())
    m = res["tiny"]["gnss_kf"]
    assert m["algo_version"] == "1.0" and m["generator_version"] == "1.1"
    # the tiny field's outage falls on the braking + turn, so gnss_kf drifts there
    assert 0.0 < m["h_rms_m"] < 1.0 and "outage1_end_err_m" in m
    assert (out / "estimates" / "tiny" / "gnss_kf.csv").exists()
    assert (gen / "tiny" / "meta.yaml").exists()            # generated on demand
    assert "tiny" in capsys.readouterr().out


def test_run_is_deterministic_between_fresh_and_cached_data(scenario_dir, tmp_path):
    args = ["run", "--scenarios", str(scenario_dir), "--generated", str(tmp_path / "gen")]
    main(args + ["--out", str(tmp_path / "a")])               # generates
    main(args + ["--out", str(tmp_path / "b")])               # reuses cached CSVs
    assert (tmp_path / "a" / "metrics.json").read_text() == \
        (tmp_path / "b" / "metrics.json").read_text()


def test_unknown_algorithm(scenario_dir, tmp_path):
    with pytest.raises(SystemExit, match="unknown algorithm"):
        main(["run", "--algo", "magic", "--scenarios", str(scenario_dir),
              "--out", str(tmp_path)])


def test_stale_generated_data_is_regenerated(scenario_dir, tmp_path):
    yaml_path, gen = scenario_dir / "tiny.yaml", tmp_path / "gen"
    assert not is_up_to_date(yaml_path, gen / "tiny")
    scn = ensure_generated(yaml_path, gen)
    assert is_up_to_date(yaml_path, gen / "tiny")
    yaml_path.write_text(yaml_path.read_text().replace("seed: 1", "seed: 2"))
    assert not is_up_to_date(yaml_path, gen / "tiny")
    scn2 = ensure_generated(yaml_path, gen)
    assert scn2.meta["seed"] == 2 and not scn2.gnss.equals(scn.gnss)
    meta = gen / "tiny" / "meta.yaml"
    meta.write_text(meta.read_text().replace("generator_version: '1.1'", "generator_version: '0'"))
    assert not is_up_to_date(yaml_path, gen / "tiny")


def test_run_one_returns_estimate_and_flat_metrics(scenario_dir, tmp_path):
    scn = ensure_generated(scenario_dir / "tiny.yaml", tmp_path / "gen")
    est, m = run_one(scn, "gnss_kf")
    assert len(est) > 0 and all(not isinstance(v, dict) for v in m.values())
