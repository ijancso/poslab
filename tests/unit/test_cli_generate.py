"""CLI ``generate`` + loader round trip, and the committed scenario files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from poslab.__main__ import main
from poslab.io.loaders import load_scenario
from poslab.sim.scenario import load_config, simulate

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = REPO / "data" / "scenarios"
REQUIRED = {"open_sky_rtk", "standalone_noisy", "outage_10s_row", "outage_30s_turn",
            "heavy_vibration", "wheel_slip_5pct"}


@pytest.fixture
def scenario_dir(tmp_path: Path) -> Path:
    """Two tiny scenarios, one of them marked smoke."""
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "tiny_a.yaml").write_text("seed: 1\nsmoke: true\nfield: {rows: 2, row_length: 20.0}\n")
    (d / "tiny_b.yaml").write_text("seed: 2\nfield: {rows: 1, row_length: 10.0}\n")
    return d


def test_generate_smoke_writes_files_and_loader_round_trips(scenario_dir, tmp_path, capsys):
    out = tmp_path / "gen"
    assert main(["generate", "--smoke", "--scenarios", str(scenario_dir), "--out", str(out)]) == 0
    assert (out / "tiny_a").is_dir() and not (out / "tiny_b").exists()
    assert "tiny_a" in capsys.readouterr().out
    for f in ("truth.csv", "gnss.csv", "imu.csv", "odom.csv", "meta.yaml"):
        assert (out / "tiny_a" / f).exists()

    loaded = load_scenario(out / "tiny_a")
    fresh = simulate(load_config(scenario_dir / "tiny_a.yaml"))
    assert loaded.name == "tiny_a" and loaded.meta["seed"] == 1
    assert loaded.meta["generator_version"] == fresh.meta["generator_version"]
    np.testing.assert_allclose(loaded.gnss.to_numpy(), fresh.gnss.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(loaded.truth.to_numpy(), fresh.truth.to_numpy(), atol=1e-6)


def test_generate_by_name_and_all(scenario_dir, tmp_path):
    out = tmp_path / "gen"
    assert main(["generate", "tiny_b", "--scenarios", str(scenario_dir), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["tiny_b"]
    assert main(["generate", "--all", "--scenarios", str(scenario_dir), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["tiny_a", "tiny_b"]


def test_generate_errors(scenario_dir, tmp_path):
    with pytest.raises(SystemExit, match="unknown scenario"):
        main(["generate", "nope", "--scenarios", str(scenario_dir), "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="no scenario files"):
        main(["generate", "--scenarios", str(tmp_path / "empty"), "--out", str(tmp_path)])


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate"):
        load_scenario(tmp_path)
    (tmp_path / "meta.yaml").write_text("name: x\n")
    for name in ("truth", "gnss", "imu", "odom"):
        (tmp_path / f"{name}.csv").write_text("t\n0.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_scenario(tmp_path)


def test_committed_scenarios_are_valid():
    """Every required scenario exists, parses, and exactly two are marked smoke.
    Outage windows lie inside the (default-field) run time."""
    files = {f.stem: f for f in SCENARIOS.glob("*.yaml")}
    assert REQUIRED <= set(files)
    smoke = [n for n, f in files.items() if yaml.safe_load(f.read_text()).get("smoke")]
    assert len(smoke) == 2
    for f in files.values():
        cfg = load_config(f)
        for t0, t1 in cfg["gnss"]["outages"]:
            assert 0 < t0 < t1 < 700


def test_outage_30s_turn_covers_a_whole_turn():
    scn = simulate(load_config(SCENARIOS / "outage_30s_turn.yaml"))
    (t0, t1), = scn.meta["gnss"]["outages"]
    turns = [iv for iv in scn.meta["intervals"] if iv["kind"] == "turn"]
    assert any(t0 < iv["t0"] and iv["t1"] < t1 for iv in turns)
