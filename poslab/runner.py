"""Glue between scenarios, algorithms and metrics (used by the CLI and the scenario tests)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from poslab.fusion.base import SensorData
from poslab.fusion.registry import get_algorithm
from poslab.io.loaders import load_scenario
from poslab.metrics import compute_metrics
from poslab.sim.scenario import (
    GENERATOR_VERSION,
    Scenario,
    load_config,
    simulate,
    write_scenario,
)

# keys the generator adds to meta.yaml on top of the resolved config
_META_EXTRA = {"generator_version", "duration_s", "intervals"}


def is_up_to_date(yaml_path: Path, gen_dir: Path) -> bool:
    """True if ``gen_dir`` holds data generated from exactly this YAML and generator version."""
    meta_path = gen_dir / "meta.yaml"
    if not meta_path.exists():
        return False
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if meta.get("generator_version") != GENERATOR_VERSION:
        return False
    stored = {k: v for k, v in meta.items() if k not in _META_EXTRA}
    # compare after a YAML round trip, exactly as the config was stored
    return stored == yaml.safe_load(yaml.safe_dump(load_config(yaml_path)))


def ensure_generated(yaml_path: Path, gen_root: Path) -> Scenario:
    """Load the generated scenario, (re)generating it first if missing or stale."""
    gen_dir = Path(gen_root) / Path(yaml_path).stem
    if not is_up_to_date(Path(yaml_path), gen_dir):
        write_scenario(simulate(load_config(yaml_path)), gen_dir)
    # always read back from CSV, so fresh and cached runs give bit-identical metrics
    return load_scenario(gen_dir)


def sensor_data(scn: Scenario) -> SensorData:
    """The algorithm's view of a scenario: sensors only, no truth."""
    return SensorData(gnss=scn.gnss, imu=scn.imu, odom=scn.odom)


def run_one(scn: Scenario, algo: str, config: dict[str, Any] | None = None
            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one algorithm on one scenario; return (estimate, flat metric dict)."""
    alg = get_algorithm(algo, config)
    est = alg.run(sensor_data(scn))
    metrics = compute_metrics(est, scn.truth, scn.meta["gnss"]["outages"])
    return est, {"algo_version": alg.VERSION,
                 "generator_version": scn.meta.get("generator_version"), **metrics}


def write_metrics(results: dict[str, dict[str, dict[str, Any]]], path: Path) -> Path:
    """Write ``{scenario: {algo: metrics}}`` as pretty, sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
