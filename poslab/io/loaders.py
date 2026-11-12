"""Load generated scenarios (and, from M2, GnssLogger phone recordings)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from poslab.sim.scenario import Scenario

SCENARIO_COLUMNS: dict[str, list[str]] = {
    "truth": ["t", "x", "y", "heading", "speed", "yaw_rate", "accel_long"],
    "gnss": ["t", "x", "y"],
    "imu": ["t", "accel_long", "accel_lat", "yaw_rate"],
    "odom": ["t", "speed"],
}


def load_scenario(path: str | Path) -> Scenario:
    """Load a generated scenario directory (``data/generated/<name>/``)."""
    path = Path(path)
    if not (path / "meta.yaml").exists():
        raise FileNotFoundError(f"{path}: no meta.yaml - run 'python -m poslab generate' first")
    with open(path / "meta.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    frames: dict[str, pd.DataFrame] = {}
    for name, cols in SCENARIO_COLUMNS.items():
        df = pd.read_csv(path / f"{name}.csv")
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{path / name}.csv: missing columns {sorted(missing)}")
        frames[name] = df[cols]
    return Scenario(name=meta.get("name", path.name), meta=meta, **frames)
