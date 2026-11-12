"""Command line: ``python -m poslab generate [--all|--smoke|NAME ...]``.

``run``, ``report`` and ``baseline-update`` are added in later milestones (M2-M4).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_SCENARIOS = Path("data/scenarios")
DEFAULT_GENERATED = Path("data/generated")


def select_scenarios(scenario_dir: Path, names: list[str], smoke: bool) -> list[Path]:
    """Scenario YAML files to process: the given names, the smoke subset, or all."""
    files = sorted(scenario_dir.glob("*.yaml"))
    if not files:
        raise SystemExit(f"no scenario files in {scenario_dir}")
    if names:
        by_name = {f.stem: f for f in files}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown scenario(s) {unknown}; available: {sorted(by_name)}")
        return [by_name[n] for n in names]
    if smoke:
        files = [f for f in files if _is_smoke(f)]
    return files


def _is_smoke(path: Path) -> bool:
    """True if the scenario file has ``smoke: true``."""
    return bool((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("smoke"))


def cmd_generate(a: argparse.Namespace) -> int:
    """Generate the selected scenarios into ``--out/<name>/``."""
    from poslab.sim.scenario import generate_file

    for f in select_scenarios(Path(a.scenarios), a.names, a.smoke):
        print(f"{f.stem:20s} -> {generate_file(f, a.out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m poslab`` and the ``poslab`` command."""
    p = argparse.ArgumentParser(prog="poslab", description="Positioning replay test harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic scenario data")
    sel = g.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="all scenarios (default)")
    sel.add_argument("--smoke", action="store_true", help="only scenarios marked smoke: true")
    g.add_argument("names", nargs="*", help="scenario names (file stems) instead of --all")
    g.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS), help="scenario YAML directory")
    g.add_argument("--out", default=str(DEFAULT_GENERATED), help="output root directory")
    g.set_defaults(func=cmd_generate)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
