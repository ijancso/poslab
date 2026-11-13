"""Command line.

    python -m poslab generate [--all|--smoke|NAME ...]
    python -m poslab run [--algo NAME ...] [--all|--smoke|NAME ...]   -> out/metrics.json
    python -m poslab report                                          -> out/report/index.html
    python -m poslab baseline-update [--yes]      out/metrics.json -> baseline/metrics.json
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


def cmd_run(a: argparse.Namespace) -> int:
    """Run algorithms on scenarios; write metrics JSON and estimate CSVs."""
    from poslab.fusion.registry import ALGORITHMS
    from poslab.runner import ensure_generated, run_one, write_metrics

    algos = a.algo or sorted(ALGORITHMS)
    unknown = [n for n in algos if n not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"unknown algorithm(s) {unknown}; available: {sorted(ALGORITHMS)}")
    out = Path(a.out)
    results: dict[str, dict[str, dict]] = {}
    print(f"{'scenario':20s} {'algo':8s} {'h_rms':>8s} {'h_cep95':>8s} {'xt_rms':>8s} "
          f"{'xt_max':>8s} {'outage_end':>10s}   [m]")
    for f in select_scenarios(Path(a.scenarios), a.names, a.smoke):
        scn = ensure_generated(f, Path(a.generated))
        for algo in algos:
            est, m = run_one(scn, algo)
            results.setdefault(scn.name, {})[algo] = m
            est_path = out / "estimates" / scn.name / f"{algo}.csv"
            est_path.parent.mkdir(parents=True, exist_ok=True)
            est.to_csv(est_path, index=False, float_format="%.6f")
            oe = m.get("outage_end_err_max_m")
            print(f"{scn.name:20s} {algo:8s} {m['h_rms_m']:8.3f} {m['h_cep95_m']:8.3f} "
                  f"{m['xt_rms_m']:8.3f} {m['xt_max_m']:8.3f} "
                  f"{'-' if oe is None else f'{oe:.3f}':>10s}")
    print(f"\nMetrics: {write_metrics(results, out / 'metrics.json')}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    """Render the static HTML report from the last run."""
    from poslab.report import build_report

    path = build_report(a.out, a.generated, a.baseline, a.thresholds)
    print(f"Report: {path}")
    return 0


def cmd_baseline_update(a: argparse.Namespace) -> int:
    """Show the diff between the last run and the baseline, then promote it."""
    from poslab.baseline import (
        compare,
        format_deltas,
        load_regression_config,
        load_results,
        promote,
    )

    current = load_results(Path(a.out) / "metrics.json")
    if not current:
        raise SystemExit(f"{Path(a.out) / 'metrics.json'} missing - run 'python -m poslab run'")
    deltas = compare(current, load_results(a.baseline), load_regression_config(a.thresholds))
    print(f"Changes vs {a.baseline}:\n{format_deltas(deltas)}")
    if all(d.status == "same" for d in deltas):
        print("Baseline is already up to date.")
        return 0
    if not a.yes:
        answer = input(f"Write these metrics to {a.baseline}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Baseline not changed.")
            return 1
    print(f"Baseline updated: {promote(current, a.baseline)}")
    return 0


def _add_selection(p: argparse.ArgumentParser) -> None:
    """Scenario selection options shared by generate and run."""
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="all scenarios (default)")
    sel.add_argument("--smoke", action="store_true", help="only scenarios marked smoke: true")
    p.add_argument("names", nargs="*", help="scenario names (file stems) instead of --all")
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS), help="scenario YAML directory")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m poslab`` and the ``poslab`` command."""
    p = argparse.ArgumentParser(prog="poslab", description="Positioning replay test harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic scenario data")
    _add_selection(g)
    g.add_argument("--out", default=str(DEFAULT_GENERATED), help="output root directory")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run algorithms on scenarios -> out/metrics.json")
    _add_selection(r)
    r.add_argument("--algo", nargs="+", help="algorithm name(s); default: all registered")
    r.add_argument("--generated", default=str(DEFAULT_GENERATED),
                   help="generated data root (missing/stale scenarios are regenerated)")
    r.add_argument("--out", default="out", help="output directory")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="static HTML report -> out/report/index.html")
    rp.add_argument("--out", default="out", help="directory with metrics.json and estimates/")
    rp.add_argument("--generated", default=str(DEFAULT_GENERATED))
    rp.add_argument("--baseline", default="baseline/metrics.json")
    rp.add_argument("--thresholds", default="thresholds.yaml")
    rp.set_defaults(func=cmd_report)

    b = sub.add_parser("baseline-update", help="promote out/metrics.json to the baseline")
    b.add_argument("--out", default="out", help="directory with metrics.json")
    b.add_argument("--baseline", default="baseline/metrics.json")
    b.add_argument("--thresholds", default="thresholds.yaml")
    b.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    b.set_defaults(func=cmd_baseline_update)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
