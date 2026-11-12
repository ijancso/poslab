# poslab — Positioning Replay Test Harness (project spec for Claude Code)

## Context
Learning/portfolio project, built inside the existing `gnss-phone-lab` repo (which I may rename
to `poslab` on GitHub). Keep existing scripts working; put the new code in a package `poslab/`.

## Step 0 — integrate with the existing repo (do this before M1, then STOP)
1. Inspect the repo and list what exists: scripts, data folders, dependency files
   (requirements.txt / pyproject.toml), tests, CI workflows, README.
2. Propose an integration plan and wait for my OK. Default proposal:
   - move the earlier phone experiments to `legacy/` (keep them runnable, fix import paths)
   - one dependency system only: `pyproject.toml` (with a `dev` extra); fold any
     requirements.txt into it
   - reconcile `data/` with the layout below (move/rename, don't delete data)
   - keep any existing tests passing; merge existing CI into the new workflow
   - README: poslab first; one short "Earlier experiments" section pointing to `legacy/`
3. Flag anything that conflicts with this spec instead of silently choosing.

Goal: show, in a small but real codebase, how I would test a positioning algorithm for an
autonomous agriculture vehicle: recorded/simulated test data → algorithm under test →
metrics vs ground truth → pass/fail thresholds → regression against a stored baseline →
a clear report for developers → all of it running in CI.

It mirrors a test-automation role that asks for: SIL/HIL test automation, Python (+ MATLAB)
test cases with high code coverage, data analysis tools that present positioning and sensor
performance baselines, GNSS/IMU/odometry/sensor fusion, Linux, Git, CAN (J1939).

Honesty rule: the README must say plainly that the data is synthetic (plus a few of my own
phone recordings) and this is a learning project. No claims of field-testing experience.

## Tech constraints
- Python 3.11+, small dependency set: numpy, scipy, pandas, matplotlib, pyyaml.
  Dev: pytest, pytest-cov, ruff. CLI with argparse (no typer/click).
- Type hints and short docstrings everywhere. Readable over clever.
- Deterministic: every random process seeded from the scenario file.
- Core math in a local 2D ENU frame (metres, x=east, y=north). No lat/lon in the core.

## Repo layout
```
poslab/
  __main__.py          # CLI: generate | run | report | baseline-update
  sim/scenario.py      # synthetic tractor scenario generator
  io/loaders.py        # load generated scenarios + GnssLogger recordings
  dsp.py               # spectrum + filtering
  fusion/gnss_kf.py    # baseline algorithm: GNSS-only constant-velocity KF
  fusion/ekf.py        # algorithm under test: GNSS + odometry + IMU EKF
  fusion/registry.py   # algorithm name -> class, each with a VERSION string
  metrics.py
  baseline.py          # load/compare/promote baseline metrics
  report.py            # static HTML report
data/
  scenarios/*.yaml     # scenario definitions (small, committed)
  recordings/<name>/   # my GnssLogger .txt + meta.yaml (committed if small)
  generated/           # build output, gitignored
baseline/metrics.json  # committed baseline
thresholds.yaml        # per-scenario pass/fail limits
tests/
  unit/                # dsp, kf, metrics, loaders
  scenario/            # parametrized scenario × algorithm tests
.github/workflows/ci.yml
notes/                 # gitignored: my private code walkthrough (see Docs)
```

## 1. Scenario generator (`sim/scenario.py`)
Field pattern: N parallel rows (default 8 rows × 200 m, 6 m spacing) joined by headland
U-turns (semicircle, r = 3 m). Speed 2.5 m/s on rows, 1.2 m/s in turns, smooth transitions.
Truth sampled at 100 Hz: t, x, y, heading, speed, yaw_rate, accel_long.

Sensors (all parameters in the scenario YAML):
- GNSS, 10 Hz: position noise σ (e.g. 0.02 m "RTK-like", 1.5 m "standalone"), optional
  slow bias random walk, optional multipath spikes, outage windows `[[t0, t1], ...]`.
- IMU, 100 Hz: longitudinal + lateral accel and yaw rate, with white noise and constant bias.
  Add engine vibration on accel: sinusoid at `vib_hz` (default 27 Hz) with amplitude `vib_amp`.
- Wheel odometry, 50 Hz: speed with scale-factor error (wheel slip, e.g. +2%) and noise.

Output per scenario in `data/generated/<name>/`: truth.csv, gnss.csv, imu.csv, odom.csv,
meta.yaml (resolved parameters + seed).

Provide at least these scenarios: `open_sky_rtk`, `standalone_noisy`, `outage_10s_row`,
`outage_30s_turn`, `heavy_vibration`, `wheel_slip_5pct`. Mark 2 of them as `smoke: true`.

## 2. Recordings loader (`io/loaders.py`)
Parse GnssLogger `Fix` lines (GPS provider) into t, lat, lon, accuracy; convert to local ENU
relative to the first fix (simple equirectangular is fine, document the approximation).
Recordings have no truth: only static-noise metrics apply (see Metrics).

## 3. DSP (`dsp.py`)
- `dominant_frequency(signal, fs)` using Welch PSD.
- `lowpass(signal, fs, cutoff, order, causal: bool)`: Butterworth SOS; `sosfiltfilt` offline,
  `sosfilt` when causal. The fusion must use the causal one — document the group delay
  trade-off in a comment.

## 4. Algorithms
- `gnss_kf`: 2D constant-velocity KF on GNSS position only (the "old" baseline algorithm).
- `ekf`: state [x, y, heading, v]. Predict with gyro yaw rate and low-pass-filtered
  longitudinal accel (unicycle model). Updates: GNSS position when available, odometry speed.
  Process/measurement noise from a config dict. Must coast through GNSS outages.
Common interface: `run(data) -> DataFrame[t, x, y, heading, v]`, plus `VERSION`.

## 5. Metrics (`metrics.py`)
Interpolate truth to estimate timestamps, then:
- horizontal error: RMS, CEP50, CEP95, max
- cross-track error (perpendicular distance to the truth path — key for row guidance): RMS, max
- outage drift: error at the end of each outage, and error vs time-into-outage
- static (recordings): std east/north, CEP50/95 around the mean
Return a flat dict per (scenario, algorithm).

## 6. Tests
Unit:
- `dominant_frequency` finds an injected 27 Hz tone within ±0.5 Hz
- low-pass attenuates the vibration by > 20 dB and passes 1 Hz motion
- KF/EKF converge on a noiseless straight line; EKF coasts on a straight outage
- metrics return known values on hand-built error cases
- loader parses a small GnssLogger fixture file
Scenario (parametrized scenario × algorithm), marker `scenario`; `smoke` subset marker:
- metrics within `thresholds.yaml`
- regression: fail if any metric is worse than `baseline/metrics.json` by more than its
  tolerance (default 5%); print a readable diff table on failure
Coverage target: ≥ 85% on `poslab/`.

## 7. Baseline (`baseline.py`) and CLI
- `python -m poslab generate [--all|--smoke]`
- `python -m poslab run --algo ekf [--smoke]` → `out/metrics.json`
- `python -m poslab report` → `out/report/index.html`
- `python -m poslab baseline-update` → copies current metrics to baseline (prints the diff
  first; asks for confirmation unless `--yes`)

## 8. Report (`report.py`) — the "performance baseline" tool
One static HTML page (no web framework; PNGs embedded as base64):
- summary table per scenario: current vs baseline, delta, colored better/worse, pass/fail
- per scenario: trajectory overlay (truth / GNSS / estimate), error vs time with outages
  shaded, cross-track error histogram, accel spectrum before/after the low-pass filter
- algorithm comparison: `gnss_kf` vs `ekf` on the same scenarios
Written for a developer who did not write the algorithm: short captions, units everywhere.

## 9. CI (`.github/workflows/ci.yml`)
- on push/PR: ruff, unit tests, smoke scenarios, coverage report
- nightly (schedule): all scenarios + full report
- upload `out/report/` and the coverage HTML as artifacts

## Milestones (stop after each, show me how to run it, wait for review)
- M0: Step 0 integration (above)
- M1: scaffolding, scenario generator, a truth/sensor plot script
- M2: DSP, `gnss_kf`, metrics, unit tests
- M3: `ekf`, scenario tests, thresholds
- M4: baseline regression, HTML report, CI

## Stretch (only after M4)
- S1 MATLAB: `matlab/` with a port of the EKF step and a `matlab.unittest` test reading the
  same CSVs (run manually in MATLAB Online; not in CI).
- S2 SIL over virtual CAN (Linux): `tools/setup_vcan.sh` (modprobe vcan, create vcan0);
  `tools/replay_can.py` streams odometry in real time as J1939 PGN 65265 (CCVS),
  SPN 84 wheel-based vehicle speed (1/256 km/h per bit) using python-can; a fusion node reads
  from vcan0. Test: the SIL run reproduces the offline result within tolerance.
- S3 Formats: TDMS loader via npTDMS (VeriStand logs in TDMS); NMEA sentence parsing of the
  GnssLogger NMEA lines.

## Docs
- `README.md` (public): poslab first — what it is, the honesty note, quick start, an example report
  screenshot, how it maps to SIL/HIL testing practice, limitations.
- `notes/WALKTHROUGH.md` (gitignored, private, for me): module-by-module explanation of the
  code and the math (KF/EKF equations, Welch PSD, Butterworth, CEP, cross-track), written so
  I can explain every design choice in an interview.
