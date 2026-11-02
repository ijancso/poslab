# gnss-phone-lab

A small, measurement-first lab for learning GNSS positioning with an ordinary Android phone.

Two questions, answered with real recordings rather than datasheets:

1. **Static test** – how accurate is phone GNSS when the phone does not move, in open sky versus between buildings?
2. **Drive test** – if GNSS drops out while driving (tunnel, underpass, urban canyon), how far does a simple Kalman filter drift in 5, 10, 20, 30 seconds?

> **Status:** the analysis pipeline works and is tested (unit tests + synthetic end-to-end run in CI). Real phone recordings are being collected; the results section will be filled from those.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q

# synthetic demo data (clearly marked as synthetic inside the files)
gnsslab simulate --out data/sample
gnsslab static open=data/sample/static_open_sky.txt urban=data/sample/static_urban.txt
gnsslab drive data/sample/drive.csv
```

Each command prints a summary table and writes `report.md`, `metrics.json` and PNG figures to `results/`.

## Recording data with an Android phone

Any of these work as input:

| App | Format | Notes |
|---|---|---|
| Google **GnssLogger** | `.txt` | Uses the `Fix` records. Only `GPS` provider fixes are kept, so fused/network locations do not contaminate the result. |
| **GPSLogger** (or similar) | `.gpx` / `.csv` | Log GNSS only (disable network location in the app). |
| Anything else | `.csv` | Needs time, lat and lon columns; accuracy, altitude and speed are optional. |

**Static protocol**

- Put the phone on a fixed spot (tripod, wall, car roof). Wait until it has a fix, then record 15–20 minutes.
- Do it twice: once in **open sky**, once **between buildings**. Same phone, same app settings.
- Optional: a surveyed point (e.g. a geodetic marker) lets you measure absolute accuracy with `--ref lat,lon`. Without it, the tool measures precision (scatter around the session's mean).

**Drive protocol**

- Phone mounted in the car, 1 Hz logging, 20–30 minutes of mixed driving (straights, turns, junctions).
- The passenger operates the phone. The outages are simulated afterwards, so normal driving is enough.

## Usage

```bash
# static: several sessions side by side
gnsslab static open=data/raw/open.txt urban=data/raw/street.txt
gnsslab static marker=data/raw/marker.txt --ref 47.49790,19.04020   # absolute accuracy

# drive: outages of 5/10/20/30 s, default process noise q = 3 m²/s³
gnsslab drive data/raw/drive.txt
gnsslab drive data/raw/drive.txt --durations 5,15,60 --q 1
```

## Method

**Coordinates.** All positions are converted to a local East-North-Up frame in metres (WGS84 → ECEF → ENU), so errors are real distances.

**Static metrics.** CEP50 / CEP95 (radius holding 50 % / 95 % of fixes), horizontal RMS, max error, vertical spread. Two extra checks:

- *Inside reported accuracy* – Android reports accuracy as a 68 % radius. A well-calibrated phone has ~68 % of fixes inside it; a much lower share means the phone is over-confident.
- *Error correlation time* – GNSS errors wander slowly. Fixes closer together than this time are not independent, so 900 fixes are not 900 independent samples.

**Drive: filter.** A 2-D constant-velocity Kalman filter (state: position and velocity) with white-noise acceleration as process noise. The measurement noise comes from the phone's reported accuracy (68 % radius → per-axis sigma, ÷1.51). A chi-square innovation gate drops multipath jumps.

**Drive: outages.** The filter runs once over the whole drive with all fixes. Then, at many points where the car is moving, GNSS is cut for N seconds and the filter predicts forward with no measurements. Drift is the distance between that prediction and the solution that had GNSS. Two comparisons:

- a baseline that simply *holds the last position* – anything useful must beat it;
- the filter's *own* uncertainty: an honest filter has ~95 % of real drifts inside its 95 % error ellipse. This column is how `q` gets tuned: too low and the filter is over-confident, too high and it is needlessly pessimistic.

## Results

Real recordings in progress. Planned table:

| Test | Phone | CEP50 | CEP95 | Inside reported acc. |
|---|---|---|---|---|
| Static, open sky | | | | |
| Static, between buildings | | | | |

| Outage | KF drift median | KF drift p95 | Hold-position drift median |
|---|---|---|---|
| 5 s | | | |
| 10 s | | | |
| 30 s | | | |

## Limitations and next steps

- The constant-velocity model cannot know about turns or braking during an outage – that is exactly why drift grows fast. The natural next step is fusing the phone's **IMU** (accelerometer + gyroscope) to propagate through the outage.
- The drift reference is the filter with GNSS, which has its own error of a few metres – fine for drift over tens of metres, not for sub-metre questions.
- Only position fixes are used. GnssLogger also records **raw pseudoranges**; computing positions from them is a later step.
- A forward-backward (RTS) smoother would give a better reference trajectory.

## Layout

```
gnsslab/
  io.py        load GnssLogger / GPX / CSV into one Track structure
  geo.py       WGS84 geodetic <-> ECEF <-> ENU
  static.py    static accuracy statistics
  kalman.py    constant-velocity Kalman filter, outage prediction
  outage.py    outage planning, drift and consistency metrics
  simulate.py  synthetic logs for demo and tests
  plots.py     figures
  report.py    Markdown + JSON reports
  cli.py       command line
tests/         unit tests and an end-to-end run
```

## License

MIT
