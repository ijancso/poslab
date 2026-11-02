import pytest

from gnsslab.io import load_track, parse_time

GNSSLOGGER = """\
# Header Description:
# Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis,SpeedAccuracyMps
Raw,1,2,3
Fix,GPS,47.5,19.0,120.0,0.1,3.5,,1780000000000,0.5
Fix,FLP,47.6,19.1,120.0,0.1,30.0,,1780000000300,0.5
Fix,GPS,47.50001,19.00001,121.0,0.1,3.0,,1780000001000,0.5
"""

GPX = """<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><trkseg>
  <trkpt lat="47.5" lon="19.0"><ele>100</ele><time>2026-06-01T10:00:00Z</time>
   <extensions><speed>3.2</speed><accuracy>4</accuracy></extensions></trkpt>
  <trkpt lat="47.5001" lon="19.0001"><ele>101</ele><time>2026-06-01T10:00:01.500Z</time></trkpt>
 </trkseg></trk>
</gpx>
"""

CSV = """time,lat,lon,elevation,accuracy,speed,provider
2026-06-01T10:00:01Z,47.5001,19.0001,100,5,2,gps
2026-06-01T10:00:00Z,47.5,19.0,100,5,2,gps
2026-06-01T10:00:00Z,47.5,19.0,100,5,2,gps
2026-06-01T10:00:02Z,47.7,19.2,100,50,0,network
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_gnsslogger_keeps_only_gps_fixes(tmp_path):
    tr = load_track(write(tmp_path, "log.txt", GNSSLOGGER))
    assert len(tr) == 2
    assert tr.acc.tolist() == [3.5, 3.0]
    assert tr.t[1] - tr.t[0] == pytest.approx(1.0)


def test_gnsslogger_unknown_provider_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="providers in file"):
        load_track(write(tmp_path, "log.txt", GNSSLOGGER), provider="nlp")


def test_gpx_with_extensions(tmp_path):
    tr = load_track(write(tmp_path, "a.gpx", GPX))
    assert len(tr) == 2
    assert tr.speed[0] == 3.2 and tr.acc[0] == 4.0
    assert tr.t[1] - tr.t[0] == pytest.approx(1.5)


def test_csv_sorts_dedups_and_filters_provider(tmp_path):
    tr = load_track(write(tmp_path, "a.csv", CSV))
    assert len(tr) == 2
    assert tr.lat[0] == 47.5 and tr.alt[0] == 100


def test_parse_time_formats():
    assert parse_time("1780000000") == 1780000000
    assert parse_time("1780000000123") == pytest.approx(1780000000.123)
    assert parse_time("1970-01-01T00:01:00Z") == 60
