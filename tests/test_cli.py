from gnsslab.cli import main


def test_demo_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data"
    assert main(["simulate", "--out", str(data)]) == 0
    assert main(["static", f"open={data / 'static_open_sky.txt'}", f"urban={data / 'static_urban.txt'}",
                 "--out", str(tmp_path / "static")]) == 0
    assert main(["drive", str(data / "drive.csv"), "--out", str(tmp_path / "drive")]) == 0
    for name in ("static/report.md", "static/static_scatter.png", "drive/report.md", "drive/drive_drift.png"):
        assert (tmp_path / name).exists()
