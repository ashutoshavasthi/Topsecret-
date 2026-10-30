import json

from skymind.cli import main
from skymind.config import load_config
from skymind.mission import SimulationRunner


def test_open_field_mission(tmp_path):
    cfg = load_config(overrides={"mission": {"waypoints": [[8.0, 0.0]]}})
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.status == "landed"
    assert r.waypoints_reached == 1 and r.collisions == 0


def test_forest_mission_avoids_all_obstacles(tmp_path):
    cfg = load_config("config/default.yaml")
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.status == "landed", r.events
    assert r.collisions == 0
    assert r.waypoints_reached == r.waypoints_total
    assert r.min_clearance > cfg.vehicle.body_radius


def test_wall_forces_climb(tmp_path):
    # A low wall spanning the whole geofence: the only way through is over it.
    cfg = load_config(overrides={
        "mission": {"waypoints": [[10.0, 0.0]], "blocked_timeout": 2.0, "geofence": [-5, -6, 15, 6]},
        "world": {"boxes": [[4.0, -20.0, 5.0, 20.0, 3.0]]},
    })
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.collisions == 0
    assert r.waypoints_reached == 1
    assert any("climbing" in e for _, e in r.events)


def test_low_battery_triggers_rtl(tmp_path):
    cfg = load_config(overrides={"battery": {"capacity_wh": 0.5}, "mission": {"waypoints": [[30.0, 0.0]], "geofence": [-5, -5, 40, 5]}})
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.failsafe == "low battery"
    assert r.status == "landed"


def test_cli_sim_writes_report_and_video(tmp_path):
    out = tmp_path / "run"
    code = main(["sim", "--config", "config/default.yaml", "--out", str(out)])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["verdict"] == "PASS" and report["overall"] > 70
    for name in ["flight.mp4", "telemetry.jsonl", "report.md", "flight_map.png", "timeline.png", "mission.json"]:
        assert (out / name).exists(), name
    assert any((out / "snapshots").iterdir())

    # Offline analysis of the recorded video works on its own.
    assert main(["analyze", str(out / "flight.mp4"), "--out", str(tmp_path / "ana"), "--every", "5"]) == 0
    assert (tmp_path / "ana" / "report.md").exists()
