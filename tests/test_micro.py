import math

import numpy as np

from skymind.config import SensorConfig, WorldConfig, load_config
from skymind.design import evaluate, with_battery
from skymind.mission import FlightBrain, SimulationRunner
from skymind.perception.mapping import GlobalPlanner
from skymind.perception.range_sensors import body_angles, collapse_to_cones, cone_subray_angles, expand_cone_readings
from skymind.power import PowerModel, relative_noise_db
from skymind.sim.lidar import SimLidar
from skymind.sim.world import World


def test_config_base_include_and_dict_merge():
    cfg = load_config("config/micro_indoor.yaml")
    assert cfg.vehicle.mass < 0.25  # from micro.yaml
    assert cfg.sensors.range_sensor == "tof_ring"
    assert cfg.mission.waypoints[0] == [9.0, 3.0]  # from the scenario itself
    # overriding a few judge thresholds keeps the others
    assert cfg.judge.thresholds["min_clearance"] == [0.35, 0.12]
    assert "sharpness" in cfg.judge.thresholds


def test_power_model_matches_momentum_theory():
    cfg = load_config("config/micro.yaml")
    pm = PowerModel(cfg.vehicle)
    t = cfg.vehicle.mass * 9.81 / 4
    area = math.pi * (cfg.vehicle.prop_diameter / 2) ** 2
    ideal = 4 * t**1.5 / math.sqrt(2 * 1.225 * area)
    expected = ideal / (cfg.vehicle.figure_of_merit * cfg.vehicle.drivetrain_efficiency) + cfg.vehicle.avionics_power
    assert math.isclose(pm.hover_power(), expected, rel_tol=1e-9)
    assert 35 < pm.hover_power() < 70  # realistic for a ~235 g 3.5" quad
    assert 6 < pm.hover_endurance_min(cfg.battery) < 11


def test_bigger_props_are_quieter_and_more_efficient():
    small = evaluate(load_config("config/micro.yaml"))
    quiet = evaluate(load_config("config/micro_quiet.yaml"), reference_tip_speed=small.hover_tip_speed)
    assert quiet.hover_tip_speed < small.hover_tip_speed
    assert quiet.noise_vs_reference_db < -2.0
    assert quiet.hover_power_w < small.hover_power_w
    assert quiet.sub_250g and small.sub_250g
    assert math.isclose(relative_noise_db(100.0, 10.0), 50.0)


def test_battery_sweep_changes_mass_and_endurance():
    cfg = load_config("config/micro.yaml")
    big = evaluate(with_battery(cfg, 17.8))
    small = evaluate(with_battery(cfg, 4.4))
    assert big.mass_g > small.mass_g
    assert big.hover_endurance_min > small.hover_endurance_min
    assert not big.sub_250g


def test_tof_cones_report_nearest_return_in_cone():
    cfg = SensorConfig(range_sensor="tof_ring", lidar_max_range=4.0, lidar_noise_std=0.0, tof_noise_frac=0.0)
    angles = body_angles(cfg)
    assert len(angles) == len(cfg.tof_headings_deg) * cfg.tof_subrays
    assert np.allclose(cone_subray_angles(cfg).reshape(-1, cfg.tof_subrays).mean(axis=1), np.radians(cfg.tof_headings_deg))
    # A pole 2 m ahead and slightly left: only the forward cone (0 deg) sees it, at its nearest point.
    world = World(WorldConfig(cylinders=[[2.0, 0.15, 0.1, 3.0]]))
    ranges, _ = SimLidar(cfg, world, np.random.default_rng(0)).scan(np.array([0.0, 0.0, 1.0]), 0.0)
    cones = collapse_to_cones(ranges, cfg)
    fwd = cfg.tof_headings_deg.index(0.0)
    assert 1.85 < cones[fwd] < 2.0  # nearest surface is 1.906 m; 5 sub-rays resolve it to a few cm
    assert np.all(np.delete(cones, fwd) == 4.0)
    assert np.allclose(expand_cone_readings(cones, cfg), ranges)


def test_unobserved_directions_are_not_treated_as_free():
    cfg = load_config("config/micro.yaml")
    brain = FlightBrain(cfg, cfg.sensors.lidar_max_range)
    obs = brain._observed_headings(0.0)
    h = np.degrees(brain.avoider.headings)
    assert obs[np.argmin(np.abs(h - 0))] and obs[np.argmin(np.abs(h - 90))]
    assert not obs[np.argmin(np.abs(h - 60))]  # the gap between the 30 and 90 degree cones
    ranges = np.full(len(body_angles(cfg.sensors)), 3.5)
    cmd = brain.avoider.plan(ranges, body_angles(cfg.sensors), np.array([2.5, 4.33]), 1.5, obs)  # goal at 60 deg: blind
    heading = math.degrees(cmd.heading)
    assert any(abs(heading - c) <= cfg.sensors.tof_fov_deg / 2 + 1e-6 for c in cfg.sensors.tof_headings_deg)


def test_planner_never_routes_outside_geofence():
    planner = GlobalPlanner((0.0, -2.0, 10.0, 2.0), inflate=0.3, resolution=0.2)
    # Wall from the fence's south edge up to y=0.5 leaves a 1.5 m gap at the north edge, inside the fence.
    for y in np.arange(-2.0, 2.0, 0.1):
        if y < 0.5:
            r, c = planner.grid.to_cell(np.array([5.0, y]))
            planner.grid.log_odds[r, c] = 3.0
    assert planner.plan(np.array([1.0, 0.0, 1.0]), np.array([9.0, 0.0]))
    path = np.array(planner.path)
    assert np.all((path[:, 1] >= -2.0) & (path[:, 1] <= 2.0))
    # Close the gap completely: no path, rather than one around the fence.
    for y in np.arange(0.5, 2.0, 0.1):
        r, c = planner.grid.to_cell(np.array([5.0, y]))
        planner.grid.log_odds[r, c] = 3.0
    assert not planner.plan(np.array([1.0, 0.0, 1.0]), np.array([9.0, 0.0]))


def test_micro_indoor_mission(tmp_path):
    cfg = load_config("config/micro_indoor.yaml")
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.status == "landed", r.events
    assert r.collisions == 0
    assert r.waypoints_reached == r.waypoints_total
    assert r.min_clearance > cfg.vehicle.body_radius
    assert 30 < r.avg_power_w < 80 and r.energy_per_km_wh > 0


def test_cli_design_and_bench(capsys):
    from skymind.cli import main

    assert main(["design", "config/micro.yaml", "config/micro_quiet.yaml", "--sweep-battery", "4.4,9.6"]) == 0
    out = capsys.readouterr().out
    assert "micro_quiet" in out and "micro@4.4Wh" in out
    assert main(["bench", "--config", "config/micro_indoor.yaml", "--iterations", "3"]) == 0
    assert "total (one core)" in capsys.readouterr().out
