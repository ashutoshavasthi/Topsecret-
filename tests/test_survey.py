import math

import numpy as np
import pytest

from skymind.assistant.skills import DroneSkills
from skymind.config import load_config
from skymind.live import SimLiveSession
from skymind.mission import FlightBrain, FlightMode
from skymind.perception.mapping import OccupancyGrid
from skymind.survey import analyse, load_prior, triangulate_landmarks

FENCE = (0.0, -2.5, 8.0, 2.5)


def two_rooms(door=(0.2, 1.2), east_known=True):
    g = OccupancyGrid(FENCE, 0.1)
    xs = g.x0 + (np.arange(g.nx) + 0.5) * g.res
    ys = g.y0 + (np.arange(g.ny) + 0.5) * g.res
    X, Y = np.meshgrid(xs, ys)
    g.log_odds[:] = np.where(g.outside, 0.0, -2.0)
    wall = (X >= 4.6) & (X <= 4.75) & ~((Y > door[0]) & (Y < door[1])) & ~g.outside
    g.log_odds[wall] = 3.5
    if not east_known:
        g.log_odds[(X > 4.8) & ~g.outside] = 0.0
    return g


def test_finds_the_doorway_and_its_width():
    scan = analyse(two_rooms(), FENCE, drone_width=0.5)
    doors = [p for p in scan.passages if p.kind == "doorway"]
    assert len(doors) == 1
    d = doors[0]
    assert 0.8 <= d.width <= 1.15 and d.passable
    assert abs(d.center[0] - 4.67) < 0.2 and abs(d.center[1] - 0.7) < 0.2
    assert any(o.kind == "wall" for o in scan.obstacles)


def test_doorway_into_unexplored_space_is_an_exit():
    scan = analyse(two_rooms(east_known=False), FENCE, drone_width=0.5)
    assert [p.kind for p in scan.passages if p.kind != "tight gap"] == ["exit"]
    assert scan.explored_fraction < 0.7


def test_too_narrow_gap_is_reported_as_impassable():
    scan = analyse(two_rooms(door=(0.5, 0.85)), FENCE, drone_width=0.5)  # a 35 cm slot in the wall
    assert not [p for p in scan.passages if p.kind == "doorway"]  # not a door...
    gap = [p for p in scan.passages if abs(p.center[0] - 4.67) < 0.2][0]
    assert gap.kind == "tight gap" and not gap.passable and gap.width < 0.5  # ...but a gap I can't fit through


def test_low_obstacles_give_a_safe_altitude():
    g = two_rooms()
    r, c = g.to_cell(np.array([[2.0, 2.0], [2.1, 2.0], [2.2, 2.0]]))
    scan = analyse(g, FENCE, 0.5, {(int(a), int(b)): 0.75 for a, b in zip(r, c)})
    assert len(scan.low_obstacles) == 1 and scan.recommended_altitude == pytest.approx(1.05)
    assert "low obstacle" in scan.summary()


def test_triangulation_locates_what_the_camera_saw():
    rng = np.random.default_rng(0)
    target = np.array([7.9, 1.8])
    rays = []
    for view in ([1.0, 0.0], [3.0, -1.0], [5.5, 0.5]):  # three places, bearings only (no range)
        for _ in range(10):
            d = target - view
            rays.append(("yellow_marker", np.array(view), math.atan2(d[1], d[0]) + rng.normal(0, 0.01), 0.9))
    rays += [("yellow_marker", np.array([2.0, 2.0]), rng.uniform(-3, 3), 0.5) for _ in range(5)]  # junk
    lms = triangulate_landmarks(rays, FENCE)
    assert len(lms) == 1 and np.linalg.norm(np.array(lms[0].position) - target) < 0.1
    # Seen from one spot only: direction is known but not distance, so nothing is reported.
    assert triangulate_landmarks([r for r in rays[:10]], FENCE) == []


def test_terrain_following_and_no_time_limit_in_interactive_mode():
    cfg = load_config("config/nano_room.yaml")
    b = FlightBrain(cfg, 1.5)
    b.altitude = 0.9
    b.ground_below = 0.75  # flying over a table
    assert b.altitude_command == pytest.approx(0.75 + cfg.avoidance.min_ground_clearance)
    b.ground_below = 0.0
    assert b.altitude_command == pytest.approx(0.9)
    b.interactive = True
    b.mode = FlightMode.HOLD
    from skymind.control.estimator import StateEstimate
    est = StateEstimate(np.array([1.0, 0.0, 0.9]), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))
    b._check_failsafes(est, 1.0, cfg.mission.max_duration + 100.0, [])
    assert b.mode is FlightMode.HOLD


@pytest.fixture(scope="module")
def surveyed(tmp_path_factory):
    out = tmp_path_factory.mktemp("survey")
    cfg = load_config("config/nano_room.yaml")
    s = SimLiveSession(cfg, out, speed=0)
    s.start()
    result = DroneSkills(s).survey()
    yield s, result, out
    s.stop()


def test_full_room_survey(surveyed):
    s, result, out = surveyed
    assert result.ok and s.collisions == 0
    scan = s.room_scan
    assert scan.explored_fraction > 0.85
    doors = [p for p in scan.passages if p.kind in ("doorway", "exit")]
    real = [d for d in doors if abs(d.center[0] - 4.67) < 0.3 and 0.2 < d.center[1] < 1.2]
    assert len(real) == 1 and real[0].passable  # the actual doorway, found and judged passable
    assert len(doors) <= 2 and all(d.width >= 0.5 for d in doors)
    truth = {"red_marker": [(4.55, 2.0), (7.95, -1.8)], "yellow_marker": [(7.95, 1.8)]}
    for lm in scan.landmarks:
        assert min(np.hypot(lm.position[0] - x, lm.position[1] - y) for x, y in truth[lm.label]) < 0.3
    assert len(scan.landmarks) >= 2
    assert any(lo.height == pytest.approx(0.75) for lo in scan.low_obstacles)  # the table
    assert any(name.startswith("doorway") or name.startswith("exit") for name in s.cfg.mission.places)
    for f in ("room_map.npz", "room_scan.json", "room_scan.png"):
        assert (out / f).exists()
    fresh = FlightBrain(load_config("config/nano_room.yaml"), 1.5)
    assert load_prior(fresh.planner.grid, out / "room_map.npz")
    assert (fresh.planner.grid.log_odds > 0.6).sum() > 100


def test_live_dodge_in_the_app_simulator(tmp_path):
    cfg = load_config("config/nano_room.yaml")
    s = SimLiveSession(cfg, tmp_path, speed=0)
    s.start()
    try:
        s.takeoff(1.1)
        assert s.wait_for(lambda st: st["mode"] == "hold", 20)
        for _ in range(4):
            assert s.throw_ball(4.0)
            s.wait_for(lambda st: False, 2.5)
        stats = s.status()["dodge"]
        assert stats["thrown"] == 4 and stats["dodged"] >= 3
        assert s.collisions == 0 and s.status()["mode"] == "hold"
    finally:
        s.stop()
