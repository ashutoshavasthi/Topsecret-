import math

import numpy as np

from skymind.config import AvoidanceConfig, WorldConfig
from skymind.perception.avoidance import AvoidMode, ObstacleAvoider
from skymind.sim.world import World

ANGLES = np.linspace(-np.pi, np.pi, 72, endpoint=False)


def test_raycast_cylinder_and_box():
    w = World(WorldConfig(cylinders=[[5.0, 0.0, 1.0, 4.0]], boxes=[[-3.0, -1.0, -2.0, 1.0, 4.0]]))
    hits = w.raycast(np.array([0.0, 0.0, 1.0]), np.array([0.0, np.pi, np.pi / 2]), 10.0)
    assert np.isclose(hits.distance[0], 4.0)
    assert np.isclose(hits.distance[1], 2.0)
    assert np.isclose(hits.distance[2], 10.0) and hits.obstacle[2] == -1


def test_raycast_respects_height():
    w = World(WorldConfig(cylinders=[[5.0, 0.0, 1.0, 2.0]]))
    assert np.isclose(w.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0]), 10.0).distance[0], 10.0)


def test_clearance():
    w = World(WorldConfig(cylinders=[[5.0, 0.0, 1.0, 4.0]]))
    assert np.isclose(w.clearance(np.array([2.0, 0.0, 1.0])), 2.0)
    assert np.isclose(w.clearance(np.array([5.0, 0.0, 7.0])), 3.0)


def _scan(world, pos, max_range=8.0):
    return world.raycast(np.array(pos), ANGLES, max_range).distance


def test_clear_path_goes_straight():
    av = ObstacleAvoider(AvoidanceConfig(), 8.0)
    cmd = av.plan(np.full(72, 8.0), ANGLES, np.array([10.0, 0.0]), 2.5)
    assert cmd.mode is AvoidMode.CLEAR
    assert np.allclose(cmd.velocity_xy, [2.5, 0.0])


def test_steers_around_obstacle_ahead():
    w = World(WorldConfig(cylinders=[[3.0, 0.0, 0.6, 5.0]]))
    av = ObstacleAvoider(AvoidanceConfig(), 8.0)
    cmd = av.plan(_scan(w, [0, 0, 2]), ANGLES, np.array([10.0, 0.0]), 2.5)
    assert cmd.mode is AvoidMode.DETOUR
    assert abs(math.degrees(cmd.heading)) > 10  # deviates from the goal line
    # and the chosen direction really is free of the obstacle for the drone's width
    assert av.corridor_clearance(_scan(w, [0, 0, 2]), ANGLES, np.array([cmd.heading]))[0] > 3.0


def test_rejects_gap_narrower_than_drone():
    # Two poles 1.0 m apart (surface to surface) ahead: too narrow for a 0.9 m safety radius.
    w = World(WorldConfig(cylinders=[[3.0, 0.8, 0.3, 5.0], [3.0, -0.8, 0.3, 5.0]]))
    av = ObstacleAvoider(AvoidanceConfig(), 8.0)
    ranges = _scan(w, [0, 0, 2])
    assert av.corridor_clearance(ranges, ANGLES, np.array([0.0]))[0] < 3.0
    cmd = av.plan(ranges, ANGLES, np.array([10.0, 0.0]), 2.5)
    assert cmd.mode is not AvoidMode.CLEAR


def test_blocked_when_boxed_in():
    w = World(WorldConfig(cylinders=[[0.0, 0.0, 1.6, 5.0]]))  # we are inside a ring-like wall
    w = World(WorldConfig(boxes=[[1.2, -5, 3, 5, 5], [-3, -5, -1.2, 5, 5], [-3, 1.2, 3, 5, 5], [-3, -5, 3, -1.2, 5]]))
    av = ObstacleAvoider(AvoidanceConfig(), 8.0)
    cmd = av.plan(_scan(w, [0, 0, 2]), ANGLES, np.array([10.0, 0.0]), 2.5)
    assert cmd.mode is AvoidMode.BLOCKED
    assert np.linalg.norm(cmd.velocity_xy) <= 0.31


def test_speed_limited_by_clearance():
    w = World(WorldConfig(boxes=[[3.0, -5.0, 4.0, 5.0, 5.0]]))  # wall 3 m ahead, goal behind it
    av = ObstacleAvoider(AvoidanceConfig(), 8.0)
    cmd = av.plan(_scan(w, [0, 0, 2]), ANGLES, np.array([10.0, 0.0]), 5.0)
    assert np.linalg.norm(cmd.velocity_xy) < 5.0
