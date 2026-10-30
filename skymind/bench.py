"""Measure the companion computer's CPU budget for a configuration.

Run it on the actual board (e.g. the Pi Zero 2 W) to see whether the
autonomy and vision rates in your config fit:

    python -m skymind bench --config config/micro_indoor.yaml

On a real drone the 250 Hz estimator/controller runs on the flight
controller, so only the brain (mapping, planning, avoidance) and the vision
pipeline load the companion computer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from skymind.config import SkyMindConfig
from skymind.mission import FlightBrain, FlightMode
from skymind.sim.camera_sim import SimCamera
from skymind.sim.lidar import SimLidar
from skymind.sim.world import World
from skymind.vision.analyzer import FrameAnalyzer


@dataclass
class BenchLine:
    name: str
    ms_per_call: float
    calls_per_s: float

    @property
    def cpu_fraction(self) -> float:
        return self.ms_per_call * self.calls_per_s / 1000.0


def _time(fn, n: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def run_bench(cfg: SkyMindConfig, iterations: int = 60) -> list[BenchLine]:
    rng = np.random.default_rng(0)
    world = World(cfg.world)
    lidar = SimLidar(cfg.sensors, world, rng)
    cam = SimCamera(cfg.camera, world, rng)
    brain = FlightBrain(cfg, cfg.sensors.lidar_max_range)
    brain.mode = FlightMode.MISSION
    home = np.array([cfg.mission.home[0], cfg.mission.home[1], cfg.mission.cruise_altitude])
    ranges, angles = lidar.scan(home, 0.0)

    lines = []
    if brain.planner is not None:
        grid = brain.planner.grid
        max_r = cfg.sensors.lidar_max_range
        # Map the whole site first so A* is timed on a realistic, cluttered map (worst case:
        # planning to the last waypoint), not on an empty one.
        xmin, ymin, xmax, ymax = cfg.mission.geofence
        for x in np.arange(xmin + 0.5, xmax, 1.5):
            for y in np.arange(ymin + 0.5, ymax, 1.5):
                p = np.array([x, y, cfg.mission.cruise_altitude])
                for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                    r, a = lidar.scan(p, yaw)
                    grid.update(p, r, a, max_r)
        brain.wp_index = len(brain.waypoints) - 1
        lines.append(BenchLine("map update", _time(lambda: grid.update(home, ranges, angles, max_r), iterations), cfg.sensors.lidar_rate_hz))
        replan_ms = _time(lambda: brain.planner.plan(home, brain.goal), max(5, iterations // 6))
        lines.append(BenchLine("A* replan", replan_ms, 1.0 / cfg.avoidance.replan_interval))
    to_goal = brain.goal - home[:2]
    avoid_ms = _time(lambda: brain.avoider.plan(ranges, angles, to_goal, cfg.mission.cruise_speed), iterations)
    lines.append(BenchLine("avoidance", avoid_ms, cfg.sensors.lidar_rate_hz))

    frame, _ = cam.render(home, 0.0)
    analyzer = FrameAnalyzer(cfg.vision)
    det_rate = cfg.camera.fps / max(1, cfg.vision.detect_every)
    lines.append(BenchLine("vision (detect + quality)", _time(lambda: analyzer.analyze(frame), iterations), det_rate))
    return lines


def format_bench(lines: list[BenchLine]) -> str:
    out = [f"{'stage':<28}{'ms/call':>9}{'calls/s':>9}{'CPU %':>8}", "-" * 54]
    for b in lines:
        out.append(f"{b.name:<28}{b.ms_per_call:9.2f}{b.calls_per_s:9.1f}{100 * b.cpu_fraction:8.1f}")
    total = sum(b.cpu_fraction for b in lines)
    out.append("-" * 54)
    out.append(f"{'total (one core)':<28}{'':>18}{100 * total:8.1f}")
    out.append("Video encoding is not included: use the hardware H.264 encoder on Raspberry Pi.")
    return "\n".join(out)
