"""Simulated range sensing: 360-degree 2D lidar (RPLidar) or a ring of ToF cones (VL53L1X)."""

from __future__ import annotations

import numpy as np

from skymind.config import SensorConfig
from skymind.perception.range_sensors import body_angles, collapse_to_cones, expand_cone_readings
from skymind.sim.world import World


class SimLidar:
    def __init__(self, cfg: SensorConfig, world: World, rng: np.random.Generator):
        self.cfg, self.world, self.rng = cfg, world, rng
        self.body_angles = body_angles(cfg)

    def scan(self, position: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (ranges, world_angles). Beams with no return read ``max_range``."""
        world_angles = self.body_angles + yaw
        max_range = self.cfg.lidar_max_range
        hits = self.world.raycast(position, world_angles, max_range)
        frac = self.cfg.tof_noise_frac if self.cfg.range_sensor == "tof_ring" else 0.0
        std = np.maximum(self.cfg.lidar_noise_std, frac * hits.distance)
        noisy = hits.distance + self.rng.normal(0.0, 1.0, len(world_angles)) * std
        # Beams with no return report exactly max_range (no noise), like real lidar drivers.
        ranges = np.where(hits.obstacle >= 0, np.clip(noisy, 0.0, max_range * 0.99), max_range)
        if self.cfg.range_sensor == "tof_ring":
            ranges = expand_cone_readings(collapse_to_cones(ranges, self.cfg), self.cfg)
        return ranges, world_angles
