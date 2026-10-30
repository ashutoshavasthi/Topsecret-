"""Simulated 360-degree 2D lidar (e.g. RPLidar / ring of ToF sensors)."""

from __future__ import annotations

import numpy as np

from skymind.config import SensorConfig
from skymind.sim.world import World


class SimLidar:
    def __init__(self, cfg: SensorConfig, world: World, rng: np.random.Generator):
        self.cfg, self.world, self.rng = cfg, world, rng
        self.body_angles = np.linspace(-np.pi, np.pi, cfg.lidar_beams, endpoint=False)

    def scan(self, position: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (ranges, world_angles). Beams with no return read ``max_range``."""
        world_angles = self.body_angles + yaw
        max_range = self.cfg.lidar_max_range
        hits = self.world.raycast(position, world_angles, max_range)
        noisy = hits.distance + self.rng.normal(0.0, self.cfg.lidar_noise_std, len(world_angles))
        # Beams with no return report exactly max_range (no noise), like real lidar drivers.
        ranges = np.where(hits.obstacle >= 0, np.clip(noisy, 0.0, max_range * 0.99), max_range)
        return ranges, world_angles
