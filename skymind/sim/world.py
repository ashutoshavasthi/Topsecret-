"""Simulated environment: vertical cylinders (trees, poles), boxes (buildings),
and coloured targets the vision system should find. Provides vectorised
ray casting for the lidar and the synthetic camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skymind.config import WorldConfig

INF = 1e9

TARGET_COLORS_BGR = {
    "red_marker": (30, 30, 220),
    "yellow_marker": (20, 210, 240),
}


@dataclass
class Target:
    position: np.ndarray
    radius: float
    label: str


@dataclass
class RayHits:
    distance: np.ndarray
    obstacle: np.ndarray  # index into World.obstacle_colors, -1 = nothing
    height: np.ndarray


class World:
    def __init__(self, cfg: WorldConfig):
        self.cyl = np.array(cfg.cylinders, dtype=float).reshape(-1, 4)
        self.box = np.array(cfg.boxes, dtype=float).reshape(-1, 5)
        self.targets = [Target(np.array(t[:3], dtype=float), float(t[3]), str(t[4])) for t in cfg.targets]
        self.wind = np.array(cfg.wind, dtype=float)
        self.gust_std = cfg.gust_std
        n_c, n_b = len(self.cyl), len(self.box)
        rng = np.random.default_rng(123)
        greens = [(40 + int(rng.integers(0, 40)), 110 + int(rng.integers(0, 50)), 40) for _ in range(n_c)]
        greys = [(120 + int(rng.integers(0, 40)),) * 3 for _ in range(n_b)]
        self.obstacle_colors = np.array(greens + greys, dtype=float).reshape(-1, 3)
        self.obstacle_heights = np.concatenate([self.cyl[:, 3], self.box[:, 4]]) if n_c + n_b else np.zeros(0)

    @property
    def num_obstacles(self) -> int:
        return len(self.cyl) + len(self.box)

    # ------------------------------------------------------------------ rays
    def raycast(self, origin: np.ndarray, angles: np.ndarray, max_range: float, respect_height: bool = True,
                height_tolerance: float = 0.0) -> RayHits:
        """Horizontal rays from ``origin`` at world-frame ``angles`` (radians).

        With ``respect_height``, objects lower than the sensor are missed unless they reach
        within ``height_tolerance`` of it (a cone-shaped ToF beam also sees slightly below itself)."""
        n = len(angles)
        d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n, 2)
        o = origin[:2]
        best = np.full(n, max_range)
        hit = np.full(n, -1)

        if len(self.cyl):
            oc = o[None, :] - self.cyl[:, :2]  # (m, 2)
            b = d @ oc.T  # (n, m)
            c = np.sum(oc * oc, axis=1) - self.cyl[:, 2] ** 2  # (m,)
            disc = b * b - c[None, :]
            sq = np.sqrt(np.maximum(disc, 0.0))
            t = np.where(c[None, :] < 0.0, 0.0, -b - sq)
            valid = (disc >= 0.0) & (t >= 0.0)
            if respect_height:
                valid &= (origin[2] < self.cyl[:, 3] + height_tolerance)[None, :]
            t = np.where(valid, t, INF)
            j = np.argmin(t, axis=1)
            tj = t[np.arange(n), j]
            closer = tj < best
            best = np.where(closer, tj, best)
            hit = np.where(closer, j, hit)

        if len(self.box):
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)  # (n, 2)
            tx1 = (self.box[None, :, 0] - o[0]) * inv[:, 0:1]
            tx2 = (self.box[None, :, 2] - o[0]) * inv[:, 0:1]
            ty1 = (self.box[None, :, 1] - o[1]) * inv[:, 1:2]
            ty2 = (self.box[None, :, 3] - o[1]) * inv[:, 1:2]
            tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            valid = tmax >= np.maximum(tmin, 0.0)
            if respect_height:
                valid &= (origin[2] < self.box[:, 4] + height_tolerance)[None, :]
            t = np.where(valid, np.maximum(tmin, 0.0), INF)
            j = np.argmin(t, axis=1)
            tj = t[np.arange(n), j]
            closer = tj < best
            best = np.where(closer, tj, best)
            hit = np.where(closer, j + len(self.cyl), hit)

        heights = np.where(hit >= 0, self.obstacle_heights[np.maximum(hit, 0)] if self.num_obstacles else 0.0, 0.0)
        return RayHits(best, hit, heights)

    # -------------------------------------------------------------- distance
    def clearance(self, p: np.ndarray) -> float:
        """Distance from point ``p`` to the nearest obstacle surface (or ground-free inf)."""
        best = INF
        if len(self.cyl):
            d2 = np.maximum(np.linalg.norm(self.cyl[:, :2] - p[:2], axis=1) - self.cyl[:, 2], 0.0)
            dz = np.maximum(p[2] - self.cyl[:, 3], 0.0)
            best = min(best, float(np.min(np.hypot(d2, dz))))
        if len(self.box):
            dx = np.maximum.reduce([self.box[:, 0] - p[0], np.zeros(len(self.box)), p[0] - self.box[:, 2]])
            dy = np.maximum.reduce([self.box[:, 1] - p[1], np.zeros(len(self.box)), p[1] - self.box[:, 3]])
            dz = np.maximum(p[2] - self.box[:, 4], 0.0)
            best = min(best, float(np.min(np.sqrt(dx * dx + dy * dy + dz * dz))))
        return best

    def height_below(self, p: np.ndarray) -> float:
        """Height of whatever is directly under point ``p`` (0 = floor), as a downward ToF sees it."""
        h = 0.0
        if len(self.cyl):
            under = np.linalg.norm(self.cyl[:, :2] - p[:2], axis=1) <= self.cyl[:, 2]
            under &= self.cyl[:, 3] < p[2]
            if np.any(under):
                h = max(h, float(self.cyl[under, 3].max()))
        if len(self.box):
            b = self.box
            under = (b[:, 0] <= p[0]) & (p[0] <= b[:, 2]) & (b[:, 1] <= p[1]) & (p[1] <= b[:, 3]) & (b[:, 4] < p[2])
            if np.any(under):
                h = max(h, float(b[under, 4].max()))
        return h

    def height_below_footprint(self, p: np.ndarray, radius: float, samples: int = 8, where: bool = False):
        """Highest thing inside a downward sensor's footprint (a disc of ``radius`` around ``p``).
        With ``where=True`` also returns the point where it was seen."""
        h, at = self.height_below(p), np.asarray(p, float)
        for k in range(samples):
            a = 2 * np.pi * k / samples
            q = p + np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
            hq = self.height_below(q)
            if hq > h:
                h, at = hq, q
        return (h, at) if where else h

    def wind_at(self, rng: np.random.Generator) -> np.ndarray:
        if self.gust_std <= 0.0:
            return self.wind
        return self.wind + rng.normal(0.0, self.gust_std, 3)
