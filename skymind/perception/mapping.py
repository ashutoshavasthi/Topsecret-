"""Occupancy-grid mapping from lidar and A* global path planning.

The reactive avoider alone can get trapped (e.g. a goal straight behind a
building, with the way around it off to the side). The drone therefore
builds a map as it flies and plans a path on it. The path's furthest visible
point becomes the avoider's local goal.

Unknown space is treated as free (optimistic planning), so the drone
explores towards the goal and replans when it discovers new walls.
"""

from __future__ import annotations

import heapq
import math

import cv2
import numpy as np


class OccupancyGrid:
    def __init__(self, bounds: tuple[float, float, float, float], resolution: float = 0.25, margin: float = 2.0):
        xmin, ymin, xmax, ymax = bounds
        self.res = resolution
        self.x0, self.y0 = xmin - margin, ymin - margin
        self.nx = int(math.ceil((xmax - xmin + 2 * margin) / resolution))
        self.ny = int(math.ceil((ymax - ymin + 2 * margin) / resolution))
        self.log_odds = np.zeros((self.ny, self.nx), np.float32)
        # Cells outside the geofence are permanently blocked for planning.
        xs = self.x0 + (np.arange(self.nx) + 0.5) * resolution
        ys = self.y0 + (np.arange(self.ny) + 0.5) * resolution
        self.outside = ~((ys[:, None] >= ymin) & (ys[:, None] <= ymax) & (xs[None, :] >= xmin) & (xs[None, :] <= xmax))
        self.l_occ, self.l_free, self.l_min, self.l_max = 0.85, -0.4, -2.0, 3.5

    def reset(self) -> None:
        self.log_odds[:] = 0.0

    def to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(xy)
        return ((xy[:, 1] - self.y0) / self.res).astype(int), ((xy[:, 0] - self.x0) / self.res).astype(int)

    def to_world(self, row: int, col: int) -> np.ndarray:
        return np.array([self.x0 + (col + 0.5) * self.res, self.y0 + (row + 0.5) * self.res])

    def _valid(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (r >= 0) & (r < self.ny) & (c >= 0) & (c < self.nx)

    def update(self, position: np.ndarray, ranges: np.ndarray, angles: np.ndarray, max_range: float) -> None:
        o = position[:2]
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # Free space: sample along every beam up to just before the return.
        steps = np.arange(0.0, max_range, self.res * 0.5)
        pts = o[None, None, :] + dirs[:, None, :] * steps[None, :, None]  # (beams, steps, 2)
        free_mask = steps[None, :] < (ranges[:, None] - self.res)
        fr, fc = self.to_cell(pts[free_mask])
        ok = self._valid(fr, fc)
        free = np.zeros_like(self.log_odds, dtype=bool)
        free[fr[ok], fc[ok]] = True
        self.log_odds[free] += self.l_free
        # Occupied: beam endpoints that actually hit something.
        hit = ranges < max_range * 0.995  # max range means 'no return'
        hr, hc = self.to_cell(o[None, :] + dirs[hit] * ranges[hit, None])
        ok = self._valid(hr, hc)
        self.log_odds[hr[ok], hc[ok]] += self.l_occ - self.l_free * free[hr[ok], hc[ok]]
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def obstacle_mask(self, inflate: float) -> np.ndarray:
        occ = ((self.log_odds > 0.6) | self.outside).astype(np.uint8)
        k = max(1, int(math.ceil(inflate / self.res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        return cv2.dilate(occ, kernel).astype(bool)


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_expansions: int = 60000) -> list[tuple[int, int]] | None:
    ny, nx = blocked.shape
    if not (0 <= goal[0] < ny and 0 <= goal[1] < nx) or blocked[goal]:
        return None
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def h(n: tuple[int, int]) -> float:
        dr, dc = abs(n[0] - goal[0]), abs(n[1] - goal[1])
        return max(dr, dc) + 0.414 * min(dr, dc)

    g = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(h(start), 0.0, start)]
    closed = set()
    while heap and len(closed) < max_expansions:
        _, gc, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1]
        closed.add(node)
        r, c = node
        for dr, dc, cost in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ny and 0 <= nc < nx) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[r, nc] or blocked[nr, c]):
                continue  # no corner cutting
            ng = gc + cost
            nb = (nr, nc)
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                parent[nb] = node
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return None


def line_of_sight(blocked: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rs = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cs = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return not blocked[rs, cs].any()


class GlobalPlanner:
    def __init__(self, bounds: tuple[float, float, float, float], inflate: float, resolution: float = 0.25):
        self.grid = OccupancyGrid(bounds, resolution)
        self.inflate = inflate
        self.path: list[np.ndarray] = []
        self.blocked: np.ndarray | None = None

    def reset(self) -> None:
        self.grid.reset()
        self.path = []

    def plan(self, position: np.ndarray, goal: np.ndarray) -> bool:
        blocked = self.grid.obstacle_mask(self.inflate)
        sr, sc = self.grid.to_cell(position[:2])
        gr, gc = self.grid.to_cell(goal)
        start, end = (int(sr[0]), int(sc[0])), (int(gr[0]), int(gc[0]))
        if not (0 <= start[0] < blocked.shape[0] and 0 <= start[1] < blocked.shape[1]):
            self.path = []
            return False
        # The drone may sit inside the inflated zone after a close pass; free the start cell area.
        rr, cc = start
        k = max(1, int(self.inflate / self.grid.res))
        blocked[max(0, rr - k) : rr + k + 1, max(0, cc - k) : cc + k + 1] &= self.grid.log_odds[
            max(0, rr - k) : rr + k + 1, max(0, cc - k) : cc + k + 1
        ] > 0.6
        cells = astar(blocked, start, end)
        self.blocked = blocked
        if cells is None:
            self.path = []
            return False
        self.path = [self.grid.to_world(r, c) for r, c in cells]
        self._cells = cells
        return True

    def local_goal(self, position: np.ndarray, lookahead: float) -> np.ndarray | None:
        """Furthest path point within ``lookahead`` metres that is in straight line of sight."""
        if not self.path or self.blocked is None:
            return None
        pos = position[:2]
        d = [float(np.linalg.norm(p - pos)) for p in self.path]
        i0 = int(np.argmin(d))
        sr, sc = self.grid.to_cell(pos)
        start = (int(sr[0]), int(sc[0]))
        best = self.path[min(i0 + 1, len(self.path) - 1)]
        for i in range(i0, len(self.path)):
            if d[i] > lookahead:
                break
            if line_of_sight(self.blocked, start, self._cells[i]):
                best = self.path[i]
        return best
