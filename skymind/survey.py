"""Room survey: explore a room, then turn the map into a "room scan".

The drone combines two kinds of input:

* **bounce-off ranging** (ToF / lidar): where things are, how big, how far
* **vision** (camera + AI): *what* they are (markers, or chairs and people with YOLO)

It explores with frontier exploration: it repeatedly flies to the edge between mapped free
space and the unknown, looks around, and maps more, until nothing reachable is left
unexplored. The scan then reports:

* **obstacles**: walls, furniture, poles (position, size, what vision saw on them)
* **doorways / exits**: gaps that connect two areas. Each one is found by asking "if this
  gap were closed, would the free space split in two?" Each gets a width and a
  passable/too-narrow verdict for this drone. Doorways that lead into unexplored space
  are reported as exits.
* **tight gaps**: narrow places between objects that are not doorways
* **low obstacles**: things below flight height (tables), seen by the downward range
  sensor when flying over them, with the minimum safe height to cross them
* **unexplored areas**: what's still unknown, so the planner treats it with caution

The scan is saved (``room_map.npz`` + ``room_scan.json``). Later flights can start with
it (``--prior-map``), so the planner routes through doorways and around furniture from
the first second instead of discovering them in flight.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from skymind.perception.mapping import OccupancyGrid

OCC, FREE = 0.6, -0.3  # log-odds thresholds: occupied / known free


@dataclass
class ScanObstacle:
    id: int
    kind: str  # "wall" | "large obstacle" | "small obstacle"
    center: list[float]
    size: list[float]  # [dx, dy] m
    labels: list[str] = field(default_factory=list)  # what vision saw on/near it


@dataclass
class ScanPassage:
    id: int
    kind: str  # "doorway" | "exit" (doorway into unexplored space) | "tight gap"
    a: list[float]  # one edge of the gap
    b: list[float]  # the other edge
    width: float  # best estimate (only strongly confirmed walls bound it)
    passable: bool  # min_width fits, or the estimate fits with a 1.5x margin
    center: list[float] = field(default_factory=list)
    min_width: float = 0.0  # conservative: every weak obstacle reading counts as solid


@dataclass
class ScanLowObstacle:
    center: list[float]
    size: list[float]
    height: float
    min_safe_altitude: float


@dataclass
class ScanLandmark:
    label: str
    position: list[float]
    confidence: float
    seen: int


@dataclass
class RoomScan:
    mapped_area_m2: float
    explored_fraction: float
    obstacles: list[ScanObstacle]
    passages: list[ScanPassage]
    low_obstacles: list[ScanLowObstacle]
    landmarks: list[ScanLandmark]
    frontier_cells: int
    recommended_altitude: float | None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        doors = [p for p in self.passages if p.kind in ("doorway", "exit")]
        exits = [p for p in self.passages if p.kind == "exit"]
        tight = [p for p in self.passages if p.kind == "tight gap"]
        parts = [f"I mapped about {self.mapped_area_m2:.0f} square metres ({self.explored_fraction * 100:.0f}% of the area)."]
        if self.obstacles:
            kinds: dict[str, int] = {}
            for o in self.obstacles:
                kinds[o.kind] = kinds.get(o.kind, 0) + 1
            parts.append("Obstacles: " + ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k, n in kinds.items()) + ".")
        if doors:
            ds = ", ".join(f"about {d.width:.1f} m wide{'' if d.passable else ' (too narrow for me)'}" for d in doors)
            parts.append(f"{len(doors)} doorway{'s' if len(doors) > 1 else ''}: {ds}.")
        if exits:
            parts.append(f"{len(exits)} of them lead{'s' if len(exits) == 1 else ''} to unexplored space.")
        if tight:
            parts.append(f"{len(tight)} tight gap{'s' if len(tight) > 1 else ''} between objects.")
        if self.low_obstacles:
            h = max(lo.height for lo in self.low_obstacles)
            parts.append(f"{len(self.low_obstacles)} low obstacle{'s' if len(self.low_obstacles) > 1 else ''} "
                         f"up to {h:.2f} m high; I'll fly at {self.recommended_altitude:.1f} m or higher over them.")
        if self.landmarks:
            parts.append("I saw " + ", ".join(sorted({lm.label.replace('_', ' ') for lm in self.landmarks})) + ".")
        return " ".join(parts)


# --------------------------------------------------------------------------- exploration
def frontier_targets(grid: OccupancyGrid, blocked: np.ndarray, position: np.ndarray, min_cells: int = 3,
                     avoid: list[np.ndarray] | None = None, pull_back: float = 0.3) -> list[np.ndarray]:
    """Viewpoints for exploring the borders between known-free and unknown space, nearest first.

    Each target is the border cell nearest its cluster's centre, pulled ``pull_back`` metres back
    toward the drone through known-free space. So the drone stops short of the unseen area, faces
    it (turn-then-go), and the sweep maps what's beyond; it never flies *into* unseen space.
    """
    lo = grid.log_odds
    free = (lo < FREE) & ~grid.outside
    unknown = (lo >= FREE) & (lo <= OCC) & ~grid.outside
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]
    nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]
    nb[:, :-1] |= unknown[:, 1:]
    frontier = (free & nb).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(frontier, connectivity=8)
    # Large unknown regions (another room, beyond a doorway) are worth a viewpoint even when their
    # border lies inside an obstacle margin; small unseen pockets beside furniture are not.
    m_u, lab_u, st_u, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), connectivity=4)
    big = np.zeros_like(unknown)
    for q in range(1, m_u):
        if st_u[q, cv2.CC_STAT_AREA] * grid.res ** 2 >= 1.0:
            big |= lab_u == q
    near_big = cv2.dilate(big.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    d_unknown = cv2.distanceTransform((~unknown).astype(np.uint8), cv2.DIST_L2, 3) * grid.res
    xs = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.res
    ys = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.res
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_cells:
            continue
        rows, cols = np.nonzero(labels == i)
        ok = ~blocked[rows, cols]
        if not np.any(ok):
            if not np.any(near_big[rows, cols]):
                continue
            # Viewpoint: nearest safe known-free cell within 1 m of this border.
            d_front = cv2.distanceTransform((labels != i).astype(np.uint8), cv2.DIST_L2, 3) * grid.res
            cand = free & ~blocked & (d_front <= 1.0) & (d_unknown >= 0.3)  # never right next to unseen space
            for a in avoid or []:
                cand &= ((xs[None, :] - a[0]) ** 2 + (ys[:, None] - a[1]) ** 2) >= 0.6 ** 2
            if not np.any(cand):
                continue
            cr, cc = np.nonzero(cand)
            k = int(np.argmin(d_front[cr, cc]))
            out.append(np.array([xs[cc[k]], ys[cr[k]]]))
            continue
        rows, cols = rows[ok], cols[ok]
        pts = np.stack([grid.x0 + (cols + 0.5) * grid.res, grid.y0 + (rows + 0.5) * grid.res], axis=1)
        p = pts[int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))]
        back = position[:2] - p
        n_back = float(np.linalg.norm(back))
        if n_back > 1e-6:
            for step in (pull_back, pull_back / 2, 0.0):
                q = p + back / n_back * min(step, n_back)
                r, c = grid.to_cell(q)
                if free[r[0], c[0]] and not blocked[r[0], c[0]]:
                    p = q
                    break
        if avoid and any(np.linalg.norm(p - a) < 0.6 for a in avoid):
            continue
        out.append(p)
    out.sort(key=lambda p: float(np.linalg.norm(p - position[:2])))
    return out


def frontier_count(grid: OccupancyGrid) -> int:
    lo = grid.log_odds
    free = (lo < FREE) & ~grid.outside
    unknown = (lo >= FREE) & (lo <= OCC) & ~grid.outside
    k = np.ones((3, 3), np.uint8)
    return int(np.sum(free & cv2.dilate(unknown.astype(np.uint8), k).astype(bool)))


# --------------------------------------------------------------------------- analysis
def _component_points(labels: np.ndarray, i: int, grid: OccupancyGrid, limit: int = 600,
                      mask: np.ndarray | None = None) -> np.ndarray:
    """World points of component ``i`` (restricted to ``mask``: the real cells, not the merge dilation)."""
    sel = labels == i if mask is None else (labels == i) & mask
    rows, cols = np.nonzero(sel)
    if len(rows) > limit:
        idx = np.linspace(0, len(rows) - 1, limit).astype(int)
        rows, cols = rows[idx], cols[idx]
    return np.stack([grid.x0 + (cols + 0.5) * grid.res, grid.y0 + (rows + 0.5) * grid.res], axis=1)


def _free_span(lo: np.ndarray, grid: OccupancyGrid, mid: np.ndarray, direction: np.ndarray, limit: float,
               strong: float = 2.5) -> float:
    """Width of the opening through ``mid`` along ``direction``, bounded only by *strongly confirmed*
    obstacle cells. Wide ToF cones smear wall ends into the gap with weak evidence; repeated
    centre-ray hits are what make a cell strong, so this recovers most of the true width."""
    n = float(np.linalg.norm(direction))
    if n < 1e-6:
        return 0.0
    u = direction / n
    total = 0.0
    for sign in (1.0, -1.0):
        d = 0.0
        while d < limit:
            p = mid + sign * d * u
            r, c = grid.to_cell(p)
            r, c = int(r[0]), int(c[0])
            if not (0 <= r < grid.ny and 0 <= c < grid.nx) or grid.outside[r, c] or lo[r, c] > strong:
                break
            d += grid.res / 2
        total += d
    return total - grid.res / 2


def _opens_into_unknown(free: np.ndarray, unknown: np.ndarray, grid: OccupancyGrid, mid: np.ndarray,
                        nrm: np.ndarray, reach: float = 0.5) -> bool:
    """Known-free on one side of the gap and unexplored on the other: an exit nobody has been through."""
    sides = []
    for sign in (1.0, -1.0):
        r, c = grid.to_cell(mid + sign * reach * nrm)
        r, c = int(r[0]), int(c[0])
        if not (0 <= r < grid.ny and 0 <= c < grid.nx):
            return False
        sides.append("free" if free[r, c] else "unknown" if unknown[r, c] else "other")
    return sorted(sides) == ["free", "unknown"]


def analyse(grid: OccupancyGrid, fence: tuple[float, float, float, float], drone_width: float,
            low_cells: dict[tuple[int, int], float] | None = None, landmarks: list[ScanLandmark] | None = None,
            max_door: float = 1.6) -> RoomScan:
    """Turn an occupancy grid (+ low-obstacle readings + vision landmarks) into a RoomScan."""
    res = grid.res
    inside = ~grid.outside
    lo = grid.log_odds
    occ = (lo > OCC) & inside
    free = (lo < FREE) & inside
    unknown = inside & ~occ & ~free
    mapped = float(np.sum(free | occ)) * res * res
    explored = float(np.sum(free | occ)) / max(float(np.sum(inside)), 1.0)

    # ---- obstacles: occupied cells, merged across ToF gaps, plus the fence edges as walls
    occ_u8 = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ_u8, connectivity=8)
    xmin, ymin, xmax, ymax = fence
    obstacles: list[ScanObstacle] = []
    comps: list[np.ndarray] = []
    for i in range(1, n):
        pts = _component_points(labels, i, grid, mask=occ)
        if len(pts) < 2:
            continue
        lo_xy, hi_xy = pts.min(axis=0) - res / 2, pts.max(axis=0) + res / 2
        dx, dy = hi_xy - lo_xy
        area = float(np.sum((labels == i) & occ)) * res * res
        touches_edge = (lo_xy[0] - xmin < 0.3 or xmax - hi_xy[0] < 0.3 or lo_xy[1] - ymin < 0.3 or ymax - hi_xy[1] < 0.3)
        if max(dx, dy) > 1.2 and max(dx, dy) > 2.5 * min(dx, dy):
            kind = "wall"
        elif touches_edge and max(dx, dy) > 0.6:
            kind = "wall"  # attached to the room's boundary: a wall section, however much of it we saw
        elif area > 0.4 or max(dx, dy) > 1.0:
            kind = "large obstacle"
        else:
            kind = "small obstacle"
        obstacles.append(ScanObstacle(len(obstacles) + 1, kind, [round(float(v), 2) for v in (lo_xy + hi_xy) / 2],
                                      [round(float(dx), 2), round(float(dy), 2)]))
        comps.append(pts)
    # The geofence edges act as walls (in a room they are the walls).
    edge_pts = []
    for (x0, y0), (x1, y1) in (((xmin, ymin), (xmax, ymin)), ((xmax, ymin), (xmax, ymax)),
                               ((xmax, ymax), (xmin, ymax)), ((xmin, ymax), (xmin, ymin))):
        k = max(2, int(math.hypot(x1 - x0, y1 - y0) / res))
        edge_pts.append(np.stack([np.linspace(x0, x1, k), np.linspace(y0, y1, k)], axis=1))
    all_comps = comps + edge_pts
    n_obs = len(comps)
    structural = [o.kind != "small obstacle" for o in obstacles] + [True] * len(edge_pts)  # walls, furniture, fence
    kinds = [o.kind for o in obstacles] + ["fence"] * len(edge_pts)
    # A doorway: a gap >= 0.5 m (nobody calls a 30 cm slot a door) with a wall or room edge on at least one side.
    # Only sizeable unknown regions count as "unexplored"; small unseen pockets behind furniture don't.
    m_u, lab_u, st_u, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), connectivity=4)
    big_unknown = np.zeros_like(unknown)
    for q in range(1, m_u):
        if st_u[q, cv2.CC_STAT_AREA] * res * res >= 1.0:
            big_unknown |= lab_u == q
    frontier_mask = free & cv2.dilate(big_unknown.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # ---- passages: closest gaps between pairs of obstacles / walls
    free_u8 = free.astype(np.uint8)
    n_free, _ = cv2.connectedComponents(free_u8, connectivity=4)
    min_region = int(1.0 / (res * res))  # a "room" side must be at least ~1 m^2

    def splits(a: np.ndarray, b: np.ndarray) -> tuple[bool, list[int]]:
        """Would closing the gap a-b split the free space? Returns (split, sizes of the two sides)."""
        test = free_u8.copy()
        ra, ca = grid.to_cell(a)
        rb, cb = grid.to_cell(b)
        cv2.line(test, (int(ca[0]), int(ra[0])), (int(cb[0]), int(rb[0])), 0, 2)
        m, lab, st, _ = cv2.connectedComponentsWithStats(test, connectivity=4)
        sizes = sorted((int(st[j, cv2.CC_STAT_AREA]) for j in range(1, m)), reverse=True)
        big = [s for s in sizes if s >= min_region]
        return len(big) >= 2 and m - 1 > n_free - 1, big[:2]

    passages: list[ScanPassage] = []
    for i in range(len(all_comps)):
        for j in range(i + 1, len(all_comps)):
            if i >= n_obs and j >= n_obs:
                continue  # two fence edges
            A, B = all_comps[i], all_comps[j]
            d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            k = int(np.argmin(d))
            ia, ib = divmod(k, d.shape[1])
            min_width = float(d[ia, ib]) - res  # cell centres -> free width
            if min_width < res / 2 or min_width > max_door:
                continue
            a, b = A[ia], B[ib]
            mid = (a + b) / 2
            u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-6)
            nrm = np.array([-u[1], u[0]])
            # Narrowest cross-section near the gap = the opening itself (not a line beside the wall).
            width = max(min_width, min(_free_span(lo, grid, mid + o * nrm, u, max_door) for o in np.linspace(-0.3, 0.3, 7)))
            r, c = grid.to_cell(mid)
            if not (0 <= r[0] < grid.ny and 0 <= c[0] < grid.nx) or not free[r[0], c[0]]:
                continue  # the gap itself has not been seen as free
            split, sides = splits(a, b)
            # Passable if even the conservative width fits, or the estimate fits with a 1.5x margin.
            passable = min_width >= drone_width or width >= 1.5 * drone_width
            wallish = kinds[i] in ("wall", "fence") or kinds[j] in ("wall", "fence")
            if split and structural[i] and structural[j] and wallish and width >= 0.5:
                # A doorway joins two areas between walls or furniture. If one side still has
                # unexplored edges, it leads somewhere we haven't been: an exit.
                test = free_u8.copy()
                ra, ca = grid.to_cell(a)
                rb, cb = grid.to_cell(b)
                cv2.line(test, (int(ca[0]), int(ra[0])), (int(cb[0]), int(rb[0])), 0, 2)
                m, lab = cv2.connectedComponents(test, connectivity=4)
                sizes = sorted(((int(np.sum(lab == q)), q) for q in range(1, m)), reverse=True)[:2]
                leads_out = any(np.sum((lab == q) & frontier_mask) >= max(8, 0.05 * n) for n, q in sizes)
                kind = "exit" if leads_out else "doorway"
            elif structural[i] and structural[j] and wallish and width >= 0.5 and _opens_into_unknown(free, unknown, grid, mid, nrm):
                kind = "exit"  # a doorway with unexplored space right behind it
            else:
                kind = "tight gap"
                if passable and width > 2.0 * drone_width:
                    continue  # plenty of room: not worth reporting
            passages.append(ScanPassage(0, kind, [round(float(v), 2) for v in a], [round(float(v), 2) for v in b],
                                        round(width, 2), passable, [round(float(v), 2) for v in mid], round(min_width, 2)))
    # Keep the best (narrowest) report per location.
    passages.sort(key=lambda p: (p.kind == "tight gap", p.width))
    kept: list[ScanPassage] = []
    for p in passages:
        if all(np.linalg.norm(np.array(p.center) - np.array(q.center)) > 0.6 for q in kept):
            kept.append(p)
    for k, p in enumerate(kept, 1):
        p.id = k

    # ---- low obstacles (seen by the downward sensor while flying over them)
    lows: list[ScanLowObstacle] = []
    if low_cells:
        mask = np.zeros(lo.shape, np.uint8)
        hmap = np.zeros(lo.shape, np.float32)
        for (r, c), h in low_cells.items():
            if 0 <= r < grid.ny and 0 <= c < grid.nx:
                mask[r, c] = 1
                hmap[r, c] = max(hmap[r, c], h)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for q in range(1, m):
            if st[q, cv2.CC_STAT_AREA] < 2:
                continue
            pts = _component_points(lab, q, grid)
            lo_xy, hi_xy = pts.min(axis=0) - res / 2, pts.max(axis=0) + res / 2
            h = float(hmap[lab == q].max())
            lows.append(ScanLowObstacle([round(float(v), 2) for v in (lo_xy + hi_xy) / 2],
                                        [round(float(v), 2) for v in hi_xy - lo_xy], round(h, 2), round(h + 0.3, 2)))

    # ---- vision landmarks -> label the obstacles they sit on
    lms = landmarks or []
    for lm in lms:
        p = np.array(lm.position)
        best, dist = None, 0.6
        for o, pts in zip(obstacles, comps):
            dd = float(np.min(np.linalg.norm(pts - p, axis=1)))
            if dd < dist:
                best, dist = o, dd
        if best is not None and lm.label not in best.labels:
            best.labels.append(lm.label)

    rec = max((lo_.min_safe_altitude for lo_ in lows), default=None)
    return RoomScan(round(mapped, 1), round(explored, 3), obstacles, kept, lows, lms, frontier_count(grid), rec)


def triangulate_landmarks(rays: list, fence, min_inliers: int = 4, min_baseline: float = 0.4, seed: int = 0) -> list[ScanLandmark]:
    """Locate what the camera saw by intersecting its viewing directions from different places.

    ``rays`` are ``(label, drone_xy, bearing_rad, confidence)``. Like a surveyor taking bearings
    from two spots, any two rays from viewpoints >= ``min_baseline`` apart pin down a point. A
    consistent point is accepted once enough rays from separate viewpoints agree (RANSAC). It
    needs no range sensor, so it works beyond the ToF range and isn't fooled by a door frame
    standing between the drone and the object.
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = fence
    out: list[ScanLandmark] = []
    for label in sorted({r[0] for r in rays}):
        R = [r for r in rays if r[0] == label]
        if len(R) > 400:
            R = [R[i] for i in np.linspace(0, len(R) - 1, 400).astype(int)]
        O = np.array([r[1] for r in R], float)
        D = np.array([[math.cos(r[2]), math.sin(r[2])] for r in R])
        C = np.array([r[3] for r in R], float)
        alive = np.ones(len(R), bool)
        for _ in range(8):
            idx = np.nonzero(alive)[0]
            if len(idx) < min_inliers:
                break
            best_p, best_in = None, None
            for _ in range(300):
                i, j = rng.choice(idx, 2, replace=False)
                if np.linalg.norm(O[i] - O[j]) < min_baseline:
                    continue
                m = np.stack([D[i], -D[j]], axis=1)
                if abs(np.linalg.det(m)) < math.sin(math.radians(12)):
                    continue  # nearly parallel views: no depth information
                ti, tj = np.linalg.solve(m, O[j] - O[i])
                if ti < 0.2 or tj < 0.2:
                    continue
                p = O[i] + ti * D[i]
                if not (xmin - 0.3 <= p[0] <= xmax + 0.3 and ymin - 0.3 <= p[1] <= ymax + 0.3):
                    continue
                rel = p[None, :] - O[idx]
                along = np.sum(rel * D[idx], axis=1)
                perp = np.abs(rel[:, 0] * D[idx, 1] - rel[:, 1] * D[idx, 0])
                inl = idx[(along > 0.1) & (perp < 0.15 + 0.04 * along)]
                if best_in is None or len(inl) > len(best_in):
                    best_p, best_in = p, inl
            if best_in is None or len(best_in) < min_inliers:
                break
            views = O[best_in]
            spread = float(np.max(np.linalg.norm(views[:, None] - views[None, :], axis=2)))
            if spread < min_baseline:
                alive[best_in] = False
                continue
            # Least-squares point closest to all inlier rays.
            a_sum, b_sum = np.zeros((2, 2)), np.zeros(2)
            for k in best_in:
                pm = np.eye(2) - np.outer(D[k], D[k])
                a_sum += C[k] * pm
                b_sum += C[k] * pm @ O[k]
            try:
                p = np.linalg.solve(a_sum, b_sum)
            except np.linalg.LinAlgError:
                p = best_p
            out.append(ScanLandmark(label, [round(float(v), 2) for v in p], round(float(C[best_in].max()), 2), int(len(best_in))))
            alive[best_in] = False
    return out


# --------------------------------------------------------------------------- output
def render(grid: OccupancyGrid, scan: RoomScan, fence, px_per_m: float = 60.0, places: dict | None = None) -> np.ndarray:
    xmin, ymin, xmax, ymax = fence
    pad = 0.6
    scale = min(px_per_m, 1100.0 / max(xmax - xmin + 2 * pad, 1e-6))
    w, h = int((xmax - xmin + 2 * pad) * scale), int((ymax - ymin + 2 * pad) * scale) + 100
    img = np.full((h, w, 3), (28, 24, 20), np.uint8)

    def px(x, y):
        return int((x - xmin + pad) * scale), int((ymax + pad - y) * scale)

    lo = grid.log_odds
    for r, c in zip(*np.nonzero(lo < FREE)):
        x, y = grid.to_world(r, c)
        cv2.rectangle(img, px(x - grid.res / 2, y + grid.res / 2), px(x + grid.res / 2, y - grid.res / 2), (58, 50, 40), -1)
    for r, c in zip(*np.nonzero(lo > OCC)):
        x, y = grid.to_world(r, c)
        cv2.rectangle(img, px(x - grid.res / 2, y + grid.res / 2), px(x + grid.res / 2, y - grid.res / 2), (210, 205, 190), -1)
    for lo_ in scan.low_obstacles:
        (cx, cy), (sx, sy) = lo_.center, lo_.size
        p0, p1 = px(cx - sx / 2, cy + sy / 2), px(cx + sx / 2, cy - sy / 2)
        overlay = img.copy()
        cv2.rectangle(overlay, p0, p1, (0, 140, 255), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.rectangle(img, p0, p1, (0, 160, 255), 1)
        cv2.putText(img, f"low {lo_.height:.2f}m", (p0[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 190, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, px(xmin, ymax), px(xmax, ymin), (80, 80, 200), 1)
    for o in scan.obstacles:
        (cx, cy), (sx, sy) = o.center, o.size
        col = {"wall": (170, 170, 170), "large obstacle": (90, 200, 255), "small obstacle": (120, 255, 160)}[o.kind]
        cv2.rectangle(img, px(cx - sx / 2, cy + sy / 2), px(cx + sx / 2, cy - sy / 2), col, 1)
        tag = f"#{o.id} {o.kind}" + (f" ({', '.join(l.replace('_', ' ') for l in o.labels)})" if o.labels else "")
        cv2.putText(img, tag, px(cx - sx / 2, cy + sy / 2 + 0.08), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    for p in scan.passages:
        col = (90, 255, 120) if p.passable else (80, 80, 255)
        if p.kind == "exit":
            col = (255, 200, 60) if p.passable else (80, 80, 255)
        cv2.line(img, px(*p.a), px(*p.b), col, 3, cv2.LINE_AA)
        mx, my = px(*p.center)
        cv2.putText(img, f"{p.kind} {p.width:.1f}m", (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for lm in scan.landmarks:
        cv2.drawMarker(img, px(*lm.position), (255, 120, 255), cv2.MARKER_DIAMOND, 12, 2)
        cv2.putText(img, lm.label.replace("_", " "), (px(*lm.position)[0] + 7, px(*lm.position)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 150, 255), 1, cv2.LINE_AA)
    for name, (x, y) in (places or {}).items():
        cv2.circle(img, px(x, y), 3, (120, 200, 255), -1)
    y0 = h - 88
    cv2.putText(img, "ROOM SCAN", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 120), 1, cv2.LINE_AA)
    legend = [((210, 205, 190), "mapped obstacle"), ((58, 50, 40), "mapped free"), ((90, 255, 120), "doorway"),
              ((255, 200, 60), "exit to unexplored"), ((80, 80, 255), "too narrow"), ((0, 160, 255), "low obstacle"),
              ((255, 120, 255), "seen by camera")]
    x, y = 10, y0 + 14
    for col, txt in legend:
        item_w = 22 + 7 * len(txt)
        if x + item_w > w - 6:  # wrap onto the next line
            x, y = 10, y + 16
        cv2.rectangle(img, (x, y), (x + 12, y + 11), col, -1)
        cv2.putText(img, txt, (x + 16, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        x += item_w
    cv2.putText(img, f"explored {scan.explored_fraction * 100:.0f}%  |  {scan.frontier_cells} frontier cells left",
                (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def save(path: str | Path, grid: OccupancyGrid, scan: RoomScan, fence) -> tuple[Path, Path]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    npz = path / "room_map.npz"
    np.savez_compressed(npz, log_odds=grid.log_odds, x0=grid.x0, y0=grid.y0, res=grid.res, fence=np.array(fence, float))
    js = path / "room_scan.json"
    js.write_text(json.dumps(scan.to_dict(), indent=2), encoding="utf-8")
    cv2.imwrite(str(path / "room_scan.png"), render(grid, scan, fence))
    return npz, js


def load_prior(grid: OccupancyGrid, npz_path: str | Path) -> bool:
    """Seed a planner grid with a saved room map (same geofence and resolution)."""
    d = np.load(npz_path)
    if d["log_odds"].shape != grid.log_odds.shape or abs(float(d["res"]) - grid.res) > 1e-9:
        return False
    # Keep the evidence but soften it, so a moved chair can still be re-mapped in flight.
    grid.log_odds[:] = np.clip(d["log_odds"], -1.2, 1.5)
    return True
