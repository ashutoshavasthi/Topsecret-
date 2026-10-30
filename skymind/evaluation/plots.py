"""Dependency-free (OpenCV only) plots for the judge's report."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from skymind.config import SkyMindConfig

MODE_COLORS = {
    "takeoff": (200, 120, 0),
    "mission": (40, 160, 40),
    "rtl": (0, 140, 255),
    "land": (180, 0, 180),
}


def flight_map(records: list[dict], cfg: SkyMindConfig, path: str | Path, px_per_m: float = 18.0) -> Path:
    states = [r for r in records if r.get("type") == "state"]
    xmin, ymin, xmax, ymax = cfg.mission.geofence
    pad = 1.0
    w = int((xmax - xmin + 2 * pad) * px_per_m)
    h = int((ymax - ymin + 2 * pad) * px_per_m)
    img = np.full((h, w, 3), 250, np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return int((x - xmin + pad) * px_per_m), int((ymax + pad - y) * px_per_m)

    for gx in range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1, 5):
        cv2.line(img, px(gx, ymin), px(gx, ymax), (230, 230, 230), 1)
    for gy in range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1, 5):
        cv2.line(img, px(xmin, gy), px(xmax, gy), (230, 230, 230), 1)
    cv2.rectangle(img, px(xmin, ymax), px(xmax, ymin), (0, 0, 200), 1)

    for x, y, r, _h in cfg.world.cylinders:
        cv2.circle(img, px(x, y), max(2, int(r * px_per_m)), (60, 140, 60), -1)
    for bx0, by0, bx1, by1, _h in cfg.world.boxes:
        cv2.rectangle(img, px(bx0, by1), px(bx1, by0), (140, 140, 140), -1)
    for x, y, _z, r, label in cfg.world.targets:
        col = (30, 30, 220) if "red" in label else (20, 200, 240)
        cv2.circle(img, px(x, y), max(3, int(r * px_per_m)), col, -1)
        cv2.circle(img, px(x, y), max(3, int(r * px_per_m)), (0, 0, 0), 1)

    pts = [(s["pos"][0], s["pos"][1], s["mode"]) for s in states]
    for (x0, y0, m), (x1, y1, _) in zip(pts[:-1], pts[1:]):
        cv2.line(img, px(x0, y0), px(x1, y1), MODE_COLORS.get(m, (0, 0, 0)), 2, cv2.LINE_AA)

    hx, hy = cfg.mission.home
    cv2.drawMarker(img, px(hx, hy), (0, 0, 0), cv2.MARKER_SQUARE, 14, 2)
    for i, (x, y) in enumerate(cfg.mission.waypoints):
        cv2.drawMarker(img, px(x, y), (0, 0, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, f"WP{i}", (px(x, y)[0] + 8, px(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    y = 18
    for m, c in MODE_COLORS.items():
        cv2.line(img, (10, y), (30, y), c, 3)
        cv2.putText(img, m, (36, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        y += 16

    out = Path(path)
    cv2.imwrite(str(out), img)
    return out


def timeline(records: list[dict], path: str | Path, width: int = 900) -> Path:
    """Altitude, speed and tilt over time — a quick read on how balanced the flight was."""
    states = [r for r in records if r.get("type") == "state"]
    out = Path(path)
    if not states:
        cv2.imwrite(str(out), np.full((60, width, 3), 255, np.uint8))
        return out
    t = np.array([s["t"] for s in states])
    series = [
        ("altitude [m]", np.array([s["pos"][2] for s in states]), (200, 80, 0)),
        ("speed [m/s]", np.array([np.linalg.norm(s["vel"]) for s in states]), (40, 150, 40)),
        ("tilt [deg]", np.degrees([np.arccos(np.clip(np.cos(s["rpy"][0]) * np.cos(s["rpy"][1]), -1, 1)) for s in states]), (0, 0, 200)),
        ("lidar min [m]", np.array([s["min_range"] for s in states]), (150, 0, 150)),
    ]
    ph, margin = 110, 50
    img = np.full((ph * len(series) + 20, width, 3), 255, np.uint8)
    t0, t1 = t[0], max(t[-1], t[0] + 1e-3)
    for i, (name, v, col) in enumerate(series):
        top = 10 + i * ph
        lo, hi = float(np.min(v)), float(np.max(v))
        hi = hi if hi > lo else lo + 1.0
        cv2.rectangle(img, (margin, top), (width - 10, top + ph - 20), (210, 210, 210), 1)
        xs = margin + (t - t0) / (t1 - t0) * (width - 10 - margin)
        ys = top + ph - 20 - (v - lo) / (hi - lo) * (ph - 24)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{name}  {lo:.1f}..{hi:.1f}", (margin + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(img, f"{t0:.0f}s", (margin, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, f"{t1:.0f}s", (width - 50, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(out), img)
    return out
