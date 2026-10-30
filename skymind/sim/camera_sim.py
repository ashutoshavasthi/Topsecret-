"""Synthetic forward-looking camera.

A column ray-caster (one ray per image column) renders obstacles as shaded
vertical slabs over a sky/ground background and draws targets as coloured
discs. It is cheap enough to run in real time in pure numpy, and it also
returns ground-truth bounding boxes so the vision pipeline can be scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from skymind.config import CameraConfig
from skymind.sim.world import TARGET_COLORS_BGR, World


@dataclass
class GroundTruthBox:
    label: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    distance: float


class SimCamera:
    def __init__(self, cfg: CameraConfig, world: World, rng: np.random.Generator):
        self.cfg, self.world, self.rng = cfg, world, rng
        self.w, self.h = cfg.width, cfg.height
        self.f = (self.w / 2.0) / math.tan(math.radians(cfg.fov_deg) / 2.0)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        u = np.arange(self.w) + 0.5
        self.col_rel = np.arctan2(self.cx - u, self.f)  # +left
        rows = np.arange(self.h)[:, None].astype(float)
        self.rows = rows
        sky_top, sky_bot = np.array([235, 190, 140]), np.array([250, 230, 205])
        t = np.clip(rows / self.cy, 0, 1)
        self.sky = (sky_top * (1 - t) + sky_bot * t)[:, None, :].repeat(self.w, axis=1)

    def render(
        self, position: np.ndarray, yaw: float, roll: float = 0.0, pitch: float = 0.0, body_rate: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[GroundTruthBox]]:
        cfg = self.cfg
        if cfg.gimbal_stabilized:
            roll = pitch = 0.0
        cam_z = max(float(position[2]), 0.05)
        horizon = self.cy - self.f * math.tan(pitch)  # nose-down pitch is positive -> horizon moves up

        img = self.sky.copy()
        # Ground with distance-based shading.
        below = self.rows[:, 0] > horizon + 0.5
        gd = np.where(below, cam_z * self.f / np.maximum(self.rows[:, 0] - horizon, 1e-3), 0.0)
        shade = np.clip(1.0 - gd / 60.0, 0.45, 1.0)
        ground = np.array([70, 125, 95])[None, :] * shade[:, None]
        img[below] = ground[below][:, None, :]

        hits = self.world.raycast(position, yaw + self.col_rel, 60.0, respect_height=False)
        depth = hits.distance * np.cos(self.col_rel)
        has = hits.obstacle >= 0
        safe_depth = np.maximum(depth, 0.05)
        top = horizon - self.f * (hits.height - cam_z) / safe_depth
        bot = horizon + self.f * cam_z / safe_depth
        mask = has[None, :] & (self.rows >= top[None, :]) & (self.rows <= bot[None, :])
        if self.world.num_obstacles:
            col = self.world.obstacle_colors[np.maximum(hits.obstacle, 0)]
            col = col * np.clip(1.15 - depth / 40.0, 0.35, 1.1)[:, None]
            # vertical texture so the edge-based sharpness metric has something to see
            stripes = 0.9 + 0.1 * np.sin(self.rows * 0.9)
            colour = col[None, :, :] * stripes[:, :, None]
            img = np.where(mask[:, :, None], colour, img)

        img = np.clip(img, 0, 255).astype(np.uint8)
        boxes = self._draw_targets(img, position, yaw, horizon, cam_z, np.where(has, depth, np.inf))

        if roll:
            m = cv2.getRotationMatrix2D((self.cx, self.cy), math.degrees(roll), 1.0)
            img = cv2.warpAffine(img, m, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        if body_rate is not None and cfg.motion_blur_gain > 0:
            residual = np.linalg.norm(body_rate) * (0.15 if cfg.gimbal_stabilized else 1.0)
            k = int(round(residual * cfg.motion_blur_gain))
            if k >= 2:
                kernel = np.zeros((k, k), np.float32)
                kernel[k // 2, :] = 1.0 / k
                img = cv2.filter2D(img, -1, kernel)
        noise = self.rng.normal(0.0, 3.0, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        return img, boxes

    def _draw_targets(self, img, position, yaw, horizon, cam_z, col_depth) -> list[GroundTruthBox]:
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        visible = []
        for tgt in self.world.targets:
            rel = tgt.position - position
            x_f = float(rel[:2] @ fwd)
            if x_f < 0.3:
                continue
            y_l = float(rel[:2] @ left)
            u = self.cx - self.f * y_l / x_f
            v = horizon - self.f * (tgt.position[2] - cam_z) / x_f
            r = self.f * tgt.radius / x_f
            ui = int(round(u))
            if not (-r <= u < self.w + r and -r <= v < self.h + r):
                continue
            if 0 <= ui < self.w and col_depth[ui] < x_f - tgt.radius:
                continue  # hidden behind an obstacle
            colour = TARGET_COLORS_BGR.get(tgt.label, (200, 40, 200))
            cv2.circle(img, (ui, int(round(v))), max(1, int(round(r))), colour, -1, lineType=cv2.LINE_AA)
            x0, y0 = max(0, int(u - r)), max(0, int(v - r))
            x1, y1 = min(self.w, int(u + r) + 1), min(self.h, int(v + r) + 1)
            if x1 > x0 and y1 > y0:
                visible.append(GroundTruthBox(tgt.label, (x0, y0, x1 - x0, y1 - y0), float(np.linalg.norm(rel))))
        return visible
