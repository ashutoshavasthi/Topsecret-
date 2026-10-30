"""Reactive obstacle avoidance on 360-degree range data.

For every candidate heading we compute the *corridor clearance*: how far a
disc of radius ``safety_radius`` could travel in that direction before
touching any lidar return. This is the Vector Field Histogram idea, but
checking the drone's actual swept width instead of a density threshold, so
narrow gaps that the airframe cannot fit through are rejected.

A heading is then chosen to minimise

    goal_weight * |heading - goal| + smooth_weight * |heading - previous|

among headings with enough clearance, and speed is limited so the drone can
always stop before ``stop_distance``. A short-range repulsive term pushes the
vehicle away if it ever ends up inside the safety radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from skymind.config import AvoidanceConfig
from skymind.math3d import wrap_angle


class AvoidMode(str, Enum):
    CLEAR = "clear"  # flying straight at the goal
    DETOUR = "detour"  # steering around an obstacle
    BLOCKED = "blocked"  # no safe heading, holding position


@dataclass
class AvoidanceCommand:
    velocity_xy: np.ndarray
    heading: float
    mode: AvoidMode
    min_range: float
    clearance_ahead: float


class ObstacleAvoider:
    def __init__(self, cfg: AvoidanceConfig, max_range: float):
        self.cfg = cfg
        self.max_range = max_range
        self.headings = np.linspace(-np.pi, np.pi, cfg.num_headings, endpoint=False)
        self.prev_heading: float | None = None

    def reset(self) -> None:
        self.prev_heading = None

    def corridor_clearance(self, ranges: np.ndarray, angles: np.ndarray, headings: np.ndarray | None = None) -> np.ndarray:
        headings = self.headings if headings is None else headings
        r = self.cfg.safety_radius
        valid = ranges < self.max_range * 0.995  # anything at max range means 'no return'
        if not np.any(valid):
            return np.full(len(headings), self.max_range)
        pts = np.stack([ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid])], axis=1)
        u = np.stack([np.cos(headings), np.sin(headings)], axis=1)
        along = u @ pts.T  # (H, P)
        lateral = np.abs(u[:, 0:1] * pts[None, :, 1] - u[:, 1:2] * pts[None, :, 0])
        blocking = (lateral < r) & (along > -r)
        hit = along - np.sqrt(np.maximum(r * r - lateral * lateral, 0.0))
        hit = np.where(blocking, np.maximum(hit, 0.0), self.max_range)
        return hit.min(axis=1)

    def plan(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        to_goal: np.ndarray,
        cruise_speed: float,
        observed: np.ndarray | None = None,
    ) -> AvoidanceCommand:
        """``observed`` masks ``self.headings`` that the sensors can actually see. Unobserved
        headings are never chosen: with sparse sensors an empty direction is unknown, not free."""
        cfg = self.cfg
        dist_goal = float(np.linalg.norm(to_goal))
        goal_heading = math.atan2(to_goal[1], to_goal[0]) if dist_goal > 1e-6 else (self.prev_heading or 0.0)
        min_range = float(ranges.min()) if len(ranges) else self.max_range

        clear = self.corridor_clearance(ranges, angles)
        goal_clear = float(self.corridor_clearance(ranges, angles, np.array([goal_heading]))[0])
        if observed is not None:
            clear = np.where(observed, clear, 0.0)
            gi = int(np.argmin(np.abs(wrap_angle(self.headings - goal_heading))))
            if not observed[gi]:
                goal_clear = 0.0
        need = min(cfg.lookahead, dist_goal + cfg.safety_radius)

        if goal_clear >= need:
            heading, mode, heading_clear = goal_heading, AvoidMode.CLEAR, goal_clear
        else:
            ok = clear >= max(need, cfg.stop_distance + 0.5)
            if not np.any(ok):
                ok = clear >= cfg.stop_distance + 0.5
            if np.any(ok):
                cost = cfg.goal_weight * np.abs(wrap_angle(self.headings - goal_heading))
                if self.prev_heading is not None:
                    cost = cost + cfg.smooth_weight * np.abs(wrap_angle(self.headings - self.prev_heading))
                cost = np.where(ok, cost, np.inf)
                i = int(np.argmin(cost))
                heading, mode, heading_clear = float(self.headings[i]), AvoidMode.DETOUR, float(clear[i])
            else:
                i = int(np.argmax(clear))
                heading, mode, heading_clear = float(self.headings[i]), AvoidMode.BLOCKED, float(clear[i])

        # Speed so that we can always brake before stop_distance, and slow near the goal.
        room = max(heading_clear - cfg.stop_distance, 0.0)
        speed = min(cruise_speed, math.sqrt(2.0 * cfg.max_decel * room), 1.2 * dist_goal + 0.1)
        if mode is AvoidMode.BLOCKED:
            speed = min(speed, 0.3)
        vel = speed * np.array([math.cos(heading), math.sin(heading)])

        # Repulsion if something is already inside the safety radius: one push away from the
        # weighted mean obstacle direction (not a sum over beams, which explodes next to walls).
        zone = cfg.safety_radius + 0.3
        close = ranges < zone
        if np.any(close):
            weight = (zone - ranges[close]) / zone
            away = -(np.stack([np.cos(angles[close]), np.sin(angles[close])], axis=1) * weight[:, None]).sum(axis=0)
            n = float(np.linalg.norm(away))
            if n > 1e-6:
                vel = vel + cfg.repulse_gain * float(weight.max()) * away / n
        n = float(np.linalg.norm(vel))
        if n > cruise_speed:
            vel = vel * (cruise_speed / n)

        self.prev_heading = heading
        return AvoidanceCommand(vel, heading, mode, min_range, goal_clear)
