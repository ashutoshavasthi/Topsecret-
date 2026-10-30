"""ENU (SkyMind) <-> NED (MAVLink / ArduPilot / PX4) conversions."""

from __future__ import annotations

import math

import numpy as np

from skymind.math3d import wrap_angle


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]])


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]])


def yaw_enu_to_ned(yaw: float) -> float:
    """ENU yaw is CCW from east; NED yaw is CW from north."""
    return float(wrap_angle(math.pi / 2 - yaw))


def yaw_ned_to_enu(yaw: float) -> float:
    return float(wrap_angle(math.pi / 2 - yaw))


def attitude_ned_to_enu(roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    """FRD body/NED world Euler -> FLU body/ENU world Euler."""
    return roll, -pitch, yaw_ned_to_enu(yaw)
