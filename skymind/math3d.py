"""Small 3D math helpers.

Conventions used throughout SkyMind:
  * World frame is ENU (x = east, y = north, z = up).
  * Body frame is FLU (x = forward, y = left, z = up).
  * Quaternions are stored as numpy arrays ``[w, x, y, z]`` and rotate
    body-frame vectors into the world frame.
  * Euler angles are ZYX (yaw, pitch, roll) and returned as (roll, pitch, yaw).
"""

from __future__ import annotations

import math

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`."""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return q if q[0] >= 0.0 else -q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def rot_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    pitch = -math.asin(max(-1.0, min(1.0, r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    return rot_to_euler(quat_to_rot(q))


def quat_integrate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a body-rate vector over ``dt`` using the exponential map."""
    angle = np.linalg.norm(omega_body) * dt
    if angle < 1e-12:
        return q
    axis = omega_body / np.linalg.norm(omega_body)
    half = angle / 2.0
    dq = np.concatenate(([math.cos(half)], math.sin(half) * axis))
    return quat_normalize(quat_mul(q, dq))


def tilt_angle(r: np.ndarray) -> float:
    """Angle between body z axis and world up, in radians."""
    return math.acos(max(-1.0, min(1.0, r[2, 2])))
