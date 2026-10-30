"""Quad-X motor allocation and desaturating mixer.

Motor layout (top view, FLU body frame, x forward):

        3 (front-left, CCW)    0 (front-right, CW)
                        \\  /
                         \\/
                         /\\
                        /  \\
        2 (back-left, CW)      1 (back-right, CCW)
"""

from __future__ import annotations

import numpy as np

MOTOR_ANGLES = np.radians([-45.0, -135.0, 135.0, 45.0])
SPIN = np.array([1.0, -1.0, 1.0, -1.0])  # +1 produces +z reaction torque


def allocation_matrix(arm_length: float, yaw_coeff: float) -> np.ndarray:
    """Matrix mapping motor thrusts [N] -> [collective, tau_x, tau_y, tau_z]."""
    x = arm_length * np.cos(MOTOR_ANGLES)
    y = arm_length * np.sin(MOTOR_ANGLES)
    return np.vstack([np.ones(4), y, -x, SPIN * yaw_coeff])


class Mixer:
    def __init__(self, arm_length: float, yaw_coeff: float, max_thrust: float):
        self.alloc = allocation_matrix(arm_length, yaw_coeff)
        self.inv = np.linalg.inv(self.alloc)
        self.max_thrust = max_thrust

    def mix(self, collective: float, torque: np.ndarray) -> np.ndarray:
        """Return per-motor thrusts, prioritising roll/pitch over collective and yaw.

        This is what keeps the vehicle level when the motors are near their
        limits: collective thrust is shifted first, then yaw authority is
        dropped, and only as a last resort roll/pitch torque is scaled down.
        """
        tmax = self.max_thrust
        per_motor_c = self.inv[:, 0]  # equals 0.25 for every motor
        rp = self.inv[:, 1:3] @ torque[:2]
        yaw = self.inv[:, 3] * torque[2]

        def fit(diff: np.ndarray) -> tuple[np.ndarray, bool]:
            span = diff.max() - diff.min()
            if span > tmax:
                return diff, False
            c = collective * per_motor_c[0]
            lo, hi = -diff.min(), tmax - diff.max()
            return diff + min(max(c, lo), hi), True

        motors, ok = fit(rp + yaw)
        if not ok:
            motors, ok = fit(rp)
        if not ok:
            span = rp.max() - rp.min()
            scale = tmax / span if span > 0 else 0.0
            motors, _ = fit(rp * scale * 0.999)
        return np.clip(motors, 0.0, tmax)

    def forces(self, motors: np.ndarray) -> np.ndarray:
        return self.alloc @ motors


def hover_thrust_per_motor(mass: float, g: float = 9.81) -> float:
    return mass * g / 4.0


def thrust_to_weight(mass: float, max_thrust: float, g: float = 9.81) -> float:
    return 4.0 * max_thrust / (mass * g)
