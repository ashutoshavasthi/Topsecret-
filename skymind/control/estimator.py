"""State estimation: Mahony attitude filter + Kalman position/velocity filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skymind.config import EstimatorConfig
from skymind.math3d import E3, quat_integrate, quat_to_euler, quat_to_rot, wrap_angle

GRAVITY = 9.81


class MahonyAHRS:
    """Complementary attitude filter (Mahony et al., 2008).

    The gyro is integrated for fast response; the accelerometer's gravity
    direction and a heading reference slowly pull the estimate back so it
    does not drift. The integral term learns the gyro bias online.
    """

    def __init__(self, kp: float = 1.5, ki: float = 0.05, yaw_gain: float = 0.5):
        self.kp, self.ki, self.yaw_gain = kp, ki, yaw_gain
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias_integral = np.zeros(3)
        self.corrected_rate = np.zeros(3)

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        heading: float | None = None,
        linear_accel_world: np.ndarray | None = None,
    ) -> np.ndarray:
        """``linear_accel_world`` (from the velocity filter) removes manoeuvre
        acceleration from the reference. Without it a multirotor accelerometer
        mostly measures thrust along body z and makes the filter think it is level."""
        rot = quat_to_rot(self.q)
        error = np.zeros(3)
        norm = np.linalg.norm(accel)
        ref = GRAVITY * E3 if linear_accel_world is None else linear_accel_world + GRAVITY * E3
        ref_norm = np.linalg.norm(ref)
        if norm > 1e-6 and ref_norm > 1e-6 and abs(norm - ref_norm) < 0.25 * GRAVITY:
            up_meas = accel / norm
            up_est = rot.T @ (ref / ref_norm)
            error += np.cross(up_meas, up_est)
        if heading is not None:
            yaw_err = float(wrap_angle(heading - quat_to_euler(self.q)[2]))
            error += self.yaw_gain * yaw_err * (rot.T @ E3)
        self.bias_integral += self.ki * error * dt
        omega = gyro + self.kp * error + self.bias_integral
        self.q = quat_integrate(self.q, omega, dt)
        self.corrected_rate = gyro + self.bias_integral
        return self.q


class PositionKF:
    """Per-axis constant-velocity Kalman filter driven by world-frame acceleration."""

    def __init__(self, meas_std: float, accel_std: float):
        self.x = np.zeros((3, 2))  # [axis, (pos, vel)]
        self.p = np.tile(np.diag([1.0, 1.0]), (3, 1, 1))
        self.r = meas_std**2
        self.q_acc = accel_std**2

    def reset(self, position: np.ndarray) -> None:
        self.x[:, 0] = position
        self.x[:, 1] = 0.0
        self.p = np.tile(np.diag([0.01, 0.01]), (3, 1, 1))

    def predict(self, accel_world: np.ndarray, dt: float) -> None:
        f = np.array([[1.0, dt], [0.0, 1.0]])
        b = np.array([0.5 * dt * dt, dt])
        q = self.q_acc * np.outer(b, b)
        for i in range(3):
            self.x[i] = f @ self.x[i] + b * accel_world[i]
            self.p[i] = f @ self.p[i] @ f.T + q

    def correct(self, position: np.ndarray) -> None:
        for i in range(3):
            s = self.p[i, 0, 0] + self.r
            k = self.p[i, :, 0] / s
            self.x[i] = self.x[i] + k * (position[i] - self.x[i, 0])
            self.p[i] = self.p[i] - np.outer(k, self.p[i, 0, :])

    @property
    def position(self) -> np.ndarray:
        return self.x[:, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[:, 1].copy()


@dataclass
class StateEstimate:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    rate: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.quaternion)

    @property
    def euler(self) -> tuple[float, float, float]:
        return quat_to_euler(self.quaternion)


class StateEstimator:
    def __init__(self, cfg: EstimatorConfig):
        self.ahrs = MahonyAHRS(cfg.mahony_kp, cfg.mahony_ki, cfg.yaw_gain)
        self.kf = PositionKF(cfg.pos_meas_std, cfg.accel_process_std)
        self.rate = np.zeros(3)
        self.linear_accel = np.zeros(3)
        self._prev_vel: np.ndarray | None = None
        self._lin_acc_tau = 0.15  # s

    def reset(self, position: np.ndarray, quaternion: np.ndarray | None = None) -> None:
        self.kf.reset(position)
        self._prev_vel = None
        self.linear_accel = np.zeros(3)
        if quaternion is not None:
            self.ahrs.q = quaternion.copy()

    def update_imu(self, gyro: np.ndarray, accel: np.ndarray, dt: float, heading: float | None = None) -> None:
        q = self.ahrs.update(gyro, accel, dt, heading, self.linear_accel)
        self.rate = self.ahrs.corrected_rate
        accel_world = quat_to_rot(q) @ accel - GRAVITY * E3
        self.kf.predict(accel_world, dt)

    def update_position(self, position: np.ndarray, dt: float) -> None:
        self.kf.correct(position)
        vel = self.kf.velocity
        if self._prev_vel is not None and dt > 0:
            raw = (vel - self._prev_vel) / dt
            alpha = dt / (dt + self._lin_acc_tau)
            self.linear_accel = self.linear_accel + alpha * (raw - self.linear_accel)
        self._prev_vel = vel

    def estimate(self) -> StateEstimate:
        return StateEstimate(self.kf.position, self.kf.velocity, self.ahrs.q.copy(), self.rate.copy())
