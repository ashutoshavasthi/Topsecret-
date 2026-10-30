"""Cascaded self-balancing flight controller.

    position --P--> velocity --PID--> acceleration --> attitude --P--> body rate --PID--> torque --> mixer

The inner attitude/rate loops are what keep the drone balanced: any tilt
disturbance (gust, payload shift, motor mismatch) produces an attitude error
that is driven back to the commanded attitude within a few hundred ms.
Attitude error uses the geometric SO(3) formulation (Lee et al., 2010) so it
stays well-behaved at large angles, e.g. recovering from a hand-throw launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from skymind.config import ControlConfig, VehicleConfig
from skymind.control.estimator import StateEstimate
from skymind.control.mixer import Mixer
from skymind.control.pid import PID
from skymind.math3d import E3, vee

GRAVITY = 9.81


@dataclass
class Setpoint:
    """Position components that are NaN are velocity-controlled instead."""

    position: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # feed-forward
    yaw: float = 0.0


@dataclass
class ControlOutput:
    motors: np.ndarray
    collective: float
    torque: np.ndarray
    attitude_target: np.ndarray
    rate_target: np.ndarray
    velocity_target: np.ndarray


class FlightController:
    def __init__(self, vehicle: VehicleConfig, cfg: ControlConfig):
        self.v, self.cfg = vehicle, cfg
        self.mass = vehicle.mass
        self.inertia = np.array(vehicle.inertia, dtype=float)
        self.mixer = Mixer(vehicle.arm_length, vehicle.yaw_moment_coeff, vehicle.max_motor_thrust)
        self.vel_pid = PID(cfg.vel_p, cfg.vel_i, cfg.vel_d, output_limit=[6.0, 6.0, 6.0], integral_limit=[2.0, 2.0, 3.0])
        self.rate_pid = PID(cfg.rate_p, cfg.rate_i, cfg.rate_d, output_limit=[80.0, 80.0, 20.0], integral_limit=[5.0, 5.0, 2.0])
        self.max_tilt = math.radians(cfg.max_tilt_deg)
        self.max_rate = np.array(cfg.max_rate)
        # Lowest collective thrust as a fraction of 1 g. A dodge reflex lowers it briefly
        # so the drone can drop almost in free fall, the fastest way out of the way.
        self.min_thrust_frac = 0.3

    def reset(self) -> None:
        self.vel_pid.reset()
        self.rate_pid.reset()

    def velocity_target(self, est: StateEstimate, sp: Setpoint) -> np.ndarray:
        cfg = self.cfg
        pos_err = np.where(np.isnan(sp.position), 0.0, sp.position - est.position)
        vel = sp.velocity + np.array(cfg.pos_gain) * pos_err
        h = np.linalg.norm(vel[:2])
        if h > cfg.max_horizontal_speed:
            vel[:2] *= cfg.max_horizontal_speed / h
        vel[2] = np.clip(vel[2], -cfg.max_vertical_speed, cfg.max_vertical_speed)
        return vel

    def thrust_vector(self, accel_sp: np.ndarray) -> np.ndarray:
        """Desired world-frame thrust, with tilt limited to ``max_tilt``."""
        az = max(accel_sp[2] + GRAVITY, self.min_thrust_frac * GRAVITY)
        a_xy = accel_sp[:2]
        max_xy = az * math.tan(self.max_tilt)
        n = np.linalg.norm(a_xy)
        if n > max_xy:
            a_xy = a_xy * (max_xy / n)
        return self.mass * np.array([a_xy[0], a_xy[1], az])

    @staticmethod
    def attitude_from_thrust(thrust_vec: np.ndarray, yaw: float) -> np.ndarray:
        z_b = thrust_vec / np.linalg.norm(thrust_vec)
        x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        return np.column_stack([x_b, y_b, z_b])

    def attitude_control(self, rot: np.ndarray, rot_des: np.ndarray, collective: float, rate: np.ndarray, dt: float):
        e_r = 0.5 * vee(rot_des.T @ rot - rot.T @ rot_des)
        rate_sp = np.clip(-np.array(self.cfg.att_gain) * e_r, -self.max_rate, self.max_rate)
        ang_acc = self.rate_pid.update(rate_sp, rate, dt)
        torque = self.inertia * ang_acc + np.cross(rate, self.inertia * rate)
        motors = self.mixer.mix(collective, torque)
        return motors, torque, rate_sp

    def update(self, est: StateEstimate, sp: Setpoint, dt: float) -> ControlOutput:
        vel_sp = self.velocity_target(est, sp)
        accel_sp = sp.acceleration + self.vel_pid.update(vel_sp, est.velocity, dt)
        thrust_vec = self.thrust_vector(accel_sp)
        rot = est.rotation
        rot_des = self.attitude_from_thrust(thrust_vec, sp.yaw)
        collective = float(thrust_vec @ (rot @ E3))
        collective = max(collective, 0.0)
        motors, torque, rate_sp = self.attitude_control(rot, rot_des, collective, est.rate, dt)
        return ControlOutput(motors, collective, torque, rot_des, rate_sp, vel_sp)

    def idle(self) -> ControlOutput:
        z = np.zeros(3)
        return ControlOutput(np.zeros(4), 0.0, z, np.eye(3), z, z)
