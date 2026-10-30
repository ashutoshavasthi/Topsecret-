"""Rigid-body quadrotor simulation with motor lag, drag, wind and ground contact."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skymind.config import SensorConfig, VehicleConfig
from skymind.control.mixer import allocation_matrix
from skymind.math3d import E3, quat_from_euler, quat_integrate, quat_to_euler, quat_to_rot

GRAVITY = 9.81


@dataclass
class QuadState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body rates
    motor_thrust: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.quaternion)

    @property
    def euler(self) -> tuple[float, float, float]:
        return quat_to_euler(self.quaternion)


@dataclass
class ImuSample:
    gyro: np.ndarray
    accel: np.ndarray  # specific force in body frame


class QuadrotorSim:
    def __init__(self, vehicle: VehicleConfig, sensors: SensorConfig, rng: np.random.Generator):
        self.v = vehicle
        self.s = sensors
        self.rng = rng
        self.state = QuadState()
        self.inertia = np.array(vehicle.inertia, dtype=float)
        self.alloc = allocation_matrix(vehicle.arm_length, vehicle.yaw_moment_coeff)
        self.gyro_bias = rng.normal(0.0, sensors.gyro_bias_std, 3)
        self.last_accel = np.zeros(3)
        self.on_ground = True
        self.external_torque = np.zeros(3)
        self.external_force = np.zeros(3)

    def reset(self, position=None, euler=None) -> None:
        self.state = QuadState()
        if position is not None:
            self.state.position = np.asarray(position, dtype=float).copy()
        if euler is not None:
            self.state.quaternion = quat_from_euler(*euler)
        self.on_ground = self.state.position[2] <= 1e-3

    def step(self, motor_cmd: np.ndarray, dt: float, wind: np.ndarray | None = None) -> None:
        st, v = self.state, self.v
        cmd = np.clip(motor_cmd, 0.0, v.max_motor_thrust)
        alpha = min(1.0, dt / v.motor_time_constant)
        st.motor_thrust = st.motor_thrust + alpha * (cmd - st.motor_thrust)

        wrench = self.alloc @ st.motor_thrust
        thrust, torque = wrench[0], wrench[1:] + self.external_torque
        rot = st.rotation
        air_vel = st.velocity - (np.zeros(3) if wind is None else wind)
        force = rot @ (thrust * E3) - v.linear_drag * air_vel + self.external_force
        accel = force / v.mass - GRAVITY * E3

        j = self.inertia
        omega_dot = (torque - np.cross(st.omega, j * st.omega)) / j

        st.velocity = st.velocity + accel * dt
        st.position = st.position + st.velocity * dt
        st.omega = st.omega + omega_dot * dt
        st.quaternion = quat_integrate(st.quaternion, st.omega, dt)

        # Ground contact: the ground pushes back, kills horizontal sliding and spin.
        if st.position[2] <= 0.0 and st.velocity[2] <= 0.0:
            st.position[2] = 0.0
            st.velocity[:] = 0.0
            st.omega *= 0.5
            accel = np.zeros(3)
            self.on_ground = True
        else:
            self.on_ground = False
        self.last_accel = accel

    def imu(self) -> ImuSample:
        st = self.state
        gyro = st.omega + self.gyro_bias + self.rng.normal(0.0, self.s.gyro_noise_std, 3)
        specific = st.rotation.T @ (self.last_accel + GRAVITY * E3)
        accel = specific + self.rng.normal(0.0, self.s.accel_noise_std, 3)
        return ImuSample(gyro=gyro, accel=accel)

    def position_measurement(self) -> np.ndarray:
        return self.state.position + self.rng.normal(0.0, self.s.pos_noise_std, 3)

    def heading_measurement(self) -> float:
        return self.state.euler[2] + float(self.rng.normal(0.0, 0.01))

    def total_thrust(self) -> float:
        return float(self.state.motor_thrust.sum())
