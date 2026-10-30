"""Physics-based power, endurance and noise model.

Electrical power per rotor follows momentum theory:

    P = T^1.5 / (FM * eta * sqrt(2 * rho * A))

with T the rotor thrust, A the prop disc area, FM the propeller figure of merit
and eta the motor + ESC efficiency. Avionics (flight controller, companion
computer, camera, sensors) add a constant draw.

Rotor tip speed, the main driver of propeller noise, follows from
T = C_T * rho * A * (Omega R)^2. Tonal rotor noise grows roughly with the 5th
power of tip speed (about 50 dB per decade), so slower tips mean a quieter,
stealthier drone.
"""

from __future__ import annotations

import math

import numpy as np

from skymind.config import BatteryConfig, VehicleConfig

RHO = 1.225  # kg/m^3, sea-level air
GRAVITY = 9.81


class PowerModel:
    def __init__(self, v: VehicleConfig):
        self.v = v
        self.disc_area = math.pi * (v.prop_diameter / 2.0) ** 2
        self._k = 1.0 / (v.figure_of_merit * v.drivetrain_efficiency * math.sqrt(2.0 * RHO * self.disc_area))

    def electrical_power(self, motor_thrust: np.ndarray) -> float:
        t = np.maximum(np.asarray(motor_thrust, dtype=float), 0.0)
        return float(self._k * np.sum(t**1.5)) + self.v.avionics_power

    def tip_speed(self, motor_thrust: np.ndarray) -> float:
        """Mean rotor tip speed in m/s."""
        t = np.maximum(np.asarray(motor_thrust, dtype=float), 0.0)
        return float(np.mean(np.sqrt(t / (self.v.thrust_coefficient * RHO * self.disc_area))))

    def hover_power(self) -> float:
        return self.electrical_power(np.full(4, self.v.mass * GRAVITY / 4.0))

    def hover_tip_speed(self) -> float:
        return self.tip_speed(np.full(4, self.v.mass * GRAVITY / 4.0))

    def hover_endurance_min(self, battery: BatteryConfig) -> float:
        usable = battery.capacity_wh * (1.0 - battery.rtl_threshold)
        return 60.0 * usable / self.hover_power()

    def thrust_to_weight(self) -> float:
        return 4.0 * self.v.max_motor_thrust / (self.v.mass * GRAVITY)


def relative_noise_db(tip_speed: float, reference_tip_speed: float) -> float:
    """Approximate loudness change of rotor tonal noise vs. a reference tip speed."""
    if tip_speed <= 0 or reference_tip_speed <= 0:
        return 0.0
    return 50.0 * math.log10(tip_speed / reference_tip_speed)
