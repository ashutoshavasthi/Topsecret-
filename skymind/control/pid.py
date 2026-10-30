"""Vector PID controller with anti-windup and filtered derivative."""

from __future__ import annotations

import numpy as np


class PID:
    """Per-axis PID operating on numpy vectors.

    * Derivative is taken on the measurement (no derivative kick on setpoint
      steps) and low-pass filtered.
    * Integrator is clamped and frozen while the output is saturated in the
      same direction as the error (conditional integration anti-windup).
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        output_limit: float | np.ndarray | None = None,
        integral_limit: float | np.ndarray | None = None,
        d_filter_hz: float = 30.0,
    ):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.output_limit = None if output_limit is None else np.asarray(output_limit, dtype=float)
        self.integral_limit = None if integral_limit is None else np.asarray(integral_limit, dtype=float)
        self.d_filter_hz = d_filter_hz
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros_like(self.kp)
        self._prev_meas: np.ndarray | None = None
        self._d_state = np.zeros_like(self.kp)

    def update(self, setpoint, measurement, dt: float) -> np.ndarray:
        setpoint = np.asarray(setpoint, dtype=float)
        measurement = np.asarray(measurement, dtype=float)
        error = setpoint - measurement

        if self._prev_meas is None or dt <= 0.0:
            d_raw = np.zeros_like(error)
        else:
            d_raw = -(measurement - self._prev_meas) / dt
        self._prev_meas = measurement.copy()
        if dt > 0.0:
            alpha = dt / (dt + 1.0 / (2.0 * np.pi * self.d_filter_hz))
            self._d_state = self._d_state + alpha * (d_raw - self._d_state)

        unsat = self.kp * error + self.integral + self.kd * self._d_state
        out = unsat if self.output_limit is None else np.clip(unsat, -self.output_limit, self.output_limit)

        # Only integrate when that would not push further into saturation.
        saturated = out != unsat
        pushing = np.sign(error) == np.sign(unsat)
        self.integral = self.integral + np.where(saturated & pushing, 0.0, self.ki * error * dt)
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        return out
