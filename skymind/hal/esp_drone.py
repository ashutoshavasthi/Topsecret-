"""Ground-station link to an ESP-Drone (Espressif's ESP32 port of the Crazyflie firmware).

The drone balances itself on board (ESP-Drone's attitude/rate loops at 500 Hz) and
holds position with its optical-flow sensor. The laptop runs SkyMind's brain
and streams *hover setpoints* (body-frame velocity, yaw rate, absolute height)
over WiFi using Crazyflie's CRTP protocol. Requires ``pip install cflib``.

Default ESP-Drone network: join its WiFi AP, then use URI ``udp://192.168.43.42``.

The onboard commander has a watchdog. If setpoints stop for ~0.5 s it levels the
drone, and after ~2 s it stops the motors (a fall from about 1 m, survivable for an 80 g
drone with prop guards). So this class resends the last setpoint every loop (``keepalive``).

Sign conventions differ between Crazyflie versions. Verify both on the bench (props off,
watching logged values) before the first flight:
  * ``YAWRATE_SIGN``: Crazyflie's hover setpoint treats positive yaw rate as clockwise.
  * ``PITCH_SIGN``: ``stabilizer.pitch`` is logged with the legacy inverted sign.
"""

from __future__ import annotations

import math
import time

import numpy as np

from skymind.hal.mavlink_bridge import VehicleTelemetry
from skymind.math3d import wrap_angle

YAWRATE_SIGN = -1.0
PITCH_SIGN = -1.0
YAW_GAIN = 2.0  # 1/s, yaw-angle error -> yaw-rate command
MAX_YAW_RATE = math.radians(120.0)
TAKEOFF_RATE = 0.4  # m/s
LAND_RATE = 0.3  # m/s
SETPOINT_HOLD = 0.15  # s; older velocity commands are replaced by a hover


def world_to_body(v_world_xy: np.ndarray, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * v_world_xy[0] + s * v_world_xy[1], -s * v_world_xy[0] + c * v_world_xy[1]


def battery_fraction_1s(vbat: float) -> float:
    """Very rough 1S LiPo state of charge from voltage (sags under load)."""
    return float(np.clip((vbat - 3.5) / (4.15 - 3.5), 0.0, 1.0))


class EspDroneLink:
    """Same duck-typed interface as :class:`MavlinkVehicle`, used by ``RealFlightRunner``."""

    def __init__(self, uri: str, max_altitude: float = 2.0, crazyflie=None):
        self._scf = None
        if crazyflie is None:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

            cflib.crtp.init_drivers()
            self._scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./.cfcache"))
            crazyflie = self._scf.cf
        self.cf = crazyflie
        self.max_altitude = max_altitude
        self.telemetry = VehicleTelemetry()
        self.z_target = 0.0
        self.takeoff_altitude: float | None = None
        self.flying = False
        self.landing = False
        self._cmd = (0.0, 0.0, 0.0)  # body vx, vy, yaw rate (rad/s)
        self._cmd_t = 0.0
        self._last_send = time.time()

    # ------------------------------------------------------------ lifecycle
    def wait_ready(self, timeout: float = 10.0) -> None:
        if self._scf is not None:
            self._scf.open_link()
        self._start_logging()
        end = time.time() + timeout
        while time.time() < end and self.telemetry.last_update == 0.0:
            time.sleep(0.05)
        if self.telemetry.last_update == 0.0:
            raise TimeoutError("no telemetry from ESP-Drone (joined its WiFi? correct URI?)")

    def _start_logging(self) -> None:
        from cflib.crazyflie.log import LogConfig

        # CRTP log blocks carry at most 26 bytes, so split into two blocks of floats.
        pos = LogConfig(name="skymind_pos", period_in_ms=50)
        for name in ("x", "y", "z", "vx", "vy", "vz"):
            pos.add_variable(f"stateEstimate.{name}", "float")
        att = LogConfig(name="skymind_att", period_in_ms=50)
        for name in ("stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw", "pm.vbat"):
            att.add_variable(name, "float")
        for block in (pos, att):
            self.cf.log.add_config(block)
            block.data_received_cb.add_callback(self._on_log)
            block.start()

    def _on_log(self, _timestamp, data: dict, _block) -> None:
        tm = self.telemetry
        if "stateEstimate.x" in data:
            tm.position = np.array([data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"]])
            tm.velocity = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"], data["stateEstimate.vz"]])
            tm.landed = not self.flying and tm.position[2] < 0.08
        if "stabilizer.roll" in data:
            tm.rpy = (
                math.radians(data["stabilizer.roll"]),
                PITCH_SIGN * math.radians(data["stabilizer.pitch"]),
                math.radians(data["stabilizer.yaw"]),
            )
            tm.battery_fraction = battery_fraction_1s(data["pm.vbat"])
        tm.armed = self.flying
        tm.last_update = time.time()

    def set_mode(self, mode: str) -> None:  # ESP-Drone has no flight modes to switch
        pass

    def arm(self, timeout: float = 10.0) -> None:
        self.cf.commander.send_setpoint(0, 0, 0, 0)  # unlocks Crazyflie's thrust lock

    def takeoff(self, altitude: float) -> None:
        self.flying, self.landing = True, False
        self.takeoff_altitude = min(altitude, self.max_altitude)
        self.z_target = max(self.telemetry.position[2], 0.0)

    def land(self) -> None:
        self.landing = True

    rtl = land  # no GPS indoors: the brain flies home itself, this is only a fallback

    # ------------------------------------------------------------ setpoints
    def send_velocity_enu(self, velocity: np.ndarray, yaw_enu: float, altitude: float | None = None) -> None:
        yaw = self.telemetry.rpy[2]
        vx, vy = world_to_body(np.asarray(velocity[:2], dtype=float), yaw)
        rate = float(np.clip(YAW_GAIN * wrap_angle(yaw_enu - yaw), -MAX_YAW_RATE, MAX_YAW_RATE))
        if altitude is not None:
            self.takeoff_altitude = None
            self.z_target = float(np.clip(altitude, 0.15, self.max_altitude))
        self._cmd, self._cmd_t = (vx, vy, rate), time.time()
        self._send()

    def send_obstacle_distance(self, *_args) -> None:  # the ESP-Drone does not take proximity data
        pass

    def keepalive(self) -> None:
        """Call every loop while flying: ramps takeoff/landing and feeds the watchdog."""
        now = time.time()
        dt, self._last_send = min(now - self._last_send, 0.2), now
        if not self.flying:
            return
        if self.landing:
            self.z_target -= LAND_RATE * dt
            self._cmd = (0.0, 0.0, 0.0)
            if self.z_target <= 0.08:
                self.cf.commander.send_stop_setpoint()
                self.flying = False
                self.telemetry.landed = True
                return
        elif self.takeoff_altitude is not None:
            self.z_target = min(self.z_target + TAKEOFF_RATE * dt, self.takeoff_altitude)
        if now - self._cmd_t > SETPOINT_HOLD:
            self._cmd = (0.0, 0.0, 0.0)  # brain went quiet: hover rather than keep drifting
        self._send()

    def _send(self) -> None:
        if not self.flying:
            return
        vx, vy, rate = self._cmd
        self.cf.commander.send_hover_setpoint(vx, vy, YAWRATE_SIGN * math.degrees(rate), max(self.z_target, 0.0))

    def poll(self) -> VehicleTelemetry:
        return self.telemetry  # updated asynchronously by the log callbacks

    def close(self) -> None:
        try:
            self.cf.commander.send_stop_setpoint()
        finally:
            if self._scf is not None:
                self._scf.close_link()
