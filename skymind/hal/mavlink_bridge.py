"""Companion-computer link to a real flight controller over MAVLink.

On real hardware the fast inner loops (IMU fusion, attitude and rate control,
i.e. the actual self-balancing) run on the flight controller firmware
(ArduPilot or PX4 on a Pixhawk-class board) at 400 Hz+. SkyMind runs on the
companion computer (Raspberry Pi 5 / Jetson Orin Nano) and sends velocity
and yaw setpoints, exactly the interface the simulator's ``FlightBrain`` uses.

Requires ``pip install pymavlink``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from skymind.hal.frames import attitude_ned_to_enu, enu_to_ned, ned_to_enu, yaw_enu_to_ned

# SET_POSITION_TARGET_LOCAL_NED type_mask: use velocity + yaw, ignore position, accel and yaw rate.
VELOCITY_YAW_MASK = 0b0000_1001_1100_0111


@dataclass
class VehicleTelemetry:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # ENU, m, relative to EKF origin
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # ENU/FLU
    rates: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery_fraction: float = 1.0
    armed: bool = False
    mode: str = ""
    landed: bool = True
    last_update: float = 0.0


class MavlinkVehicle:
    def __init__(self, url: str, baud: int = 921600, connection=None):
        if connection is None:
            from pymavlink import mavutil  # imported lazily so the simulator never needs it

            connection = mavutil.mavlink_connection(url, baud=baud)
        self.conn = connection
        self.telemetry = VehicleTelemetry()

    # -------------------------------------------------------------- lifecycle
    def wait_ready(self, timeout: float = 30.0) -> None:
        if self.conn.wait_heartbeat(timeout=timeout) is None:
            raise TimeoutError("no heartbeat from flight controller")
        mav = self.conn.mav
        # Ask for the streams we use at useful rates (Hz).
        for msg_id, hz in ((32, 20), (30, 50), (147, 2), (1, 2), (245, 5)):  # LOCAL_POSITION_NED, ATTITUDE, BATTERY, SYS_STATUS, EXTENDED_SYS_STATE
            mav.command_long_send(self.conn.target_system, self.conn.target_component, 511, 0, msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def set_mode(self, mode: str) -> None:
        self.conn.set_mode(mode)

    def arm(self, timeout: float = 10.0) -> None:
        self.conn.arducopter_arm()
        end = time.time() + timeout
        while time.time() < end:
            self.poll()
            if self.telemetry.armed:
                return
            time.sleep(0.05)
        raise TimeoutError("vehicle did not arm (pre-arm checks?)")

    def takeoff(self, altitude: float) -> None:
        mav = self.conn.mav
        mav.command_long_send(self.conn.target_system, self.conn.target_component, 22, 0, 0, 0, 0, 0, 0, 0, altitude)

    def land(self) -> None:
        self.set_mode("LAND")

    def rtl(self) -> None:
        self.set_mode("RTL")

    # -------------------------------------------------------------- setpoints
    def send_velocity_enu(self, velocity: np.ndarray, yaw_enu: float) -> None:
        vn = enu_to_ned(velocity)
        self.conn.mav.set_position_target_local_ned_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            1,  # MAV_FRAME_LOCAL_NED
            VELOCITY_YAW_MASK,
            0, 0, 0,
            float(vn[0]), float(vn[1]), float(vn[2]),
            0, 0, 0,
            yaw_enu_to_ned(yaw_enu),
            0,
        )  # fmt: skip

    def send_obstacle_distance(self, ranges_m: np.ndarray, body_angles_flu: np.ndarray, max_range: float) -> None:
        """Forward the lidar to the autopilot so its own proximity avoidance is a second safety layer.

        OBSTACLE_DISTANCE uses 72 x 5-degree sectors, clockwise from forward in the FRD body frame.
        """
        frd_angles = np.radians(np.arange(72) * 5.0)
        diff = np.abs(((-frd_angles)[:, None] - body_angles_flu[None, :] + np.pi) % (2 * np.pi) - np.pi)
        nearest = ranges_m[np.argmin(diff, axis=1)]
        cm = np.where(nearest >= max_range - 1e-3, int(max_range * 100) + 1, np.round(nearest * 100)).astype(int)
        self.conn.mav.obstacle_distance_send(
            int(time.time() * 1e6), 0, cm.tolist(), 5, 20, int(max_range * 100), 0.0, 0.0, 12  # MAV_FRAME_BODY_FRD
        )

    # --------------------------------------------------------------- telemetry
    def poll(self) -> VehicleTelemetry:
        tm = self.telemetry
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                break
            kind = msg.get_type()
            if kind == "LOCAL_POSITION_NED":
                tm.position = ned_to_enu(np.array([msg.x, msg.y, msg.z]))
                tm.velocity = ned_to_enu(np.array([msg.vx, msg.vy, msg.vz]))
            elif kind == "ATTITUDE":
                tm.rpy = attitude_ned_to_enu(msg.roll, msg.pitch, msg.yaw)
                tm.rates = np.array([msg.rollspeed, -msg.pitchspeed, -msg.yawspeed])
            elif kind == "BATTERY_STATUS" and msg.battery_remaining >= 0:
                tm.battery_fraction = msg.battery_remaining / 100.0
            elif kind == "SYS_STATUS" and msg.battery_remaining >= 0:
                tm.battery_fraction = msg.battery_remaining / 100.0
            elif kind == "EXTENDED_SYS_STATE":
                tm.landed = msg.landed_state == 1  # MAV_LANDED_STATE_ON_GROUND
            elif kind == "HEARTBEAT" and msg.get_srcSystem() == self.conn.target_system:
                tm.armed = bool(msg.base_mode & 0x80)
                tm.mode = getattr(self.conn, "flightmode", "")
            tm.last_update = time.time()
        return tm
