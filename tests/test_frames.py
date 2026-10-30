import math

import numpy as np

from skymind.hal.frames import attitude_ned_to_enu, enu_to_ned, ned_to_enu, yaw_enu_to_ned, yaw_ned_to_enu
from skymind.hal.mavlink_bridge import VELOCITY_YAW_MASK, MavlinkVehicle


def test_enu_ned_roundtrip():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(enu_to_ned(v), [2.0, 1.0, -3.0])
    assert np.allclose(ned_to_enu(enu_to_ned(v)), v)


def test_yaw_conversion():
    assert math.isclose(yaw_enu_to_ned(0.0), math.pi / 2)  # facing east
    assert math.isclose(yaw_enu_to_ned(math.pi / 2), 0.0, abs_tol=1e-12)  # facing north
    assert math.isclose(yaw_ned_to_enu(yaw_enu_to_ned(1.0)), 1.0)
    assert attitude_ned_to_enu(0.1, 0.2, 0.0)[:2] == (0.1, -0.2)


def test_type_mask_uses_velocity_and_yaw_only():
    ignored = {0, 1, 2, 6, 7, 8, 11}  # x y z ax ay az yaw_rate
    assert VELOCITY_YAW_MASK == sum(1 << b for b in ignored)


class _FakeMav:
    def __init__(self):
        self.sent = []

    def set_position_target_local_ned_send(self, *args):
        self.sent.append(("pos_target", args))

    def obstacle_distance_send(self, *args):
        self.sent.append(("obstacle", args))


class _FakeConn:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.mav = _FakeMav()


def test_velocity_setpoint_encoding():
    v = MavlinkVehicle("fake", connection=_FakeConn())
    v.send_velocity_enu(np.array([1.0, 2.0, 0.5]), 0.0)  # east 1, north 2, up 0.5, facing east
    _, args = v.conn.mav.sent[-1]
    assert args[4] == VELOCITY_YAW_MASK
    assert args[8:11] == (2.0, 1.0, -0.5)  # N, E, D
    assert math.isclose(args[14], math.pi / 2)


def test_obstacle_distance_sector_order():
    v = MavlinkVehicle("fake", connection=_FakeConn())
    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    ranges = np.full(72, 8.0)
    ranges[np.argmin(np.abs(angles - math.pi / 2))] = 2.0  # obstacle on the LEFT (FLU +90)
    v.send_obstacle_distance(ranges, angles, 8.0)
    _, args = v.conn.mav.sent[-1]
    cm = args[2]
    # FRD is clockwise from the nose, so "left" is sector 270/5 = 54.
    assert cm[54] == 200 and cm[0] == 801
