import json
import math
import socket
import time

import numpy as np

from skymind.config import load_config
from skymind.design import evaluate
from skymind.hal import esp_drone
from skymind.hal.esp_drone import EspDroneLink, battery_fraction_1s, world_to_body
from skymind.hal.esp_sensors import EspCamSensors, parse_packet
from skymind.mission import SimulationRunner
from skymind.perception.avoidance import braking_speed


def test_braking_speed_budgets_reaction_time():
    assert math.isclose(braking_speed(1.0, 2.0, 0.0), 2.0)
    v = braking_speed(1.0, 2.0, 0.15)
    assert v < 2.0
    assert math.isclose(v * 0.15 + v * v / (2 * 2.0), 1.0)  # stops exactly within the room
    assert braking_speed(-0.5, 2.0, 0.15) == 0.0


def test_nano_profile_is_cheap_class_hardware():
    r = evaluate(load_config("config/nano.yaml"))
    assert r.mass_g <= 100 and r.sub_250g
    assert 1.8 <= r.thrust_to_weight <= 2.5
    assert 3.0 < r.hover_endurance_min < 7.0  # brushed micro drones fly ~4-6 min


def test_nano_room_mission_over_lossy_wifi(tmp_path):
    cfg = load_config("config/nano_room.yaml")
    assert cfg.sim.link_latency > 0 and cfg.sim.link_drop_prob > 0
    r = SimulationRunner(cfg, tmp_path, record_video=False).run()
    assert r.status == "landed", r.events
    assert r.collisions == 0
    assert r.waypoints_reached == r.waypoints_total


def test_dead_link_never_takes_off(tmp_path):
    cfg = load_config("config/nano_room.yaml", {"sim": {"link_drop_prob": 1.0}, "mission": {"max_duration": 3.0}})
    runner = SimulationRunner(cfg, tmp_path, record_video=False)
    r = runner.run()
    assert r.collisions == 0
    assert runner.sim.state.position[2] < 0.05  # no setpoint ever arrived: motors stayed off


class _FakeCommander:
    def __init__(self):
        self.sent = []

    def send_hover_setpoint(self, vx, vy, yawrate, z):
        self.sent.append(("hover", vx, vy, yawrate, z))

    def send_stop_setpoint(self):
        self.sent.append(("stop",))

    def send_setpoint(self, *a):
        self.sent.append(("raw", *a))


class _FakeCf:
    def __init__(self):
        self.commander = _FakeCommander()


def test_world_to_body():
    vx, vy = world_to_body(np.array([0.0, 0.5]), math.pi / 2)  # facing north, flying north
    assert math.isclose(vx, 0.5) and abs(vy) < 1e-12
    assert battery_fraction_1s(4.2) == 1.0 and battery_fraction_1s(3.4) == 0.0


def test_esp_drone_setpoints_takeoff_and_landing(monkeypatch):
    cf = _FakeCf()
    link = EspDroneLink("udp://fake", max_altitude=1.5, crazyflie=cf)
    link._on_log(0, {"stabilizer.roll": 0.0, "stabilizer.pitch": 0.0, "stabilizer.yaw": 0.0, "pm.vbat": 4.0}, None)
    link.arm()
    assert cf.commander.sent[-1] == ("raw", 0, 0, 0, 0)  # thrust-lock release
    link.takeoff(1.0)
    clock = [100.0]
    monkeypatch.setattr(esp_drone.time, "time", lambda: clock[0])
    link._last_send = clock[0]
    for _ in range(40):  # 4 s of keepalive at 10 Hz: climbs to 1.0 m and holds
        clock[0] += 0.1
        link.keepalive()
    assert math.isclose(link.z_target, 1.0)
    # Face east, ask to fly north-east and turn left (CCW) toward north.
    link.send_velocity_enu(np.array([0.3, 0.3, 0.0]), math.pi / 2, altitude=1.2)
    _, vx, vy, yawrate, z = cf.commander.sent[-1]
    assert math.isclose(vx, 0.3) and math.isclose(vy, 0.3)
    assert yawrate * esp_drone.YAWRATE_SIGN > 0 and z == 1.2
    # A quiet brain degrades to a hover, it does not keep flying the old velocity.
    clock[0] += 0.5
    link.keepalive()
    assert cf.commander.sent[-1][1:3] == (0.0, 0.0)
    link.land()
    for _ in range(60):
        clock[0] += 0.1
        link.keepalive()
    assert cf.commander.sent[-1] == ("stop",) and not link.flying and link.telemetry.landed


def test_parse_tof_packet():
    r = parse_packet(json.dumps({"t": 1, "tof_mm": [500, 8190, 0]}).encode(), 3, 1.5)
    assert np.allclose(r, [0.5, 1.5, 1.5])
    assert parse_packet(b"garbage", 3, 1.5) is None
    assert parse_packet(json.dumps({"tof_mm": [1, 2]}).encode(), 3, 1.5) is None


def test_udp_sensors_and_stale_data_reads_blocked():
    cfg = load_config("config/nano.yaml").sensors
    sensors = EspCamSensors(cfg, port=0, stale_after=0.3)
    try:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(json.dumps({"t": 1, "tof_mm": [400, 900, 1200]}).encode(), ("127.0.0.1", sensors.port))
        deadline = time.time() + 2.0
        while time.time() < deadline and sensors.scan()[0].max() == 0.0:
            time.sleep(0.02)
        ranges, angles = sensors.scan()
        assert len(ranges) == len(angles) == 3 * cfg.tof_subrays
        assert np.allclose(ranges.reshape(3, -1)[:, 0], [0.4, 0.9, 1.2])
        time.sleep(0.4)
        assert sensors.scan()[0].max() == 0.0  # no fresh data: everything looks blocked
    finally:
        sensors.close()
