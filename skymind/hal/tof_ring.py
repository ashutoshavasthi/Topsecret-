"""Ring of VL53L1X time-of-flight sensors on the Raspberry Pi I2C bus.

All VL53L1X boot at address 0x29, so each sensor's XSHUT pin goes to its own
GPIO. At start-up the sensors are released one by one and given a unique
address. Needs ``pip install adafruit-circuitpython-vl53l1x adafruit-blinka``.

Wiring (per sensor): VIN to 3.3 V, GND, SDA to GPIO2, SCL to GPIO3, XSHUT to the GPIO listed
in ``xshut_pins`` (same order as ``sensors.tof_headings_deg``).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from skymind.config import SensorConfig
from skymind.perception.range_sensors import body_angles, expand_cone_readings

DEFAULT_XSHUT_PINS = ["D17", "D27", "D22", "D23", "D24", "D25"]


class ToFRing:
    def __init__(self, cfg: SensorConfig, xshut_pins: list[str] | None = None, first_address: int = 0x30):
        import adafruit_vl53l1x
        import board
        import digitalio

        self.cfg = cfg
        self.body_angles = body_angles(cfg)
        pins = (xshut_pins or DEFAULT_XSHUT_PINS)[: len(cfg.tof_headings_deg)]
        if len(pins) < len(cfg.tof_headings_deg):
            raise ValueError("need one XSHUT pin per ToF sensor")
        i2c = board.I2C()
        shut = []
        for name in pins:
            p = digitalio.DigitalInOut(getattr(board, name))
            p.switch_to_output(value=False)  # hold every sensor in reset
            shut.append(p)
        time.sleep(0.01)
        self.sensors = []
        for i, p in enumerate(shut):
            p.value = True  # wake this one only, then move it off 0x29
            time.sleep(0.01)
            s = adafruit_vl53l1x.VL53L1X(i2c)
            s.set_address(first_address + i)
            s.distance_mode = 2  # long range (up to ~4 m indoors)
            s.timing_budget = 33  # ms -> ~30 Hz per sensor
            s.start_ranging()
            self.sensors.append(s)
        self._readings = np.full(len(self.sensors), cfg.lidar_max_range)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            for i, s in enumerate(self.sensors):
                if s.data_ready:
                    d = s.distance  # cm, or None when nothing is in range
                    s.clear_interrupt()
                    with self._lock:
                        self._readings[i] = self.cfg.lidar_max_range if d is None else d / 100.0
            time.sleep(0.005)

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            readings = self._readings.copy()
        return expand_cone_readings(readings, self.cfg), self.body_angles

    def close(self) -> None:
        self._stop.set()
        for s in self.sensors:
            s.stop_ranging()
