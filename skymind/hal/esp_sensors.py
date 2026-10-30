"""Receive ToF readings broadcast by the ESP32-CAM sensor firmware
(``firmware/esp32cam_sensors``) as UDP JSON packets::

    {"t": 123456, "tof_mm": [812, 1500, 64]}

Readings older than ``stale_after`` make every direction read 0 m, so the brain
sees "blocked" and holds position instead of flying blind.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np

from skymind.config import SensorConfig
from skymind.perception.range_sensors import body_angles, expand_cone_readings

OUT_OF_RANGE_MM = 8000  # VL53L0X reports 8190 / 65535 when nothing is in range


def parse_packet(payload: bytes, n_sensors: int, max_range: float) -> np.ndarray | None:
    try:
        mm = json.loads(payload.decode("utf-8"))["tof_mm"]
    except (ValueError, KeyError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(mm, list) or len(mm) != n_sensors:
        return None
    m = np.array(mm, dtype=float)
    return np.where((m <= 0) | (m >= OUT_OF_RANGE_MM), max_range, np.minimum(m / 1000.0, max_range))


class EspCamSensors:
    def __init__(self, cfg: SensorConfig, port: int = 2399, stale_after: float = 0.3):
        self.cfg = cfg
        self.body_angles = body_angles(cfg)
        self.stale_after = stale_after
        self._readings = np.zeros(len(cfg.tof_headings_deg))
        self._t = 0.0
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", port))
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        n = len(self.cfg.tof_headings_deg)
        while not self._stop.is_set():
            try:
                payload, _ = self._sock.recvfrom(512)
            except (socket.timeout, OSError):
                continue
            r = parse_packet(payload, n, self.cfg.lidar_max_range)
            if r is not None:
                with self._lock:
                    self._readings, self._t = r, time.time()

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            readings = self._readings.copy() if time.time() - self._t < self.stale_after else np.zeros_like(self._readings)
        return expand_cone_readings(readings, self.cfg), self.body_angles

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
