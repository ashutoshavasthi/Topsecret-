"""Slamtec RPLidar A1/A2/C1 driver wrapper (``pip install rplidar-roboticia``)."""

from __future__ import annotations

import threading

import numpy as np


class RPLidarSource:
    """Runs the lidar in a background thread and serves the latest binned scan.

    Assumes the lidar's 0-degree mark faces the drone's nose. RPLidar angles
    are clockwise; SkyMind body angles are counter-clockwise (FLU).
    """

    def __init__(self, port: str, beams: int, max_range: float):
        from rplidar import RPLidar

        self.lidar = RPLidar(port)
        self.beams, self.max_range = beams, max_range
        self.body_angles = np.linspace(-np.pi, np.pi, beams, endpoint=False)
        self._ranges = np.full(beams, max_range)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for scan in self.lidar.iter_scans():
            if self._stop.is_set():
                break
            ranges = np.full(self.beams, self.max_range)
            for _q, angle_deg, dist_mm in scan:
                if dist_mm <= 0:
                    continue
                a = (-np.radians(angle_deg) + np.pi) % (2 * np.pi) - np.pi
                i = int((a + np.pi) / (2 * np.pi) * self.beams) % self.beams
                ranges[i] = min(ranges[i], dist_mm / 1000.0)
            with self._lock:
                self._ranges = np.clip(ranges, 0.0, self.max_range)

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._ranges.copy(), self.body_angles

    def close(self) -> None:
        self._stop.set()
        self.lidar.stop()
        self.lidar.disconnect()
