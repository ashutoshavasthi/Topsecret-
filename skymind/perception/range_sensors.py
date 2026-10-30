"""Range-sensor layouts shared by the simulator and the hardware drivers.

Two layouts are supported:

* ``lidar``: a spinning 360-degree lidar with many narrow beams (RPLidar).
* ``tof_ring``: a few cheap single-zone time-of-flight sensors (VL53L1X,
  2 g and about $6 each). Each one reports the *nearest* return inside a
  ~27 degree cone. A cone reading is expanded into several sub-rays that all
  carry that distance. This is conservative (the obstacle may be anywhere in
  the cone), so the avoider and mapper keep working unchanged.

Gaps between cones are blind. The flight brain compensates by turning to face
its direction of travel before moving fast (``avoidance.max_heading_offset_deg``).
"""

from __future__ import annotations

import numpy as np

from skymind.config import SensorConfig


def cone_subray_angles(cfg: SensorConfig) -> np.ndarray:
    """Body-frame (FLU, CCW) angles of every sub-ray, sensor-major order."""
    heads = np.radians(np.asarray(cfg.tof_headings_deg, dtype=float))
    half = np.radians(cfg.tof_fov_deg) / 2.0
    offsets = np.linspace(-half, half, cfg.tof_subrays)
    return (heads[:, None] + offsets[None, :]).ravel()


def body_angles(cfg: SensorConfig) -> np.ndarray:
    if cfg.range_sensor == "tof_ring":
        return cone_subray_angles(cfg)
    return np.linspace(-np.pi, np.pi, cfg.lidar_beams, endpoint=False)


def expand_cone_readings(readings: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Per-sensor distances (m) -> per-sub-ray ranges matching :func:`cone_subray_angles`."""
    r = np.clip(np.asarray(readings, dtype=float), 0.0, cfg.lidar_max_range)
    return np.repeat(r, cfg.tof_subrays)


def collapse_to_cones(subray_ranges: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Nearest return per cone, as a single-zone ToF sensor reports it."""
    return np.asarray(subray_ranges).reshape(len(cfg.tof_headings_deg), cfg.tof_subrays).min(axis=1)
