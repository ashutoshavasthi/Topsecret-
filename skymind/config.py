"""Typed configuration loaded from YAML.

Every field has a sensible default so ``SkyMindConfig()`` is a complete,
flyable configuration. YAML files only need to override what they change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VehicleConfig:
    mass: float = 1.2  # kg
    arm_length: float = 0.18  # m, centre to motor
    inertia: tuple[float, float, float] = (0.0123, 0.0123, 0.0224)  # kg m^2
    max_motor_thrust: float = 8.0  # N per motor
    yaw_moment_coeff: float = 0.016  # N m of reaction torque per N of thrust
    motor_time_constant: float = 0.03  # s
    linear_drag: float = 0.25  # N per m/s
    body_radius: float = 0.3  # m, used for collision checks
    # Power / noise model (see skymind/power.py)
    prop_diameter: float = 0.254  # m (10")
    figure_of_merit: float = 0.6  # propeller hover efficiency
    drivetrain_efficiency: float = 0.75  # motor x ESC
    thrust_coefficient: float = 0.01  # C_T in T = C_T rho A (Omega R)^2
    avionics_power: float = 7.0  # W: flight controller + companion computer + camera + sensors


@dataclass
class ControlConfig:
    pos_gain: tuple[float, float, float] = (1.0, 1.0, 1.5)
    vel_p: tuple[float, float, float] = (2.5, 2.5, 4.0)
    vel_i: tuple[float, float, float] = (0.4, 0.4, 1.0)
    vel_d: tuple[float, float, float] = (0.05, 0.05, 0.0)
    att_gain: tuple[float, float, float] = (8.0, 8.0, 3.0)
    rate_p: tuple[float, float, float] = (20.0, 20.0, 8.0)
    rate_i: tuple[float, float, float] = (2.0, 2.0, 1.0)
    rate_d: tuple[float, float, float] = (0.3, 0.3, 0.0)
    max_tilt_deg: float = 25.0
    max_rate: tuple[float, float, float] = (4.0, 4.0, 2.0)  # rad/s
    max_horizontal_speed: float = 4.0  # m/s
    max_vertical_speed: float = 2.0  # m/s


@dataclass
class EstimatorConfig:
    mahony_kp: float = 1.0
    mahony_ki: float = 0.1
    yaw_gain: float = 0.5
    pos_meas_std: float = 0.05  # m (GNSS-RTK / optical flow / mocap)
    accel_process_std: float = 3.0  # m/s^2 (covers attitude-error leakage into the accel model)


@dataclass
class SensorConfig:
    gyro_noise_std: float = 0.005  # rad/s
    gyro_bias_std: float = 0.01  # rad/s
    accel_noise_std: float = 0.05  # m/s^2
    pos_noise_std: float = 0.05  # m
    pos_rate_hz: float = 20.0
    range_sensor: str = "lidar"  # "lidar" (360-degree scanner) or "tof_ring" (cheap VL53L1X cones)
    lidar_beams: int = 72
    lidar_max_range: float = 8.0  # m (VL53L1X long mode: ~4 m indoors, ~2 m in sunlight)
    lidar_noise_std: float = 0.02  # m
    lidar_rate_hz: float = 20.0
    tof_headings_deg: list[float] = field(default_factory=lambda: [-90.0, -30.0, 0.0, 30.0, 90.0, 180.0])
    tof_fov_deg: float = 27.0
    tof_subrays: int = 5  # sub-rays per cone used to represent each reading
    tof_noise_frac: float = 0.02  # ToF error grows with distance


@dataclass
class AvoidanceConfig:
    safety_radius: float = 0.9  # m, clearance corridor half-width
    stop_distance: float = 1.0  # m, emergency brake distance
    lookahead: float = 4.0  # m, required free distance to commit to a heading
    num_headings: int = 72
    goal_weight: float = 1.0
    smooth_weight: float = 0.35
    max_decel: float = 2.5  # m/s^2 used for speed limiting near obstacles
    repulse_gain: float = 1.5
    # Jerk-limited reference model that smooths the avoider's velocity commands
    # (smooth flight = steady footage) and provides a lag-free acceleration feed-forward.
    max_accel: float = 2.5  # m/s^2
    max_jerk: float = 15.0  # m/s^3
    ref_gain: float = 3.0  # 1/s, how quickly the reference converges to the command
    use_global_planner: bool = True  # lidar occupancy map + A* to escape local minima
    map_resolution: float = 0.25  # m per grid cell
    replan_interval: float = 1.0  # s
    path_lookahead: float = 5.0  # m, furthest path point handed to the reactive avoider
    # With sparse sensors (ToF ring), slow down until the nose (best-covered sector) faces the
    # direction of travel. None = move in any direction at full speed (360-degree lidar).
    max_heading_offset_deg: float | None = None


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 240
    fov_deg: float = 90.0
    fps: float = 10.0
    device: int | str = 0  # used for real cameras only
    gimbal_stabilized: bool = True
    motion_blur_gain: float = 6.0  # px of blur per rad/s of residual body rate


@dataclass
class VisionConfig:
    onnx_model: str | None = None  # path to a YOLOv8 ONNX model (optional)
    onnx_labels: list[str] = field(default_factory=list)
    onnx_conf_threshold: float = 0.4
    color_targets: dict[str, list] = field(
        default_factory=lambda: {
            # label: [[h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]] or a list of such ranges (OpenCV HSV)
            "red_marker": [[[0, 140, 90], [8, 255, 255]], [[170, 140, 90], [180, 255, 255]]],
            "yellow_marker": [[22, 140, 120], [34, 255, 255]],
        }
    )
    onnx_input_size: int = 640  # 320 is ~4x cheaper (Pi Zero / Pi 4), at some loss on small objects
    min_blob_area: int = 12
    snapshot_on_detection: bool = True
    detect_every: int = 1  # run the detectors on every Nth camera frame (CPU saving)


@dataclass
class BatteryConfig:
    capacity_wh: float = 40.0
    cells: int = 4  # LiPo cells in series (3.7 V nominal each)
    specific_energy: float = 0.15  # Wh per gram of pack (small packs ~0.13, large ~0.17)
    rtl_threshold: float = 0.2  # fraction remaining


@dataclass
class MissionConfig:
    home: tuple[float, float] = (0.0, 0.0)
    cruise_altitude: float = 2.0
    cruise_speed: float = 2.5
    waypoint_radius: float = 0.6
    waypoints: list[list[float]] = field(default_factory=lambda: [[20.0, 0.0]])
    max_duration: float = 120.0
    geofence: tuple[float, float, float, float] = (-10.0, -15.0, 35.0, 15.0)  # xmin ymin xmax ymax
    max_altitude: float = 30.0
    blocked_timeout: float = 8.0
    max_yaw_rate: float = 1.2  # rad/s


@dataclass
class WorldConfig:
    # cylinders: [x, y, radius, height]; boxes: [xmin, ymin, xmax, ymax, height]
    cylinders: list[list[float]] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    # targets: [x, y, z, radius, label]
    targets: list[list[Any]] = field(default_factory=list)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_std: float = 0.0


@dataclass
class JudgeConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {"stability": 0.25, "navigation": 0.2, "safety": 0.3, "vision": 0.1, "efficiency": 0.15}
    )
    # [good, bad] thresholds; a metric scores 100 at "good" and 0 at "bad", linear between.
    thresholds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "rms_body_rate": [0.2, 1.5],  # rad/s, roll+pitch
            "rms_altitude_error": [0.1, 0.8],  # m
            "max_tilt_deg": [20.0, 45.0],
            "rms_attitude_estimate_error_deg": [1.0, 6.0],
            "rms_velocity_tracking_error": [0.2, 1.5],  # m/s, commanded vs actual
            "waypoint_completion": [1.0, 0.0],
            "path_efficiency": [0.85, 0.4],
            "min_clearance": [0.8, 0.3],  # m from vehicle centre to nearest surface
            "sharpness": [40.0, 8.0],  # variance of Laplacian (camera-dependent!)
            "exposure_error": [0.0, 80.0],  # |mean luma - 128|
            "clipped_fraction": [0.0, 0.2],
            "detection_recall": [0.9, 0.2],
            "detection_precision": [0.9, 0.3],
            "energy_per_km_wh": [20.0, 60.0],  # lower = more efficient
            "est_endurance_min": [15.0, 5.0],  # at the flight's average power
            "mean_tip_speed": [60.0, 110.0],  # m/s; slower rotor tips = quieter (stealthier)
        }
    )
    min_gt_area: int = 40  # px; ground-truth targets smaller than this are too far to count


@dataclass
class SimConfig:
    physics_hz: float = 250.0
    seed: int = 7


@dataclass
class SkyMindConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    sim: SimConfig = field(default_factory=SimConfig)


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> Any:
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"{path or 'config'} is not a section")
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in fields:
            raise KeyError(f"unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            elif isinstance(current, dict) and isinstance(value, dict):
                value = {**current, **value}  # override individual entries, keep the rest
            setattr(obj, key, value)
    return obj


def _apply_file(cfg: SkyMindConfig, path: Path, depth: int = 0) -> None:
    """Merge a YAML file. ``base: other.yaml`` (relative to this file) is applied first,
    so a scenario can build on a hardware profile such as ``micro.yaml``."""
    if depth > 5:
        raise RecursionError("config 'base' chain too deep")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("base", None)
    if base:
        _apply_file(cfg, (path.parent / base).resolve(), depth + 1)
    _merge(cfg, data)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> SkyMindConfig:
    cfg = SkyMindConfig()
    if path is not None:
        _apply_file(cfg, Path(path))
    if overrides:
        _merge(cfg, overrides)
    return cfg


def config_to_dict(cfg: SkyMindConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
