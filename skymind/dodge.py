"""Dodging thrown objects: threat sensors, an onboard reflex, and a throwing experiment.

Every distance sensor that works at range (light, infrared, radio, sound) detects
*reflections*, so paper, plastic and foam are seen just like metal. The hard part is
time: a ball thrown at 6 m/s covers 1.5 m in 0.25 s. The solution modelled here:

1. **Fast, wide threat sensors**
   * ``tof8x8``: VL53L5CX / VL53L7CX multizone ToF in 4x4-zone mode at 60 Hz.
     Infrared, works in the dark, white paper reflects it well.
   * ``looming``: insect-style looming detection on the onboard camera (things that
     *grow* fast in the image are coming at you). Free software on the ESP32-CAM.
2. **An onboard reflex.** It runs on the drone, wired to the sensor, with no WiFi round trip.
   ``reflex_latency`` models where it runs: ~0.01 s onboard, ~0.15 s via a laptop.
3. **Dodging like a fly.** Escape away from the predicted closest-approach point. When the
   ball would hit dead centre, *drop*: cutting thrust lets gravity accelerate the drone at
   up to ~0.9 g, which beats tilting sideways on weak motors.

The numbers are a simulation with estimated sensor specs (see ``SENSORS``). Real
sensors and real throws will differ, so verify on hardware with soft objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from skymind.config import SkyMindConfig
from skymind.control.estimator import StateEstimator
from skymind.control.flight_controller import FlightController, Setpoint
from skymind.sim.physics import QuadrotorSim

G = np.array([0.0, 0.0, -9.81])


@dataclass
class ThreatSensorSpec:
    name: str
    kind: str  # "tof" | "camera"
    rate_hz: float
    fov_h_deg: float
    fov_v_deg: float
    heading_deg: float = 0.0  # body yaw offset (0 = forward)
    latency: float = 0.005  # s from exposure to measurement
    # ToF: detection range for a small target ~ k x its diameter (weak returns), capped
    zones: int = 4
    max_range: float = 3.0
    k_per_diameter: float = 18.0
    range_noise: float = 0.03
    # camera: pixels across the field of view, minimum blob size, assumed object size
    width_px: int = 160
    min_px: float = 3.0
    size_prior: float = 0.08
    angle_noise_px: float = 0.5
    price_inr: str = ""


SENSORS: dict[str, list[ThreatSensorSpec]] = {
    "none": [],
    "tof8x8": [ThreatSensorSpec("VL53L5CX front", "tof", 60.0, 45.0, 45.0, price_inr="1,200-1,800")],
    "tof8x8_wide": [ThreatSensorSpec("VL53L7CX front (90 deg)", "tof", 60.0, 60.0, 60.0, price_inr="1,500-2,200")],
    "looming": [ThreatSensorSpec("ESP32-CAM looming (QQVGA 40 fps)", "camera", 40.0, 66.0, 50.0, latency=0.035, price_inr="0 (software)")],
    "tof8x8+looming": [
        ThreatSensorSpec("VL53L5CX front", "tof", 60.0, 45.0, 45.0),
        ThreatSensorSpec("ESP32-CAM looming", "camera", 40.0, 66.0, 50.0, latency=0.035),
    ],
    "looming_wide": [
        ThreatSensorSpec("ESP32-CAM + 120 deg lens, looming (QVGA 30 fps)", "camera", 30.0, 120.0, 90.0, latency=0.045,
                         width_px=320, price_inr="150-300 (lens)"),
    ],
    "tof8x8+looming_wide": [
        ThreatSensorSpec("VL53L5CX front", "tof", 60.0, 45.0, 45.0),
        ThreatSensorSpec("ESP32-CAM + 120 deg lens, looming", "camera", 30.0, 120.0, 90.0, latency=0.045, width_px=320),
    ],
    "tof8x8+looming_wide_x2": [
        ThreatSensorSpec("VL53L5CX front", "tof", 60.0, 45.0, 45.0),
        ThreatSensorSpec("ESP32-CAM front + 120 deg lens", "camera", 30.0, 120.0, 90.0, latency=0.045, width_px=320),
        ThreatSensorSpec("ESP32-CAM rear + 120 deg lens", "camera", 30.0, 120.0, 90.0, heading_deg=180.0, latency=0.045,
                         width_px=320, price_inr="600-900 (2nd ESP32-CAM + lens)"),
    ],
    "tof8x8x2+looming": [
        ThreatSensorSpec("VL53L5CX front", "tof", 60.0, 45.0, 45.0),
        ThreatSensorSpec("VL53L5CX rear", "tof", 60.0, 45.0, 45.0, heading_deg=180.0),
        ThreatSensorSpec("ESP32-CAM looming", "camera", 40.0, 66.0, 50.0, latency=0.035),
    ],
}


@dataclass
class Projectile:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    active: bool = True
    min_dist: float = float("inf")

    def step(self, dt: float) -> None:
        self.vel = self.vel + G * dt
        self.pos = self.pos + self.vel * dt
        if self.pos[2] < self.radius:
            self.active = False


def sense(spec: ThreatSensorSpec, drone_pos: np.ndarray, yaw: float, p: Projectile, rng) -> tuple[np.ndarray, float] | None:
    """Noisy relative position (world frame) of the projectile and its 1-sigma error, or None."""
    r = p.pos - drone_pos
    c, s = math.cos(-yaw), math.sin(-yaw)
    xb, yb = c * r[0] - s * r[1], s * r[0] + c * r[1]
    az = math.atan2(yb, xb) - math.radians(spec.heading_deg)
    az = (az + math.pi) % (2 * math.pi) - math.pi
    el = math.atan2(r[2], math.hypot(xb, yb))
    hh, hv = math.radians(spec.fov_h_deg) / 2, math.radians(spec.fov_v_deg) / 2
    if abs(az) > hh or abs(el) > hv:
        return None
    d = float(np.linalg.norm(r))
    if spec.kind == "tof":
        surface = d - p.radius
        if surface > min(spec.max_range, spec.k_per_diameter * 2 * p.radius) or surface <= 0:
            return None
        # Each zone only knows its own direction: quantise bearing to the zone centre.
        zw_h, zw_v = 2 * hh / spec.zones, 2 * hv / spec.zones
        az_q = (math.floor((az + hh) / zw_h) + 0.5) * zw_h - hh
        el_q = (math.floor((el + hv) / zw_v) + 0.5) * zw_v - hv
        rng_m = surface + rng.normal(0, spec.range_noise) + p.radius
        az_w, el_w = az_q + math.radians(spec.heading_deg) + yaw, el_q
        sigma = spec.range_noise + rng_m * zw_h / math.sqrt(12)  # range noise + zone width
    else:
        px_per_rad = spec.width_px / math.radians(spec.fov_h_deg)
        theta = 2 * math.atan(p.radius / d)
        px = theta * px_per_rad + rng.normal(0, spec.angle_noise_px)
        if px < spec.min_px:
            return None
        rng_m = (spec.size_prior / 2) / math.tan(max(px / px_per_rad, 1e-4) / 2)  # range from apparent size
        noise = spec.angle_noise_px / px_per_rad
        az_w = az + math.radians(spec.heading_deg) + yaw + rng.normal(0, noise)
        el_w = el + rng.normal(0, noise)
        sigma = rng_m * (spec.angle_noise_px / max(px, 1e-3) + 0.15)  # size-from-pixels + unknown true size
    vec = rng_m * np.array([math.cos(el_w) * math.cos(az_w), math.cos(el_w) * math.sin(az_w), math.sin(el_w)])
    return vec, sigma


@dataclass
class ReflexConfig:
    horizon: float = 0.7  # s ahead to look for a collision
    margin: float = 0.18  # m extra clearance wanted
    escape_time: float = 0.45  # s of full-effort escape
    escape_speed: float = 2.5  # m/s
    dodge_tilt_deg: float = 40.0
    min_altitude: float = 0.35
    latency: float = 0.01  # s decision + actuation path (onboard); ~0.15 via a WiFi laptop
    # What the airframe can do (set from the vehicle by ``from_vehicle``):
    accel_up: float = 6.9  # m/s^2 climbing (thrust/weight with battery-sag margin, minus 1 g)
    accel_down: float = 9.3  # m/s^2 dropping (thrust cut to ~5 %)
    accel_side: float = 8.2  # m/s^2 sideways at the dodge tilt
    lag_vertical: float = 0.06  # s for motors to change thrust
    lag_side: float = 0.14  # s: motors + tilting the airframe first

    @classmethod
    def from_vehicle(cls, vehicle, **kw) -> "ReflexConfig":
        c = cls(**kw)
        twr = 4.0 * vehicle.max_motor_thrust / (vehicle.mass * 9.81)
        usable = 0.85 * twr  # LiPo voltage sag and control headroom
        c.accel_up = max(0.0, usable - 1.0) * 9.81
        tilt = min(math.radians(c.dodge_tilt_deg), math.acos(min(1.0, 1.0 / max(usable, 1.0))))
        c.accel_side = 9.81 * math.tan(tilt)
        c.lag_vertical = 2.0 * vehicle.motor_time_constant + c.latency
        c.lag_side = c.lag_vertical + 0.08
        return c


def escape_travel(accel: float, vmax: float, t: np.ndarray, lag: float) -> np.ndarray:
    """Distance covered after ``t`` s when accelerating at ``accel`` (after ``lag``) up to ``vmax``."""
    tau = np.maximum(t - lag, 0.0)
    if accel <= 1e-6:
        return np.zeros_like(tau)
    t_sat = vmax / accel
    return np.where(tau < t_sat, 0.5 * accel * tau * tau, 0.5 * vmax * t_sat + vmax * (tau - t_sat))


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


@dataclass
class ThreatTrack:
    """Recent measurements of one object, fitted with a ballistic model (gravity known)."""

    t: float
    rel: np.ndarray
    vel: np.ndarray | None = None
    n: int = 1
    history: list = field(default_factory=list)  # (t, rel, sigma)

    def fit(self, window: float = 0.35) -> None:
        h = [m for m in self.history if self.t - m[0] <= window][-10:]
        self.n = len(h)
        if len(h) < 3:
            return
        ts = np.array([m[0] - self.t for m in h])
        w = np.array([1.0 / max(m[2], 1e-3) ** 2 for m in h])
        y = np.array([m[1] - 0.5 * G * dt * dt for m, dt in zip(h, ts)])  # remove the known gravity term
        a = np.stack([np.ones_like(ts), ts], axis=1)
        aw = a * w[:, None]
        coef = np.linalg.solve(a.T @ aw + 1e-9 * np.eye(2), aw.T @ y)  # weighted least squares, all axes at once
        self.rel, self.vel = coef[0], coef[1]


class DodgeReflex:
    """Track the threat, predict closest approach (with gravity), escape if it would hit."""

    def __init__(self, cfg: ReflexConfig, body_radius: float, ceiling: float):
        self.cfg, self.body_radius, self.ceiling = cfg, body_radius, ceiling
        self.track: ThreatTrack | None = None
        self.escape_dir = np.zeros(3)
        self.triggers = 0

    def reset(self) -> None:
        self.track = None

    @staticmethod
    def _propagate(tr: ThreatTrack, t: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Ballistic prediction of where the tracked object is at time ``t``."""
        if tr.vel is None:
            return tr.rel, None
        dt = t - tr.t
        return tr.rel + tr.vel * dt + 0.5 * G * dt * dt, tr.vel + G * dt

    def measure(self, t: float, rel: np.ndarray, sigma: float = 0.1) -> None:
        tr = self.track
        if tr is not None and tr.vel is not None:
            predicted, _ = self._propagate(tr, t)
            gate = 0.6
        else:
            predicted, gate = (None if tr is None else tr.history[-1][1]), 1.0
        if tr is None or t - tr.t > 1.0 or np.linalg.norm(rel - predicted) > gate + 2 * sigma:
            self.track = ThreatTrack(t, rel, history=[(t, rel, sigma)])
            return
        tr.history.append((t, rel, sigma))
        tr.t = t
        tr.fit()

    def check(self, t: float, altitude: float) -> bool:
        """True (and ``escape_dir`` set) if the tracked object is on course to hit us."""
        tr, c = self.track, self.cfg
        # A track stays useful after the object leaves the sensors' view (e.g. a lob dropping
        # in from above): its ballistic arc is predictable from the measurements on the way up.
        if tr is None or tr.vel is None or tr.n < 3 or t - tr.t > 1.0:
            return False
        rel_now, vel_now = self._propagate(tr, t)
        ts = np.linspace(0.0, c.horizon, 71)
        traj = rel_now[None, :] + vel_now[None, :] * ts[:, None] + 0.5 * G[None, :] * ts[:, None] ** 2
        dist = np.linalg.norm(traj, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > self.body_radius + c.margin or (i == 0 and np.dot(rel_now, vel_now) > 0):
            return False  # it will miss, or it is already moving away
        closest = traj[i]  # where the ball will be, relative to us, at closest approach
        v_dir = vel_now / max(float(np.linalg.norm(vel_now)), 1e-6)
        perp = -closest - np.dot(-closest, v_dir) * v_dir  # away from the ball's line
        side = _unit(np.cross(v_dir, [0.0, 0.0, 1.0]))
        cands = [np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])]
        if side is not None:
            cands += [side, -side]
        p = _unit(perp)
        if p is not None:
            cands += [p, _unit(p + [0, 0, -1.0]), _unit(p + [0, 0, 1.0])]
        cands = [d for d in cands if d is not None]
        # Model-predictive choice: simulate each escape with what the airframe can really do
        # (thrust, tilt, motor lag, top speed), keep it between floor and ceiling, and pick the
        # one that leaves the biggest predicted miss distance. Dropping wins ties: it needs no
        # spare thrust, which a sagging battery may not have.
        best_dir, best_score = None, -1.0
        for d in cands:
            miss = self.predicted_miss(d, traj, ts, altitude)
            if miss is None:
                continue
            score = miss + (0.01 if d[2] < -0.5 else 0.0)
            if score > best_score:
                best_dir, best_score = d, score
        if best_dir is None:
            return False
        self.escape_dir = best_dir
        self.triggers += 1
        return True

    def predicted_miss(self, d: np.ndarray, traj: np.ndarray, ts: np.ndarray, altitude: float) -> float | None:
        """Closest distance to the ball if we escape along unit vector ``d`` (None: hits floor/ceiling)."""
        c = self.cfg
        dz = float(d[2])
        a_v = c.accel_up if dz > 0 else c.accel_down
        inv = (float(np.linalg.norm(d[:2])) / max(c.accel_side, 1e-6)) ** 2 + (abs(dz) / max(a_v, 1e-6)) ** 2
        accel = 1.0 / math.sqrt(inv) if inv > 0 else 0.0
        lag = c.lag_vertical if abs(dz) > 0.9 else c.lag_side
        final_z = altitude + dz * float(escape_travel(accel, c.escape_speed, np.array([c.escape_time]), lag)[0])
        if final_z < c.min_altitude - 0.05 or final_z > self.ceiling:
            return None
        travel = escape_travel(accel, c.escape_speed, ts, lag)
        return float(np.min(np.linalg.norm(traj - travel[:, None] * d[None, :], axis=1)))

    def setpoint(self) -> Setpoint:
        d = self.escape_dir
        return Setpoint(position=np.full(3, np.nan), velocity=self.cfg.escape_speed * d, acceleration=9.0 * d, yaw=0.0)


@dataclass
class ThrowResult:
    speed: float
    hits: int
    throws: int
    detected: int
    min_clearance: list[float] = field(default_factory=list)

    @property
    def dodge_rate(self) -> float:
        return 1.0 - self.hits / max(self.throws, 1)


def run_throws(cfg: SkyMindConfig, sensors: list[ThreatSensorSpec], speed: float, throws: int = 15,
               reflex: ReflexConfig | None = None, sector_deg: float = 90.0, ball_radius: float = 0.04,
               distance: float = 2.5, seed: int = 0, hover_alt: float = 1.1) -> ThrowResult:
    """Hover, throw ``throws`` balls at ``speed`` m/s from within ``sector_deg`` of the nose.

    Throw ``k`` is identical for every sensor setup (its own random stream), so setups are
    compared on exactly the same throws. ``baseline_hits`` counts throws that hit with no reflex."""
    rng = np.random.default_rng(seed + 1000)
    reflex_cfg = reflex or ReflexConfig()
    reflex_cfg = ReflexConfig.from_vehicle(cfg.vehicle, **{k: getattr(reflex_cfg, k) for k in (
        "horizon", "margin", "escape_time", "escape_speed", "dodge_tilt_deg", "min_altitude", "latency")})
    dt = 1.0 / cfg.sim.physics_hz
    hits = detected = 0
    clearances = []
    hover = np.array([0.0, 0.0, hover_alt])
    for k in range(throws):
        sim = QuadrotorSim(cfg.vehicle, cfg.sensors, rng)
        sim.reset(position=hover)
        est = StateEstimator(cfg.estimator)
        est.reset(hover, sim.state.quaternion)
        fc = FlightController(cfg.vehicle, cfg.control)
        normal_tilt = fc.max_tilt
        rfx = DodgeReflex(reflex_cfg, cfg.vehicle.body_radius, cfg.mission.max_altitude)
        hold_sp = Setpoint(position=hover.copy())
        next_sense = [0.0] * len(sensors)
        pending: list[tuple[float, np.ndarray]] = []
        p: Projectile | None = None
        throw_at = 1.0  # let the hover settle first
        escape_start = escape_end = -1.0
        seen = False
        t, step = 0.0, 0
        while t < throw_at + 2.0:
            pos = sim.state.position
            if p is None and t >= throw_at:
                trng = np.random.default_rng([seed, k])
                az = math.radians(trng.uniform(-sector_deg / 2, sector_deg / 2))
                start = hover + distance * np.array([math.cos(az), math.sin(az), 0.0])
                start[2] = trng.uniform(0.9, 1.4)
                target = pos + trng.normal(0, 0.02, 3)  # aimed at where the drone is right now
                horiz = target[:2] - start[:2]
                tf = float(np.linalg.norm(horiz)) / speed
                v = np.array([horiz[0] / tf, horiz[1] / tf, (target[2] - start[2]) / tf - 0.5 * G[2] * tf])
                p = Projectile(start, v, ball_radius)
            if p is not None and p.active:
                for i, spec in enumerate(sensors):
                    if t >= next_sense[i]:
                        next_sense[i] = t + 1.0 / spec.rate_hz
                        m = sense(spec, pos, sim.state.euler[2], p, rng)
                        if m is not None:
                            pending.append((t + spec.latency, m))
                            seen = True
                ready = sorted((m for m in pending if m[0] <= t), key=lambda x: x[0])
                pending = [m for m in pending if m[0] > t]
                for tm, (m, sigma) in ready:
                    rfx.measure(tm, m, sigma)
                if t >= escape_end and rfx.check(t, pos[2]):
                    escape_start = t + reflex_cfg.latency  # decision -> motors
                    escape_end = escape_start + reflex_cfg.escape_time
            escaping = escape_start <= t < escape_end
            fc.max_tilt = math.radians(reflex_cfg.dodge_tilt_deg) if escaping else normal_tilt
            fc.min_thrust_frac = 0.05 if escaping else 0.3
            if escaping:
                sp = rfx.setpoint()
            else:
                if escape_end > 0 and t >= escape_end:  # escape over: hold wherever we ended up
                    hold_sp = Setpoint(position=np.array([pos[0], pos[1], max(pos[2], reflex_cfg.min_altitude)]))
                    escape_start = escape_end = -1.0
                sp = hold_sp
            imu = sim.imu()
            est.update_imu(imu.gyro, imu.accel, dt, sim.heading_measurement())
            if step % 12 == 0:
                est.update_position(sim.position_measurement(), 12 * dt)
            sim.step(fc.update(est.estimate(), sp, dt).motors, dt)
            if p is not None and p.active:
                p.step(dt)
                p.min_dist = min(p.min_dist, float(np.linalg.norm(p.pos - sim.state.position)))
            t += dt
            step += 1
        clear = p.min_dist - p.radius - cfg.vehicle.body_radius
        clearances.append(round(clear, 3))
        hits += clear < 0
        detected += seen
    return ThrowResult(speed, hits, throws, detected, clearances)


class LiveDodge:
    """Thrown balls plus the onboard reflex inside a running simulation (the app's 'throw a ball' test)."""

    BALL_MASS = 0.005  # kg, crumpled paper

    def __init__(self, cfg: SkyMindConfig, rng: np.random.Generator, setup: str = "tof8x8+looming_wide"):
        self.cfg, self.rng = cfg, rng
        self.specs = SENSORS[setup]
        self.setup = setup
        self.rcfg = ReflexConfig.from_vehicle(cfg.vehicle)
        self.reflex = DodgeReflex(self.rcfg, cfg.vehicle.body_radius, cfg.mission.max_altitude)
        self.balls: list[Projectile] = []
        self.next_sense = [0.0] * len(self.specs)
        self.pending: list[tuple[float, tuple[np.ndarray, float]]] = []
        self.escape_start = self.escape_end = -1.0
        self.enabled = True
        self.stats = {"thrown": 0, "dodged": 0, "hit": 0, "missed_anyway": 0}

    def throw(self, drone_pos: np.ndarray, yaw: float, speed: float = 5.0, distance: float = 2.5,
              sector_deg: float = 90.0, radius: float = 0.04) -> Projectile:
        az = yaw + math.radians(self.rng.uniform(-sector_deg / 2, sector_deg / 2))
        start = drone_pos + distance * np.array([math.cos(az), math.sin(az), 0.0])
        start[2] = max(0.3, drone_pos[2] + self.rng.uniform(-0.2, 0.3))
        target = drone_pos + self.rng.normal(0, 0.02, 3)
        horiz = target[:2] - start[:2]
        tf = float(np.linalg.norm(horiz)) / speed
        v = np.array([horiz[0] / tf, horiz[1] / tf, (target[2] - start[2]) / tf - 0.5 * G[2] * tf])
        p = Projectile(start, v, radius)
        p.triggered = False  # type: ignore[attr-defined]
        p.reported = False  # type: ignore[attr-defined]
        self.balls.append(p)
        self.stats["thrown"] += 1
        return p

    def escaping(self, t: float) -> bool:
        return self.escape_start <= t < self.escape_end

    def setpoint(self) -> Setpoint:
        return self.reflex.setpoint()

    def step(self, t: float, dt: float, sim: QuadrotorSim, body_radius: float) -> list[str]:
        """Advance balls, run the threat sensors and the reflex; returns events to report."""
        events: list[str] = []
        pos, yaw = sim.state.position, sim.state.euler[2]
        for i, spec in enumerate(self.specs):
            if t >= self.next_sense[i]:
                self.next_sense[i] = t + 1.0 / spec.rate_hz
                for b in self.balls:
                    if b.active:
                        m = sense(spec, pos, yaw, b, self.rng)
                        if m is not None:
                            self.pending.append((t + spec.latency, m))
        ready = sorted((m for m in self.pending if m[0] <= t), key=lambda x: x[0])
        self.pending = [m for m in self.pending if m[0] > t]
        for tm, (rel, sigma) in ready:
            self.reflex.measure(tm, rel, sigma)
        if self.enabled and t >= self.escape_end and self.reflex.check(t, float(pos[2])):
            self.escape_start = t + self.rcfg.latency
            self.escape_end = self.escape_start + self.rcfg.escape_time
            d = self.reflex.escape_dir
            word = "dropping" if d[2] < -0.7 else "climbing" if d[2] > 0.7 else "sidestepping"
            events.append(f"DODGE: threat detected, {word}")
            for b in self.balls:
                if b.active:
                    b.triggered = True  # type: ignore[attr-defined]
        for b in self.balls:
            if not b.active:
                continue
            b.step(dt)
            dist = float(np.linalg.norm(b.pos - pos))
            if dist < b.min_dist:
                b.min_dist = dist
            gap = b.min_dist - b.radius - body_radius
            passed = dist > b.min_dist + 0.6 or not b.active
            if gap < 0 and not b.reported:  # hit: a light paper ball shoves the drone, it recovers
                b.reported = True  # type: ignore[attr-defined]
                b.active = False
                rel_v = b.vel - sim.state.velocity
                sim.state.velocity = sim.state.velocity + (self.BALL_MASS / self.cfg.vehicle.mass) * rel_v
                sim.state.omega = sim.state.omega + self.rng.normal(0, 2.0, 3)
                self.stats["hit"] += 1
                events.append("HIT by the ball: recovering balance")
            elif passed and not b.reported:
                b.reported = True  # type: ignore[attr-defined]
                b.active = False
                key = "dodged" if b.triggered else "missed_anyway"  # type: ignore[attr-defined]
                self.stats[key] += 1
                events.append(f"ball {'dodged' if b.triggered else 'missed'}: cleared by {gap * 100:.0f} cm")
        self.balls = [b for b in self.balls if b.active or not b.reported]
        return events
