"""Mission executive: ties flight control, avoidance, vision and recording together.

The same :class:`FlightBrain` decision logic is used in simulation and (via
the MAVLink bridge) on a real vehicle; :class:`SimulationRunner` closes the
loop with the physics model, simulated sensors and the synthetic camera.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from skymind.config import SkyMindConfig
from skymind.control.estimator import StateEstimate, StateEstimator
from skymind.control.flight_controller import FlightController, Setpoint
from skymind.math3d import wrap_angle
from skymind.perception.avoidance import AvoidanceCommand, AvoidMode, ObstacleAvoider, braking_speed
from skymind.perception.mapping import GlobalPlanner
from skymind.perception.range_sensors import map_hit_weights
from skymind.power import PowerModel
from skymind.sim.camera_sim import SimCamera
from skymind.sim.lidar import SimLidar
from skymind.sim.physics import QuadrotorSim
from skymind.sim.world import World
from skymind.vision.analyzer import FrameAnalysis, FrameAnalyzer, draw_overlay
from skymind.vision.recorder import FlightRecorder


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v


class FlightMode(str, Enum):
    PREFLIGHT = "preflight"
    TAKEOFF = "takeoff"
    MISSION = "mission"
    HOLD = "hold"  # interactive: hover in place and wait for the next command
    RTL = "rtl"
    LAND = "land"
    LANDED = "landed"
    CRASHED = "crashed"


@dataclass
class Battery:
    capacity_wh: float
    model: PowerModel
    energy_wh: float = field(init=False)
    last_power: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.energy_wh = self.capacity_wh

    def drain(self, motor_thrust: np.ndarray, dt: float) -> float:
        self.last_power = self.model.electrical_power(motor_thrust)
        self.energy_wh = max(0.0, self.energy_wh - self.last_power * dt / 3600.0)
        return self.last_power

    @property
    def fraction(self) -> float:
        return self.energy_wh / self.capacity_wh


@dataclass
class BrainOutput:
    setpoint: Setpoint
    avoidance: AvoidanceCommand | None
    events: list[str]


class FlightBrain:
    """High-level autonomy at ~20 Hz: mode logic, waypoints, avoidance, failsafes."""

    def __init__(self, cfg: SkyMindConfig, max_range: float):
        self.cfg = cfg
        m = cfg.mission
        self.mode = FlightMode.PREFLIGHT
        self.home = np.array(m.home, dtype=float)
        self.waypoints = [np.array(w[:2], dtype=float) for w in m.waypoints]
        self.wp_index = 0
        self.altitude = m.cruise_altitude
        self.avoider = ObstacleAvoider(cfg.avoidance, max_range)
        self.yaw_sp = 0.0
        self.blocked_time = 0.0
        self.land_xy: np.ndarray | None = None
        self.failsafe: str | None = None
        self.vel_cmd = np.zeros(2)
        self.acc_ff = np.zeros(2)
        self.cruise_speed = m.cruise_speed
        self.best_goal_dist = float("inf")
        self.no_progress_time = 0.0
        a = cfg.avoidance
        self.planner = GlobalPlanner(m.geofence, a.safety_radius, a.map_resolution) if a.use_global_planner else None
        self.map_weights = map_hit_weights(cfg.sensors)
        self.replan_timer = 0.0
        self.planned_goal: np.ndarray | None = None
        self.local_goal: np.ndarray | None = None
        # Interactive mode (live app / voice assistant): after takeoff or reaching a goal the
        # drone hovers (HOLD) and waits for the next command instead of flying a fixed mission.
        self.interactive = False
        self.hold_xy = self.home.copy()
        self.yaw_target = 0.0
        self.pending_events: list[str] = []

    @property
    def goal(self) -> np.ndarray:
        if self.mode is FlightMode.RTL:
            return self.home
        if not self.waypoints:
            return self.hold_xy
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)]

    @property
    def airborne(self) -> bool:
        return self.mode in (FlightMode.TAKEOFF, FlightMode.MISSION, FlightMode.HOLD, FlightMode.RTL, FlightMode.LAND)

    def start(self) -> None:
        self.mode = FlightMode.TAKEOFF

    def _enter(self, mode: FlightMode, events: list[str], why: str) -> None:
        if mode is not self.mode:
            events.append(f"{self.mode.value}->{mode.value}: {why}")
            self.mode = mode
            self.avoider.reset()
            self.blocked_time = 0.0
            self.best_goal_dist = float("inf")
            self.no_progress_time = 0.0
            if mode is FlightMode.LAND:
                self.land_xy = None
            if mode in (FlightMode.HOLD, FlightMode.LANDED):
                self.vel_cmd = np.zeros(2)
                self.acc_ff = np.zeros(2)

    # ---------------------------------------------------------- interactive commands
    # Called between update() ticks (the live session serialises them on its own thread).
    def command_takeoff(self, position: np.ndarray, altitude: float) -> bool:
        if self.mode not in (FlightMode.PREFLIGHT, FlightMode.LANDED):
            return False
        self.home = np.asarray(position[:2], dtype=float).copy()
        self.hold_xy = self.home.copy()
        self.altitude = float(np.clip(altitude, 0.3, self.cfg.mission.max_altitude))
        self.failsafe = None
        self._enter(FlightMode.TAKEOFF, self.pending_events, "takeoff command")
        return True

    def command_goto(self, xy: np.ndarray) -> bool:
        if self.mode not in (FlightMode.MISSION, FlightMode.HOLD):
            return False
        self.waypoints = [np.asarray(xy, dtype=float)[:2].copy()]
        self.wp_index = 0
        self.planned_goal = None
        if self.mode is FlightMode.MISSION:
            self.avoider.reset()
            self.best_goal_dist = float("inf")
            self.no_progress_time = 0.0
        self._enter(FlightMode.MISSION, self.pending_events, "go-to command")
        return True

    def command_hold(self, position: np.ndarray) -> bool:
        if self.mode not in (FlightMode.MISSION, FlightMode.HOLD, FlightMode.RTL, FlightMode.TAKEOFF):
            return False
        self.hold_xy = np.asarray(position[:2], dtype=float).copy()
        self.yaw_target = self.yaw_sp
        self._enter(FlightMode.HOLD, self.pending_events, "hold command")
        return True

    def command_rtl(self) -> bool:
        if self.mode not in (FlightMode.MISSION, FlightMode.HOLD):
            return False
        self._enter(FlightMode.RTL, self.pending_events, "return-home command")
        return True

    def command_land(self) -> bool:
        if self.mode not in (FlightMode.TAKEOFF, FlightMode.MISSION, FlightMode.HOLD, FlightMode.RTL):
            return False
        self._enter(FlightMode.LAND, self.pending_events, "land command")
        return True

    def command_yaw(self, yaw: float) -> bool:
        if self.mode is not FlightMode.HOLD:
            return False
        self.yaw_target = float(wrap_angle(yaw))
        return True

    def command_altitude(self, altitude: float) -> bool:
        if self.mode not in (FlightMode.MISSION, FlightMode.HOLD):
            return False
        self.altitude = float(np.clip(altitude, 0.3, self.cfg.mission.max_altitude))
        return True

    def _check_failsafes(self, est: StateEstimate, battery: float, t: float, events: list[str]) -> None:
        m = self.cfg.mission
        flying = self.mode in (FlightMode.TAKEOFF, FlightMode.MISSION, FlightMode.HOLD)
        x, y = est.position[:2]
        xmin, ymin, xmax, ymax = m.geofence
        if flying and battery < self.cfg.battery.rtl_threshold:
            self.failsafe = "low battery"
            self._enter(FlightMode.RTL, events, "battery below threshold")
        elif flying and not (xmin <= x <= xmax and ymin <= y <= ymax):
            self.failsafe = "geofence"
            self._enter(FlightMode.RTL, events, "geofence breach")
        elif self.mode in (FlightMode.TAKEOFF, FlightMode.MISSION, FlightMode.HOLD, FlightMode.RTL) and t > m.max_duration:
            self.failsafe = "timeout"
            self._enter(FlightMode.LAND, events, "mission time limit")

    def fence_ranges(self, position: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each beam to the geofence, so the avoider treats it as a wall."""
        xmin, ymin, xmax, ymax = self.cfg.mission.geofence
        c, s = np.cos(angles), np.sin(angles)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(c > 1e-9, (xmax - position[0]) / c, np.where(c < -1e-9, (xmin - position[0]) / c, np.inf))
            ty = np.where(s > 1e-9, (ymax - position[1]) / s, np.where(s < -1e-9, (ymin - position[1]) / s, np.inf))
        return np.clip(np.minimum(tx, ty), 0.0, max_range)

    def _observed_headings(self, yaw: float) -> np.ndarray:
        """Which avoider headings fall inside a range-sensor cone right now."""
        sens = self.cfg.sensors
        heads = yaw + np.radians(np.asarray(sens.tof_headings_deg, dtype=float))
        half = math.radians(sens.tof_fov_deg) / 2.0
        diff = np.abs(wrap_angle(self.avoider.headings[:, None] - heads[None, :]))
        return (diff <= half).any(axis=1)

    def _govern(self, cmd: np.ndarray, vel: np.ndarray, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Safety governor, applied after smoothing and never slew-limited.

        The avoider limits speed along the heading it *chooses*, but the vehicle's
        momentum may still point somewhere else. Along both the command and the
        actual velocity, cap the speed to what can be shed before ``stop_distance``.
        """
        a = self.cfg.avoidance
        out = cmd.copy()
        for d in (cmd, vel):
            speed = float(np.linalg.norm(d))
            if speed < 0.2:
                continue
            u = d / speed
            clear = float(self.avoider.corridor_clearance(ranges, angles, np.array([math.atan2(u[1], u[0])]))[0])
            vmax = braking_speed(clear - a.stop_distance, a.max_decel, a.reaction_time)
            along = float(out @ u)
            if along > vmax:
                out = out - (along - vmax) * u
        return out

    def update(self, est: StateEstimate, ranges: np.ndarray, angles: np.ndarray, battery: float, t: float, dt: float, landed: bool) -> BrainOutput:
        m = self.cfg.mission
        events: list[str] = self.pending_events
        self.pending_events = []
        self._check_failsafes(est, battery, t, events)
        pos = est.position
        avoid: AvoidanceCommand | None = None
        sp = Setpoint(yaw=self.yaw_sp)

        if self.mode is FlightMode.TAKEOFF:
            sp.position = np.array([self.home[0], self.home[1], self.altitude])
            if pos[2] > self.altitude - 0.25:
                if self.interactive:
                    self.hold_xy = pos[:2].copy()
                    self.yaw_target = self.yaw_sp
                    self._enter(FlightMode.HOLD, events, "takeoff altitude reached")
                else:
                    self._enter(FlightMode.MISSION, events, "takeoff altitude reached")

        if self.mode in (FlightMode.MISSION, FlightMode.RTL):
            to_goal = self.goal - pos[:2]
            if np.linalg.norm(to_goal) < m.waypoint_radius:
                if self.mode is FlightMode.RTL:
                    self._enter(FlightMode.LAND, events, "home reached")
                else:
                    events.append(f"waypoint {self.wp_index} reached")
                    self.wp_index += 1
                    if self.wp_index >= len(self.waypoints):
                        if self.interactive:
                            self.hold_xy = pos[:2].copy()
                            self.yaw_target = self.yaw_sp
                            self._enter(FlightMode.HOLD, events, "goal reached")
                        else:
                            self._enter(FlightMode.RTL, events, "mission complete")
                to_goal = self.goal - pos[:2]
        if self.mode in (FlightMode.MISSION, FlightMode.RTL):
            dist = float(np.linalg.norm(to_goal))
            steer = to_goal
            if self.planner is not None:
                self.planner.grid.update(pos, ranges, angles, self.avoider.max_range, self.map_weights)
                self.replan_timer -= dt
                goal_changed = self.planned_goal is None or not np.array_equal(self.planned_goal, self.goal)
                if goal_changed or self.replan_timer <= 0.0:
                    self.planner.plan(pos, self.goal)
                    self.planned_goal = self.goal.copy()
                    self.replan_timer = self.cfg.avoidance.replan_interval
                self.local_goal = self.planner.local_goal(pos, self.cfg.avoidance.path_lookahead)
                if self.local_goal is not None and self.planner.path:
                    steer = self.local_goal - pos[:2]
                    # Progress is measured along the planned path, so detours do not look like being stuck.
                    path = np.array(self.planner.path)
                    i0 = int(np.argmin(np.linalg.norm(path - pos[:2], axis=1)))
                    dist = float(np.linalg.norm(path[i0] - pos[:2]) + np.sum(np.linalg.norm(np.diff(path[i0:], axis=0), axis=1)))
                    # Keep full speed until the local goal is the final goal.
                    if np.linalg.norm(self.local_goal - self.goal) > 0.5:
                        steer = steer / max(np.linalg.norm(steer), 1e-6) * max(np.linalg.norm(steer), m.cruise_speed * 2.0)
            if self.failsafe != "geofence":  # when already outside, let RTL fly back in
                ranges = np.minimum(ranges, self.fence_ranges(pos, angles, self.avoider.max_range))
            observed = None
            turning = False
            a = self.cfg.avoidance
            if a.max_heading_offset_deg is not None:
                observed = self._observed_headings(est.euler[2])
                steer_heading = math.atan2(steer[1], steer[0])
                # Turn-then-go: target out of the sensors' view -> rotate in place first.
                turning = abs(float(wrap_angle(steer_heading - est.euler[2]))) > math.radians(a.max_heading_offset_deg)
            avoid = self.avoider.plan(ranges, angles, steer, 0.0 if turning else self.cruise_speed, observed)
            # Stuck = no safe heading, or no progress toward the goal for a while (e.g. sliding
            # along a long wall). Either way, try climbing over the obstacle.
            if dist < self.best_goal_dist - 1.0:
                self.best_goal_dist = dist
                self.no_progress_time = 0.0
            else:
                self.no_progress_time += dt
            self.blocked_time = self.blocked_time + dt if avoid.mode is AvoidMode.BLOCKED else 0.0
            stuck = self.blocked_time > m.blocked_timeout or self.no_progress_time > 2.0 * m.blocked_timeout
            if stuck and self.altitude + 2.0 <= m.max_altitude:
                self.altitude += 2.0
                self.blocked_time = self.no_progress_time = 0.0
                self.best_goal_dist = dist
                if self.planner is not None:
                    self.planner.reset()  # obstacles lower than the new altitude no longer matter
                    self.replan_timer = 0.0
                events.append(f"blocked: climbing to {self.altitude:.1f} m")
            desired = avoid.velocity_xy
            if a.max_heading_offset_deg is not None and np.linalg.norm(desired) > 0.2:
                # Sparse sensors see best straight ahead: crawl until the nose points where we go.
                off = abs(float(wrap_angle(math.atan2(desired[1], desired[0]) - est.euler[2])))
                lim = math.radians(a.max_heading_offset_deg)
                if off > lim:
                    desired = desired * max(0.15, 1.0 - (off - lim) / (math.pi / 2))
            # Jerk-limited reference model; allow harder manoeuvres when something is close.
            urgent = avoid.mode is AvoidMode.BLOCKED or avoid.min_range < a.stop_distance + 0.5
            a_max = a.max_accel * (2.0 if urgent else 1.0)
            j_max = a.max_jerk * (3.0 if urgent else 1.0)
            acc_des = _clip_norm(a.ref_gain * (desired - self.vel_cmd), a_max)
            self.acc_ff = self.acc_ff + _clip_norm(acc_des - self.acc_ff, j_max * dt)
            prev_cmd = self.vel_cmd
            self.vel_cmd = self._govern(prev_cmd + self.acc_ff * dt, est.velocity[:2], ranges, angles)
            if not np.allclose(self.vel_cmd, prev_cmd + self.acc_ff * dt):
                # The governor cut speed: brake now rather than through the jerk limit.
                self.acc_ff = _clip_norm((self.vel_cmd - prev_cmd) / dt, 2.0 * a.max_accel)
            sp.position = np.array([np.nan, np.nan, self.altitude])
            sp.velocity = np.array([self.vel_cmd[0], self.vel_cmd[1], 0.0])
            sp.acceleration = np.array([self.acc_ff[0], self.acc_ff[1], 0.0])
            # Point the camera (and, with a ToF ring, the best sensors) where we're going.
            if turning:
                face = steer
            elif a.max_heading_offset_deg is not None and avoid.mode is AvoidMode.BLOCKED:
                # Sensors only look one way: rotate to search for an opening.
                look = est.euler[2] + math.radians(100.0)
                face = np.array([math.cos(look), math.sin(look)])
            elif a.max_heading_offset_deg is not None:
                face = avoid.velocity_xy
            else:
                face = self.vel_cmd
            if float(np.linalg.norm(face)) > 0.2:
                target_yaw = math.atan2(face[1], face[0])
                err = float(wrap_angle(target_yaw - self.yaw_sp))
                step = m.max_yaw_rate * dt
                self.yaw_sp = float(wrap_angle(self.yaw_sp + np.clip(err, -step, step)))
            sp.yaw = self.yaw_sp

        if self.mode is FlightMode.HOLD:
            sp.position = np.array([self.hold_xy[0], self.hold_xy[1], self.altitude])
            err = float(wrap_angle(self.yaw_target - self.yaw_sp))
            step = m.max_yaw_rate * dt
            self.yaw_sp = float(wrap_angle(self.yaw_sp + np.clip(err, -step, step)))
            sp.yaw = self.yaw_sp
            if self.planner is not None:  # keep mapping while hovering / looking around
                self.planner.grid.update(pos, ranges, angles, self.avoider.max_range, self.map_weights)

        if self.mode is FlightMode.LAND:
            if self.land_xy is None:
                self.land_xy = pos[:2].copy()
                self.vel_cmd = np.zeros(2)
                self.acc_ff = np.zeros(2)
            sp.position = np.array([self.land_xy[0], self.land_xy[1], np.nan])
            sp.velocity = np.array([0.0, 0.0, -0.7 if pos[2] > 1.0 else -0.35])
            if landed:
                self._enter(FlightMode.LANDED, events, "touchdown")

        return BrainOutput(sp, avoid, events)


@dataclass
class MissionResult:
    status: str
    failsafe: str | None
    duration: float
    waypoints_reached: int
    waypoints_total: int
    distance_flown: float
    straight_line_distance: float
    collisions: int
    min_clearance: float
    battery_used: float
    energy_wh: float
    avg_power_w: float
    energy_per_km_wh: float
    est_endurance_min: float  # at this flight's average power, down to the RTL reserve
    hover_power_w: float
    mean_tip_speed: float  # m/s, rotor-noise proxy
    frames_recorded: int
    snapshots: int
    run_dir: str
    wall_time: float
    events: list[tuple[float, str]]

    def to_dict(self) -> dict:
        return asdict(self)


class SimulationRunner:
    def __init__(self, cfg: SkyMindConfig, run_dir: str | Path, record_video: bool = True, verbose: bool = False):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.verbose = verbose
        self.rng = np.random.default_rng(cfg.sim.seed)
        self.world = World(cfg.world)
        self.sim = QuadrotorSim(cfg.vehicle, cfg.sensors, self.rng)
        self.estimator = StateEstimator(cfg.estimator)
        self.controller = FlightController(cfg.vehicle, cfg.control)
        self.lidar = SimLidar(cfg.sensors, self.world, self.rng)
        self.camera = SimCamera(cfg.camera, self.world, self.rng)
        self.analyzer = FrameAnalyzer(cfg.vision)
        self.brain = FlightBrain(cfg, cfg.sensors.lidar_max_range)
        self.power = PowerModel(cfg.vehicle)
        self.battery = Battery(cfg.battery.capacity_wh, self.power)
        self.record_video = record_video

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self) -> MissionResult:
        cfg = self.cfg
        dt = 1.0 / cfg.sim.physics_hz
        hz = cfg.sim.physics_hz
        pos_every = max(1, round(hz / cfg.sensors.pos_rate_hz))
        brain_every = max(1, round(hz / cfg.sensors.lidar_rate_hz))
        cam_every = max(1, round(hz / cfg.camera.fps))
        log_every = max(1, round(hz / 25.0))

        home = np.array([cfg.mission.home[0], cfg.mission.home[1], 0.0])
        self.sim.reset(position=home)
        self.estimator.reset(home, self.sim.state.quaternion)
        self.brain.start()

        rec = FlightRecorder(self.run_dir, cfg.camera.fps, (cfg.camera.width, cfg.camera.height), self.record_video)
        events: list[tuple[float, str]] = []
        out = self.controller.idle()
        brain_out: BrainOutput | None = None
        ranges = np.full(cfg.sensors.lidar_beams, cfg.sensors.lidar_max_range)
        angles = self.lidar.body_angles.copy()
        distance = 0.0
        collisions = 0
        in_collision = False
        min_clear = float("inf")
        last_snapshot_t = -1e9
        last_labels: set[str] = set()
        prev_pos = self.sim.state.position.copy()
        airborne_time = 0.0
        tip_speeds: list[float] = []
        cam_frame = 0
        last_analysis = FrameAnalysis()
        lag_ticks = int(round(cfg.sim.link_latency * cfg.sensors.lidar_rate_hz))
        sensed: list = []
        in_flight: list = []
        active_sp: Setpoint | None = None
        active_sp_t = 0.0
        watchdog_sp: Setpoint | None = None
        t = 0.0
        wall0 = time.time()
        hard_stop = cfg.mission.max_duration + 60.0

        try:
            k = 0
            while t < hard_stop:
                st = self.sim.state
                imu = self.sim.imu()
                self.estimator.update_imu(imu.gyro, imu.accel, dt, self.sim.heading_measurement())
                if k % pos_every == 0:
                    self.estimator.update_position(self.sim.position_measurement(), pos_every * dt)
                est = self.estimator.estimate()

                if k % brain_every == 0:
                    ranges, angles = self.lidar.scan(st.position, st.euler[2])
                    # Over a ground-station link the brain sees data that is link_latency old.
                    sensed.append((est, ranges, angles))
                    b_est, b_ranges, b_angles = sensed[0] if len(sensed) > lag_ticks else sensed[-1]
                    if len(sensed) > lag_ticks:
                        sensed.pop(0)
                    landed = self.sim.on_ground and self.brain.mode is FlightMode.LAND
                    brain_out = self.brain.update(b_est, b_ranges, b_angles, self.battery.fraction, t, brain_every * dt, landed)
                    for e in brain_out.events:
                        events.append((round(t, 2), e))
                        self._log(f"[{t:6.2f}s] {e}")
                    if self.rng.random() >= cfg.sim.link_drop_prob:
                        in_flight.append((t + cfg.sim.link_latency, brain_out.setpoint))
                while in_flight and in_flight[0][0] <= t + 1e-9:
                    active_sp = in_flight.pop(0)[1]
                    active_sp_t = t
                    watchdog_sp = None

                mode = self.brain.mode
                if mode in (FlightMode.LANDED, FlightMode.CRASHED, FlightMode.PREFLIGHT) or active_sp is None:
                    out = self.controller.idle()
                    self.controller.reset()
                else:
                    sp = active_sp
                    if t - active_sp_t > cfg.sim.link_timeout:
                        # Onboard watchdog: link silent too long -> hold a hover where we are.
                        if watchdog_sp is None:
                            watchdog_sp = Setpoint(position=est.position.copy(), yaw=est.euler[2])
                            events.append((round(t, 2), "link timeout: onboard hover hold"))
                        sp = watchdog_sp
                    out = self.controller.update(est, sp, dt)

                self.sim.step(out.motors, dt, self.world.wind_at(self.rng))
                self.battery.drain(self.sim.state.motor_thrust, dt)
                if not self.sim.on_ground:
                    airborne_time += dt
                    if k % log_every == 0:
                        tip_speeds.append(self.power.tip_speed(self.sim.state.motor_thrust))
                t += dt
                k += 1

                pos = self.sim.state.position
                distance += float(np.linalg.norm(pos[:2] - prev_pos[:2]))
                prev_pos = pos.copy()
                clear = self.world.clearance(pos)
                if pos[2] > 0.3:
                    min_clear = min(min_clear, clear)
                if clear < cfg.vehicle.body_radius:
                    if not in_collision:
                        collisions += 1
                        events.append((round(t, 2), "COLLISION"))
                        self._log(f"[{t:6.2f}s] COLLISION at {np.round(pos, 2)}")
                    in_collision = True
                    self.brain._enter(FlightMode.CRASHED, [], "collision")
                else:
                    in_collision = False

                if k % log_every == 0:
                    roll, pitch, yaw = self.sim.state.euler
                    rec.log_record(
                        {
                            "type": "state",
                            "t": t,
                            "mode": self.brain.mode.value,
                            "pos": pos,
                            "vel": self.sim.state.velocity,
                            "est_pos": est.position,
                            "est_vel": est.velocity,
                            "rpy": [roll, pitch, yaw],
                            "est_rpy": list(est.euler),
                            "rate": self.sim.state.omega,
                            "alt_sp": self.brain.altitude,
                            "vel_sp": out.velocity_target,
                            "motors": self.sim.state.motor_thrust,
                            "battery": self.battery.fraction,
                            "power": self.battery.last_power,
                            "tip_speed": self.power.tip_speed(self.sim.state.motor_thrust),
                            "min_range": float(ranges.min()),
                            "clearance": clear,
                            "avoid": brain_out.avoidance.mode.value if brain_out and brain_out.avoidance else None,
                            "wp": self.brain.wp_index,
                        }
                    )

                if k % cam_every == 0 and pos[2] > 0.2:
                    roll, pitch, yaw = self.sim.state.euler
                    frame, gt = self.camera.render(pos, yaw, roll, pitch, self.sim.state.omega)
                    analyse_now = cam_frame % max(1, cfg.vision.detect_every) == 0
                    cam_frame += 1
                    if analyse_now:
                        last_analysis = self.analyzer.analyze(frame)
                    analysis = last_analysis
                    labels = {d.label for d in analysis.detections}
                    snap = None
                    if analyse_now and cfg.vision.snapshot_on_detection and labels and (labels != last_labels or t - last_snapshot_t > 3.0):
                        snap = rec.snapshot(frame, t, list(labels))
                        last_snapshot_t = t
                    last_labels = labels
                    hud = [
                        f"t {t:5.1f}s  {self.brain.mode.value.upper()}",
                        f"alt {pos[2]:4.1f}m  v {np.linalg.norm(self.sim.state.velocity):3.1f}m/s",
                        f"bat {self.battery.fraction * 100:3.0f}%  {self.battery.last_power:3.0f}W  obst {ranges.min():3.1f}m",
                    ]
                    rec.write_frame(draw_overlay(frame, analysis, hud))
                    if analyse_now:
                        rec.log_record(
                            {
                                "type": "vision",
                                "t": t,
                                **analysis.to_dict(),
                                "ground_truth": [{"label": g.label, "bbox": g.bbox, "distance": g.distance} for g in gt],
                                "snapshot": snap,
                            }
                        )

                if self.brain.mode in (FlightMode.LANDED, FlightMode.CRASHED):
                    # Let the log capture the final state, then stop.
                    break
        finally:
            rec.close()

        status = self.brain.mode.value
        energy = self.battery.capacity_wh - self.battery.energy_wh
        avg_power = energy * 3600.0 / max(airborne_time, 1e-6)
        straight = 0.0
        pts = [self.brain.home] + self.brain.waypoints + [self.brain.home]
        for a, b in zip(pts[:-1], pts[1:]):
            straight += float(np.linalg.norm(b - a))
        return MissionResult(
            status=status,
            failsafe=self.brain.failsafe,
            duration=round(t, 2),
            waypoints_reached=min(self.brain.wp_index, len(self.brain.waypoints)),
            waypoints_total=len(self.brain.waypoints),
            distance_flown=round(distance, 2),
            straight_line_distance=round(straight, 2),
            collisions=collisions,
            min_clearance=round(min_clear, 3) if min_clear < 1e8 else -1.0,
            battery_used=round(1.0 - self.battery.fraction, 4),
            energy_wh=round(energy, 3),
            avg_power_w=round(avg_power, 1),
            energy_per_km_wh=round(energy / max(distance / 1000.0, 1e-6), 2) if distance > 1.0 else -1.0,
            est_endurance_min=round(60.0 * cfg.battery.capacity_wh * (1.0 - cfg.battery.rtl_threshold) / max(avg_power, 1e-6), 1),
            hover_power_w=round(self.power.hover_power(), 1),
            mean_tip_speed=round(float(np.mean(tip_speeds)) if tip_speeds else 0.0, 1),
            frames_recorded=rec.frames,
            snapshots=rec.snapshots,
            run_dir=str(self.run_dir),
            wall_time=round(time.time() - wall0, 2),
            events=events,
        )
