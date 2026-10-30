"""Live, command-driven drone sessions for the app and the voice assistant.

A :class:`LiveSession` keeps the drone running continuously on a background
thread. The drone can be simulated (:class:`SimLiveSession`) or real
(:class:`RealLiveSession`, ESP-Drone or MAVLink). Commands can arrive at any time
from the web app, the assistant or the buttons:

    s = SimLiveSession(cfg, "runs/live")
    s.start()
    s.takeoff()
    s.go_to((4.0, 0.7))
    s.wait_for(lambda st: st["mode"] == "hold", timeout=30)

Commands go into a queue and are executed on the session thread between brain
ticks, so the flight logic never sees half-applied state. The brain's avoidance,
geofence, battery and link failsafes still decide what is safe: a command can ask
for something, but it can never switch the safety layer off.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from skymind.config import SkyMindConfig
from skymind.control.estimator import StateEstimate, StateEstimator
from skymind.control.flight_controller import FlightController, Setpoint
from skymind.math3d import quat_from_euler, wrap_angle
from skymind.mission import Battery, FlightBrain, FlightMode
from skymind.power import PowerModel
from skymind.vision.analyzer import FrameAnalysis, FrameAnalyzer, draw_overlay
from skymind.vision.recorder import FlightRecorder


class LiveSession:
    """Backend-independent part: command queue, brain, vision, recording, state snapshot."""

    def __init__(self, cfg: SkyMindConfig, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.brain = self._new_brain()
        self.analyzer = FrameAnalyzer(cfg.vision)
        self.recorder = FlightRecorder(self.run_dir, cfg.camera.fps, (cfg.camera.width, cfg.camera.height))
        cam = cfg.camera
        self.focal_px = (cam.width / 2.0) / math.tan(math.radians(cam.fov_deg) / 2.0)
        self._cmds: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False
        self.t = 0.0
        self.events: list[tuple[float, str]] = []
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_analysis = FrameAnalysis()
        self.latest_detections: list[dict] = []
        self.state: dict[str, Any] = {"mode": FlightMode.PREFLIGHT.value, "t": 0.0}
        self.error: str | None = None
        self.ranges = np.zeros(0)
        self.collisions = 0
        self._last_labels: set[str] = set()
        self._last_snap_t = -1e9
        self._cam_frames = 0
        # Room-scan inputs: low obstacles seen by the downward sensor, and camera sightings
        # placed on the map using the range sensor along the same bearing (vision + ranging).
        self.low_cells: dict[tuple[int, int], float] = {}
        self.sightings: list[tuple] = []  # camera rays: (label, drone xy, world bearing, confidence)
        self.world_angles = np.zeros(0)
        self.room_scan = None
        self.prior_map: str | None = None

    def _new_brain(self) -> FlightBrain:
        brain = FlightBrain(self.cfg, self.cfg.sensors.lidar_max_range)
        brain.interactive = True
        brain.waypoints = []
        if getattr(self, "prior_map", None):
            brain.load_prior_map(self.prior_map)
        return brain

    def use_prior_map(self, npz_path: str) -> bool:
        self.prior_map = str(npz_path)
        return self.submit(lambda: self.brain.load_prior_map(npz_path))

    # ------------------------------------------------------------ room scan
    def _note_sightings(self, dets: list[dict], position: np.ndarray) -> None:
        """Remember each camera detection as a viewing ray (where we were, which direction).
        Rays from different places are triangulated into map positions by the room scan."""
        for d in dets:
            self.sightings.append((d["label"], position[:2].copy(), float(d["bearing"]), float(d["confidence"])))
        if len(self.sightings) > 8000:
            self.sightings = self.sightings[-8000:]

    def _note_ground(self, position: np.ndarray, height_below: float) -> None:
        if height_below > 0.1 and self.brain.planner is not None:
            r, c = self.brain.planner.grid.to_cell(position[:2])
            key = (int(r[0]), int(c[0]))
            self.low_cells[key] = max(self.low_cells.get(key, 0.0), height_below)

    def _known_low_nearby(self, position: np.ndarray, radius: float = 0.45) -> float:
        """Highest low obstacle already mapped within ``radius``: climb *before* reaching the table."""
        if not self.low_cells or self.brain.planner is None:
            return 0.0
        g = self.brain.planner.grid
        r0, c0 = g.to_cell(position[:2])
        k = int(math.ceil(radius / g.res))
        best = 0.0
        for (r, c), h in self.low_cells.items():
            if abs(r - r0[0]) <= k and abs(c - c0[0]) <= k:
                best = max(best, h)
        return best

    def map_snapshot(self):
        import copy

        return self.submit(lambda: copy.deepcopy(self.brain.planner.grid) if self.brain.planner is not None else None)

    def build_scan(self, save_dir: str | Path | None = None):
        from skymind.survey import analyse, save, triangulate_landmarks

        grid = self.map_snapshot()
        if grid is None:
            return None
        cfg = self.cfg
        scan = analyse(grid, cfg.mission.geofence, 2.0 * cfg.avoidance.safety_radius, dict(self.low_cells),
                       triangulate_landmarks(list(self.sightings), cfg.mission.geofence))
        self.room_scan = scan
        # Doorways become places you can name: "go through doorway 1".
        for p in scan.passages:
            if p.kind in ("doorway", "exit") and p.passable:
                cfg.mission.places[f"{p.kind} {p.id}"] = list(p.center)
        if save_dir is not None:
            save(save_dir, grid, scan, cfg.mission.geofence)
        self._event("room scan: " + scan.summary())
        return scan

    def render_scan(self) -> bytes:
        from skymind.survey import render

        if self.room_scan is None:
            return b""
        grid = self.map_snapshot()
        ok, png = cv2.imencode(".png", render(grid, self.room_scan, self.cfg.mission.geofence, places=self.cfg.mission.places))
        return png.tobytes() if ok else b""

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="skymind-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.recorder.close()

    def _run(self) -> None:
        try:
            while self._running:
                self._drain_commands()
                self._tick()
        except Exception as exc:  # surface crashes to the app instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self._running = False  # wake every waiter: nothing is flying this drone any more
            with self._cond:
                self.state = {**self.state, "error": self.error}
                self._cond.notify_all()
            self._event(f"SESSION ERROR: {self.error}")
            raise

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, fut = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn())
            except Exception as exc:
                fut.set_exception(exc)

    def _tick(self) -> None:
        raise NotImplementedError

    def submit(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        """Run ``fn`` on the session thread between ticks and return its result."""
        if not self._running:
            return fn()
        fut: Future = Future()
        self._cmds.put((fn, fut))
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------ commands
    def takeoff(self, altitude: float | None = None) -> bool:
        alt = self.cfg.mission.cruise_altitude if altitude is None else altitude
        return self.submit(lambda: self._takeoff(alt))

    def _takeoff(self, altitude: float) -> bool:
        est = self._estimate()
        self.brain.yaw_sp = self.brain.yaw_target = est.euler[2]
        return self.brain.command_takeoff(est.position, altitude)

    def go_to(self, xy) -> bool:
        return self.submit(lambda: self.brain.command_goto(np.asarray(xy, dtype=float)))

    def hold(self) -> bool:
        return self.submit(lambda: self.brain.command_hold(self._estimate().position))

    def return_home(self) -> bool:
        return self.submit(self.brain.command_rtl)

    def land(self) -> bool:
        return self.submit(self.brain.command_land)

    def turn_to(self, yaw: float) -> bool:
        return self.submit(lambda: self.brain.command_yaw(yaw))

    def set_altitude(self, altitude: float) -> bool:
        return self.submit(lambda: self.brain.command_altitude(altitude))

    def snapshot_photo(self) -> str | None:
        def take():
            if self.latest_frame is None:
                return None
            labels = [d["label"] for d in self.latest_detections] or ["photo"]
            return self.recorder.snapshot(self.latest_frame, self.t, labels)

        return self.submit(take)

    # ------------------------------------------------------------ waiting
    def wait_for(self, predicate: Callable[[dict], bool], timeout: float, cancel: threading.Event | None = None) -> bool:
        """Wait (in *flight* seconds, so it also works for faster-than-real-time sims).

        Only states published after this call are considered, so a command issued just
        before (e.g. ``go_to``) is guaranteed to be reflected in what the predicate sees."""
        with self._cond:
            t0 = self.state.get("t", 0.0)
            tick0 = self.state.get("tick", -1)
            while True:
                if self.state.get("tick", -1) > tick0 and predicate(self.state):
                    return True
                if self.state.get("t", 0.0) - t0 > timeout or (cancel is not None and cancel.is_set()):
                    return False
                if not self._running:
                    return False
                self._cond.wait(0.05)

    def status(self) -> dict:
        with self._cond:
            return dict(self.state)

    # ------------------------------------------------------------ shared helpers
    def _estimate(self) -> StateEstimate:
        raise NotImplementedError

    def _event(self, text: str) -> None:
        self.events.append((round(self.t, 2), text))
        self.recorder.log_record({"type": "event", "t": self.t, "event": text})

    def bearing_of(self, bbox, yaw: float) -> float:
        """World-frame bearing (rad, CCW from east) of a detection's box centre."""
        cx = bbox[0] + bbox[2] / 2.0
        return float(wrap_angle(yaw + math.atan2(self.cfg.camera.width / 2.0 - cx, self.focal_px)))

    def _process_frame(self, frame: np.ndarray, yaw: float, hud_extra: str = "", ground_truth: list | None = None) -> None:
        cfg = self.cfg
        analyse = self._cam_frames % max(1, cfg.vision.detect_every) == 0
        self._cam_frames += 1
        if analyse:
            self.latest_analysis = self.analyzer.analyze(frame)
        analysis = self.latest_analysis
        dets = [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
                "area": d.bbox[2] * d.bbox[3],
                "bearing": self.bearing_of(d.bbox, yaw),
            }
            for d in analysis.detections
        ]
        st = self.state
        hud = [
            f"{cfg.assistant.name.upper()}  {st.get('mode', '').upper()}  t {self.t:5.1f}s",
            f"alt {st.get('position', [0, 0, 0])[2]:4.1f}m  bat {st.get('battery', 1.0) * 100:3.0f}%",
            f"obst {st.get('min_range', 0):3.1f}m {hud_extra}",
        ]
        if analyse and dets and self.brain.airborne:
            self._note_sightings(dets, np.array(st.get("position", [0, 0, 0])))
        annotated = draw_overlay(frame, analysis, hud)
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        labels = {d["label"] for d in dets}
        snap = None
        if analyse and cfg.vision.snapshot_on_detection and labels and (labels != self._last_labels or self.t - self._last_snap_t > 3.0):
            snap = self.recorder.snapshot(frame, self.t, list(labels))
            self._last_snap_t = self.t
        self._last_labels = labels
        self.recorder.write_frame(annotated)
        if analyse:
            rec = {"type": "vision", "t": self.t, **analysis.to_dict(), "snapshot": snap}
            if ground_truth is not None:
                rec["ground_truth"] = ground_truth
            self.recorder.log_record(rec)
        with self._cond:
            self.latest_frame = frame
            self.latest_jpeg = jpg.tobytes() if ok else self.latest_jpeg
            self.latest_detections = dets

    def _publish(self, est: StateEstimate, battery: float, power: float, extra: dict | None = None) -> None:
        b = self.brain
        path = []
        if b.planner is not None and b.planner.path and b.mode in (FlightMode.MISSION, FlightMode.RTL):
            path = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in b.planner.path[::3]]
        roll, pitch, yaw = est.euler
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        state = {
            "tick": self._tick_count,
            "t": round(self.t, 2),
            "mode": b.mode.value,
            "position": [round(float(v), 3) for v in est.position],
            "velocity": [round(float(v), 3) for v in est.velocity],
            "speed": round(float(np.linalg.norm(est.velocity[:2])), 2),
            "yaw_deg": round(math.degrees(yaw), 1),
            "roll_deg": round(math.degrees(roll), 1),
            "pitch_deg": round(math.degrees(pitch), 1),
            "battery": round(float(battery), 3),
            "power_w": round(float(power), 1),
            "min_range": round(float(self.ranges.min()), 2) if len(self.ranges) else None,
            "altitude_target": round(float(b.altitude), 2),
            "goal": [round(float(v), 2) for v in b.goal] if b.mode in (FlightMode.MISSION, FlightMode.RTL) else None,
            "home": [round(float(v), 2) for v in b.home],
            "failsafe": b.failsafe,
            "collisions": self.collisions,
            "path": path,
            "detections": [{k: d[k] for k in ("label", "confidence", "bearing")} for d in self.latest_detections],
            "error": self.error,
        }
        if extra:
            state.update(extra)
        with self._cond:
            self.state = state
            self._cond.notify_all()

    def _log_state(self, est: StateEstimate, truth_pos=None, truth_vel=None, truth_rpy=None, rate=None, battery=1.0, power=0.0, clearance=None) -> None:
        b = self.brain
        self.recorder.log_record(
            {
                "type": "state",
                "t": self.t,
                "mode": b.mode.value,
                "pos": est.position if truth_pos is None else truth_pos,
                "vel": est.velocity if truth_vel is None else truth_vel,
                "est_pos": est.position,
                "est_vel": est.velocity,
                "rpy": list(est.euler) if truth_rpy is None else list(truth_rpy),
                "est_rpy": list(est.euler),
                "rate": est.rate if rate is None else rate,
                "alt_sp": b.altitude,
                "vel_sp": [float(b.vel_cmd[0]), float(b.vel_cmd[1]), 0.0],
                "battery": battery,
                "power": power,
                "min_range": float(self.ranges.min()) if len(self.ranges) else None,
                "clearance": clearance,
                "wp": b.wp_index,
            }
        )

    # ------------------------------------------------------------ map
    def render_map(self, px_per_m: float = 40.0, ground_truth: bool = True) -> bytes:
        """Live top-down map as PNG: what the drone has mapped, plus places, path and pose."""
        cfg = self.cfg
        xmin, ymin, xmax, ymax = cfg.mission.geofence
        pad = 0.5
        scale = min(px_per_m, 900.0 / max(xmax - xmin + 2 * pad, 1e-6))
        w, h = int((xmax - xmin + 2 * pad) * scale), int((ymax - ymin + 2 * pad) * scale)
        img = np.full((h, w, 3), (24, 20, 16), np.uint8)

        def px(x, y):
            return int((x - xmin + pad) * scale), int((ymax + pad - y) * scale)

        b = self.brain
        if b.planner is not None:
            g = b.planner.grid
            occ = (g.log_odds > 0.6).astype(np.uint8) * 255
            free = (g.log_odds < -0.3).astype(np.uint8) * 60
            layer = np.maximum(occ, free)[::-1]  # rows: +y up
            x0, y0 = g.x0, g.y0
            x1, y1 = x0 + g.nx * g.res, y0 + g.ny * g.res
            (ax, ay), (bx, by) = px(x0, y1), px(x1, y0)
            if bx > ax and by > ay:
                big = cv2.resize(layer, (bx - ax, by - ay), interpolation=cv2.INTER_NEAREST)
                ox0, oy0 = max(ax, 0), max(ay, 0)
                ox1, oy1 = min(bx, w), min(by, h)
                crop = big[oy0 - ay : oy1 - ay, ox0 - ax : ox1 - ax]
                region = img[oy0:oy1, ox0:ox1]
                region[crop == 255] = (230, 220, 90)
                region[crop == 60] = (52, 44, 34)
        if ground_truth and isinstance(self, SimLiveSession):
            for x, y, r, _h in cfg.world.cylinders:
                cv2.circle(img, px(x, y), max(2, int(r * scale)), (90, 90, 90), 1, cv2.LINE_AA)
            for bx0, by0, bx1, by1, _h in cfg.world.boxes:
                cv2.rectangle(img, px(bx0, by1), px(bx1, by0), (90, 90, 90), 1)
        cv2.rectangle(img, px(xmin, ymax), px(xmax, ymin), (60, 60, 200), 1)
        for name, (x, y) in cfg.mission.places.items():
            cv2.circle(img, px(x, y), 4, (120, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(img, name, (px(x, y)[0] + 6, px(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1, cv2.LINE_AA)
        st = self.status()
        if st.get("path"):
            pts = np.array([px(x, y) for x, y in st["path"]], np.int32)
            cv2.polylines(img, [pts], False, (80, 255, 160), 1, cv2.LINE_AA)
        if st.get("goal"):
            cv2.drawMarker(img, px(*st["goal"]), (80, 255, 160), cv2.MARKER_CROSS, 14, 2)
        hx, hy = st.get("home", cfg.mission.home)
        cv2.drawMarker(img, px(hx, hy), (200, 200, 200), cv2.MARKER_SQUARE, 12, 1)
        if "position" in st:
            x, y = st["position"][:2]
            yaw = math.radians(st.get("yaw_deg", 0.0))
            tip = (x + 0.35 * math.cos(yaw), y + 0.35 * math.sin(yaw))
            left = (x + 0.2 * math.cos(yaw + 2.5), y + 0.2 * math.sin(yaw + 2.5))
            right = (x + 0.2 * math.cos(yaw - 2.5), y + 0.2 * math.sin(yaw - 2.5))
            tri = np.array([px(*tip), px(*left), px(*right)], np.int32)
            cv2.fillPoly(img, [tri], (255, 230, 0), cv2.LINE_AA)
            fov = math.radians(cfg.camera.fov_deg) / 2.0
            for a in (yaw - fov, yaw + fov):
                cv2.line(img, px(x, y), px(x + 1.2 * math.cos(a), y + 1.2 * math.sin(a)), (255, 230, 0), 1, cv2.LINE_AA)
        ok, png = cv2.imencode(".png", img)
        return png.tobytes() if ok else b""


class SimLiveSession(LiveSession):
    """Physics simulation running continuously, ``speed`` x real time (0 = as fast as possible)."""

    def __init__(self, cfg: SkyMindConfig, run_dir: str | Path, speed: float = 1.0):
        super().__init__(cfg, run_dir)
        self.speed = speed
        self._build_sim()

    def _build_sim(self) -> None:
        from skymind.sim.camera_sim import SimCamera
        from skymind.sim.lidar import SimLidar
        from skymind.sim.physics import QuadrotorSim
        from skymind.sim.world import World

        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.sim.seed)
        self.world = World(cfg.world)
        self.sim = QuadrotorSim(cfg.vehicle, cfg.sensors, self.rng)
        self.estimator = StateEstimator(cfg.estimator)
        self.controller = FlightController(cfg.vehicle, cfg.control)
        self.lidar = SimLidar(cfg.sensors, self.world, self.rng)
        self.camera = SimCamera(cfg.camera, self.world, self.rng)
        self.power = PowerModel(cfg.vehicle)
        self.battery = Battery(cfg.battery.capacity_wh, self.power)
        home = np.array([cfg.mission.home[0], cfg.mission.home[1], 0.0])
        self.sim.reset(position=home)
        self.estimator.reset(home, self.sim.state.quaternion)
        self.dt = 1.0 / cfg.sim.physics_hz
        hz = cfg.sim.physics_hz
        self.pos_every = max(1, round(hz / cfg.sensors.pos_rate_hz))
        self.brain_every = max(1, round(hz / cfg.sensors.lidar_rate_hz))
        self.cam_every = max(1, round(hz / cfg.camera.fps))
        self.log_every = max(1, round(hz / 25.0))
        self.k = 0
        self.setpoint: Setpoint | None = None
        self.in_collision = False
        self._wall0 = None
        from skymind.dodge import LiveDodge

        self.dodge = LiveDodge(cfg, self.rng)
        self._normal_tilt = self.controller.max_tilt

    def throw_ball(self, speed: float = 5.0) -> bool:
        """Simulation test: throw a paper ball at the drone from in front."""

        def do():
            if not self.brain.airborne:
                return False
            st = self.sim.state
            self.dodge.throw(st.position.copy(), st.euler[2], speed)
            self._event(f"ball thrown at {speed:.0f} m/s")
            return True

        return self.submit(do)

    def reset(self) -> bool:
        """Put the drone back on the ground at home (simulation only)."""

        def do():
            self._build_sim()
            self.brain = self._new_brain()
            self.collisions = 0
            self._event("simulation reset")
            return True

        return self.submit(do)

    def _estimate(self) -> StateEstimate:
        return self.estimator.estimate()

    def _tick(self) -> None:
        cfg, sim, dt = self.cfg, self.sim, self.dt
        if self._wall0 is None:
            self._wall0 = time.time() - self.t / self.speed if self.speed > 0 else time.time()
        est = self.estimator.estimate()
        st = sim.state
        ranges, angles = self.lidar.scan(st.position, st.euler[2])
        self.ranges, self.world_angles = ranges, angles
        if self.brain.airborne and st.position[2] > 0.3:
            # Downward ToF cone (~27 deg) + prop radius: sees a table edge before the props reach it.
            fp = cfg.vehicle.body_radius + st.position[2] * math.tan(math.radians(13.5))
            seen, at = self.world.height_below_footprint(st.position, fp, where=True)
            # Record it where the footprint saw it (estimate frame), for the room scan.
            self._note_ground(self._estimate().position + (at - st.position), seen)
            self.brain.ground_below = max(seen, self._known_low_nearby(self._estimate().position))
        landed = sim.on_ground and self.brain.mode is FlightMode.LAND
        out = self.brain.update(est, ranges, angles, self.battery.fraction, self.t, self.brain_every * dt, landed)
        for e in out.events:
            self._event(e)
        self.setpoint = out.setpoint

        for _ in range(self.brain_every):
            imu = sim.imu()
            self.estimator.update_imu(imu.gyro, imu.accel, dt, sim.heading_measurement())
            if self.k % self.pos_every == 0:
                self.estimator.update_position(sim.position_measurement(), self.pos_every * dt)
            est = self.estimator.estimate()
            was_escaping = self.dodge.escaping(self.t)
            if self.dodge.balls:
                for e in self.dodge.step(self.t, dt, sim, cfg.vehicle.body_radius):
                    self._event(e)
            escaping = self.dodge.escaping(self.t) and self.brain.airborne
            self.controller.max_tilt = math.radians(self.dodge.rcfg.dodge_tilt_deg) if escaping else self._normal_tilt
            self.controller.min_thrust_frac = 0.05 if escaping else 0.3
            if was_escaping and not escaping and self.brain.mode is FlightMode.HOLD:
                self.brain.command_hold(est.position)  # stay where the dodge took us
            if self.brain.airborne and self.brain.mode is not FlightMode.CRASHED:
                motors = self.controller.update(est, self.dodge.setpoint() if escaping else self.setpoint, dt).motors
            else:
                motors = self.controller.idle().motors
                self.controller.reset()
            sim.step(motors, dt, self.world.wind_at(self.rng))
            self.battery.drain(sim.state.motor_thrust, dt)
            self.t += dt
            self.k += 1
            pos = sim.state.position
            clear = self.world.clearance(pos)
            if clear < cfg.vehicle.body_radius and pos[2] > 0.05:
                if not self.in_collision:
                    self.collisions += 1
                    self._event("COLLISION")
                self.in_collision = True
                self.brain._enter(FlightMode.CRASHED, self.brain.pending_events, "collision")
            else:
                self.in_collision = False
            if self.k % self.log_every == 0:
                roll, pitch, yaw = sim.state.euler
                self._log_state(est, pos.copy(), sim.state.velocity.copy(), (roll, pitch, yaw), sim.state.omega.copy(),
                                self.battery.fraction, self.battery.last_power, clear if clear < 1e8 else None)
            if self.k % self.cam_every == 0:
                roll, pitch, yaw = sim.state.euler
                self.camera.extra_spheres = [(b.pos.copy(), b.radius, (235, 235, 235)) for b in self.dodge.balls if b.active]
                frame, gt = self.camera.render(pos, yaw, roll, pitch, sim.state.omega)
                gtl = [{"label": g.label, "bbox": g.bbox, "distance": g.distance} for g in gt]
                self._process_frame(frame, yaw, ground_truth=gtl)
        # The flight brain and the UI see the onboard estimate; the sim keeps the truth for scoring.
        self._publish(self.estimator.estimate(), self.battery.fraction, self.battery.last_power,
                      {"simulated": True, "dodge": dict(self.dodge.stats), "scan_ready": self.room_scan is not None})
        if self.speed > 0:
            ahead = self._wall0 + self.t / self.speed - time.time()
            if ahead > 0:
                time.sleep(ahead)


class RealLiveSession(LiveSession):
    """A real drone: ESP-Drone (``EspDroneLink``) or an ArduPilot/PX4 autopilot (``MavlinkVehicle``).

    Without ``arm=True`` it is a bench session: telemetry, sensors, camera and the assistant all
    work, but takeoff is refused, so the motors never spin.
    """

    LOOP_HZ = 20.0
    TELEMETRY_TIMEOUT = 1.0

    def __init__(self, cfg: SkyMindConfig, run_dir: str | Path, vehicle, sensors, camera=None, arm: bool = False):
        super().__init__(cfg, run_dir)
        self.vehicle, self.sensors, self.camera, self.arm = vehicle, sensors, camera, arm
        self._prev_mode = self.brain.mode
        self._t0: float | None = None
        self._k = 0
        self._est = StateEstimate(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))

    def start(self) -> None:
        self.vehicle.wait_ready()
        super().start()

    def stop(self) -> None:
        if self.brain.airborne:
            self.land()
            self.wait_for(lambda s: s["mode"] == "landed", timeout=20.0)
        super().stop()
        if hasattr(self.vehicle, "close"):
            self.vehicle.close()

    def _takeoff(self, altitude: float) -> bool:
        if not self.arm:
            self._event("takeoff refused: bench mode (start with --arm to fly)")
            return False
        return super()._takeoff(altitude)

    def _estimate(self) -> StateEstimate:
        return self._est

    def _tick(self) -> None:
        v, cfg = self.vehicle, self.cfg
        start = time.time()
        if self._t0 is None:
            self._t0 = start
        self.t = start - self._t0
        tm = v.poll()
        self._est = StateEstimate(tm.position, tm.velocity, quat_from_euler(*tm.rpy), tm.rates)
        ranges, body_angles = self.sensors.scan()
        self.ranges, self.world_angles = ranges, body_angles + tm.rpy[2]
        out = self.brain.update(self._est, ranges, body_angles + tm.rpy[2], tm.battery_fraction, self.t, 1.0 / self.LOOP_HZ, tm.landed)
        for e in out.events:
            self._event(e)
        mode = self.brain.mode
        if self.arm:
            if mode is FlightMode.TAKEOFF and self._prev_mode is not FlightMode.TAKEOFF:
                v.set_mode("GUIDED")
                v.arm()
                v.takeoff(self.brain.altitude)
            if mode is FlightMode.LAND and self._prev_mode is not FlightMode.LAND:
                v.land()
            if self.brain.airborne:
                if time.time() - tm.last_update > self.TELEMETRY_TIMEOUT:
                    self._event("telemetry lost -> LAND")
                    self.brain._enter(FlightMode.LAND, [], "telemetry lost")
                    v.land()
                v.keepalive()
                v.send_obstacle_distance(ranges, body_angles, cfg.sensors.lidar_max_range)
                if mode in (FlightMode.MISSION, FlightMode.RTL, FlightMode.HOLD):
                    if mode is FlightMode.HOLD:
                        err = (self.brain.hold_xy - tm.position[:2]) * 0.8
                        vel_xy = np.clip(err, -0.5, 0.5)
                    else:
                        vel_xy = out.setpoint.velocity[:2]
                    vz = float(np.clip(1.0 * (self.brain.altitude - tm.position[2]), -0.8, 0.8))
                    v.send_velocity_enu(np.array([vel_xy[0], vel_xy[1], vz]), out.setpoint.yaw, altitude=self.brain.altitude)
        self._prev_mode = mode
        self._log_state(self._est, battery=tm.battery_fraction)
        if self.camera is not None and self._k % max(1, round(self.LOOP_HZ / cfg.camera.fps)) == 0:
            frame = self.camera.read()
            if frame is not None:
                self._process_frame(frame, tm.rpy[2])
        self._k += 1
        self._publish(self._est, tm.battery_fraction, 0.0, {"simulated": False, "armed_allowed": self.arm})
        time.sleep(max(0.0, 1.0 / self.LOOP_HZ - (time.time() - start)))
