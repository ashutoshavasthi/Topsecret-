"""Run the SkyMind autonomy stack on a real drone.

Safety defaults: without ``arm=True`` this is a *bench* run — it connects,
reads telemetry, lidar and camera, runs avoidance and the AI pipeline and
records everything, but never arms the motors. Always test with propellers
removed first, then in an open field with a pilot holding an RC override.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from skymind.config import SkyMindConfig
from skymind.control.estimator import StateEstimate
from skymind.hal.mavlink_bridge import MavlinkVehicle
from skymind.math3d import quat_from_euler
from skymind.mission import FlightBrain, FlightMode
from skymind.vision.analyzer import FrameAnalysis, FrameAnalyzer, draw_overlay
from skymind.vision.recorder import FlightRecorder


class RangeSource(Protocol):
    def scan(self) -> tuple[np.ndarray, np.ndarray]: ...


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...


class RealFlightRunner:
    LOOP_HZ = 20.0
    TELEMETRY_TIMEOUT = 1.0  # s without MAVLink data -> land

    def __init__(
        self,
        cfg: SkyMindConfig,
        vehicle: MavlinkVehicle,
        lidar: RangeSource,
        camera: FrameSource | None,
        run_dir: str | Path,
        arm: bool = False,
    ):
        self.cfg, self.vehicle, self.lidar, self.camera, self.arm = cfg, vehicle, lidar, camera, arm
        self.brain = FlightBrain(cfg, cfg.sensors.lidar_max_range)
        self.analyzer = FrameAnalyzer(cfg.vision)
        self.rec = FlightRecorder(run_dir, cfg.camera.fps, (cfg.camera.width, cfg.camera.height))

    def run(self) -> None:
        cfg, v = self.cfg, self.vehicle
        v.wait_ready()
        if self.arm:
            v.set_mode("GUIDED")
            v.arm()
            v.takeoff(cfg.mission.cruise_altitude)
        self.brain.start()
        dt = 1.0 / self.LOOP_HZ
        cam_every = max(1, round(self.LOOP_HZ / cfg.camera.fps))
        t0 = time.time()
        k = 0
        land_sent = False
        last_labels: set[str] = set()
        last_snap_t = -1e9
        cam_frames = 0
        last_analysis = FrameAnalysis()
        try:
            while True:
                loop_start = time.time()
                t = loop_start - t0
                tm = v.poll()
                if self.arm and time.time() - tm.last_update > self.TELEMETRY_TIMEOUT:
                    v.land()
                    self.rec.log_record({"type": "event", "t": t, "event": "telemetry lost -> LAND"})
                    break

                est = StateEstimate(tm.position, tm.velocity, quat_from_euler(*tm.rpy), tm.rates)
                body_ranges, body_angles = self.lidar.scan()
                world_angles = body_angles + tm.rpy[2]
                out = self.brain.update(est, body_ranges, world_angles, tm.battery_fraction, t, dt, tm.landed)
                for e in out.events:
                    self.rec.log_record({"type": "event", "t": t, "event": e})
                    print(f"[{t:6.1f}s] {e}", flush=True)

                mode = self.brain.mode
                if self.arm:
                    v.keepalive()
                    v.send_obstacle_distance(body_ranges, body_angles, cfg.sensors.lidar_max_range)
                    if mode in (FlightMode.MISSION, FlightMode.RTL):
                        vz = float(np.clip(1.0 * (self.brain.altitude - tm.position[2]), -1.0, 1.0))
                        vel = np.array([out.setpoint.velocity[0], out.setpoint.velocity[1], vz])
                        v.send_velocity_enu(vel, out.setpoint.yaw, altitude=self.brain.altitude)
                    elif mode is FlightMode.LAND and not land_sent:
                        v.land()
                        land_sent = True

                self.rec.log_record(
                    {
                        "type": "state",
                        "t": t,
                        "mode": mode.value,
                        "pos": tm.position,
                        "vel": tm.velocity,
                        "rpy": list(tm.rpy),
                        "rate": tm.rates,
                        "alt_sp": self.brain.altitude,
                        "vel_sp": out.setpoint.velocity,
                        "battery": tm.battery_fraction,
                        "min_range": float(body_ranges.min()),
                        "avoid": out.avoidance.mode.value if out.avoidance else None,
                        "wp": self.brain.wp_index,
                    }
                )

                if self.camera is not None and k % cam_every == 0:
                    frame = self.camera.read()
                    if frame is not None:
                        analyse_now = cam_frames % max(1, cfg.vision.detect_every) == 0
                        cam_frames += 1
                        if analyse_now:
                            last_analysis = self.analyzer.analyze(frame)
                        analysis = last_analysis
                        hud = [f"t {t:5.1f}s {mode.value.upper()}", f"alt {tm.position[2]:4.1f}m bat {tm.battery_fraction * 100:3.0f}%"]
                        self.rec.write_frame(draw_overlay(frame, analysis, hud))
                        labels = {d.label for d in analysis.detections}
                        snap = None
                        if analyse_now and labels and self.cfg.vision.snapshot_on_detection and (labels != last_labels or t - last_snap_t > 3.0):
                            snap = self.rec.snapshot(frame, t, list(labels))
                            last_snap_t = t
                        last_labels = labels
                        if analyse_now:
                            self.rec.log_record({"type": "vision", "t": t, **analysis.to_dict(), "snapshot": snap})

                if mode in (FlightMode.LANDED, FlightMode.CRASHED) or (not self.arm and t > cfg.mission.max_duration):
                    break
                k += 1
                time.sleep(max(0.0, dt - (time.time() - loop_start)))
        except KeyboardInterrupt:
            if self.arm:
                v.land()
                # Keep feeding the landing ramp (ESP-Drone) until it is down.
                end = time.time() + 10.0
                while time.time() < end and not v.poll().landed:
                    v.keepalive()
                    time.sleep(0.05)
        finally:
            self.rec.close()
            if hasattr(v, "close"):
                v.close()
