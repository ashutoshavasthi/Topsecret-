# Architecture

```
                      ┌──────────────────────────── FlightBrain (20 Hz) ────────────────────────────┐
 lidar 360° ─────────►│ occupancy grid ─► A* global path ─► local goal                              │
 (ranges)             │                                        │                                     │
                      │ geofence → virtual walls ──► reactive avoider (corridor clearance, VFH-style)│
                      │                                        │                                     │
                      │   jerk-limited reference model ─► safety governor ─► velocity + accel FF     │
                      │   mode logic: TAKEOFF → MISSION → RTL → LAND, failsafes, climb-when-stuck    │
                      └────────────────────────────────────────┬─────────────────────────────────────┘
                                                               │ Setpoint(pos/vel/acc, yaw)
 IMU (gyro+accel) ─► Mahony AHRS ─┐                            ▼
 GNSS/flow (pos) ──► Kalman filter ┴─► StateEstimate ─► FlightController (250 Hz, cascaded)
                                                       pos P → vel PID → thrust vector → SO(3) attitude P
                                                       → rate PID → torque → desaturating Quad-X mixer
                                                               │
                                                               ▼ motor thrusts
                                                    Quadrotor physics (sim) / ESCs (real)

 camera ─► FrameAnalyzer (HSV markers + optional YOLOv8 ONNX, quality metrics) ─► FlightRecorder
                                                                                  (mp4 + HUD, snapshots, JSONL)
                                                                                         │
                                                                                         ▼
                                                                        FlightJudge → report.md / .json
```

## Modules

| Path | Responsibility |
|---|---|
| `skymind/math3d.py` | Quaternions, rotations, ENU/FLU conventions |
| `skymind/config.py` | Typed configuration with defaults, YAML overrides |
| `skymind/control/pid.py` | Vector PID: derivative on measurement, filtered D, conditional-integration anti-windup |
| `skymind/control/mixer.py` | Quad-X allocation. Mixer that shifts collective and then drops yaw, so roll/pitch authority (balance) is preserved at saturation |
| `skymind/control/estimator.py` | Mahony complementary attitude filter with online gyro-bias learning and linear-acceleration compensation, plus a constant-velocity Kalman filter for position/velocity |
| `skymind/control/flight_controller.py` | Cascaded controller with tilt limiting, geometric attitude error (stable at large angles) and acceleration feed-forward |
| `skymind/perception/avoidance.py` | Reactive avoidance: swept-corridor clearance per heading, goal/smoothness cost, braking-distance speed limit, repulsion |
| `skymind/perception/mapping.py` | Log-odds occupancy grid from lidar, inflated A* planner, line-of-sight local goal |
| `skymind/mission.py` | `FlightBrain` (autonomy, shared by sim and real flight) and `SimulationRunner` |
| `skymind/sim/` | Rigid-body quadrotor with motor lag, drag and wind. Ray-cast world, 2D lidar, synthetic camera with ground-truth boxes |
| `skymind/vision/` | Detection back-ends, image-quality metrics, recorder, real camera input, offline video analysis |
| `skymind/evaluation/` | Flight judge and report plots |
| `skymind/hal/` | MAVLink bridge (ENU↔NED, velocity setpoints, OBSTACLE_DISTANCE forwarding), RPLidar driver, real-flight runner |

## Key design decisions

**Self-balancing.** The inner loop drives the geometric attitude error
`e_R = ½ vee(R_dᵀR − RᵀR_d)` to zero through a rate loop. This error is
well-defined even for very large tilts, so the drone recovers from 35°+
upsets (see `tests/test_control.py`). The mixer gives balance priority over
thrust when the motors saturate.

**Estimation on a multirotor.** A multirotor's accelerometer mostly measures
thrust along body z, so a naïve complementary filter believes the drone is
level while it is accelerating. SkyMind subtracts the linear acceleration
estimated by the velocity filter from the gravity reference.

**Two-layer avoidance.** The reactive layer reacts within 50 ms to anything
the lidar sees and checks the drone's real swept width, so it never squeezes
through gaps that are too narrow. The global layer (map + A*) gets the drone
out of local minima such as a goal straight behind a building. If both fail
(e.g. a wall across the entire site) the brain climbs over it.

**Safety governor.** The avoider limits speed along the heading it chooses,
but momentum may point elsewhere. After smoothing, the governor caps speed
along both the command and the actual velocity to what can be shed before
`stop_distance`. The cap is never smoothed away.

**Smooth references.** Raw avoidance commands are passed through a
jerk-limited second-order reference model. Its acceleration feeds the
controller as feed-forward, so the drone tracks closely without lag (which
matters for safety) and the footage stays steady (which matters for the
vision score).

**Same brain in sim and reality.** `FlightBrain` only consumes a state
estimate, lidar ranges and battery level, and only emits setpoints. The sim
closes the loop with physics. The MAVLink bridge closes it with a real autopilot.
