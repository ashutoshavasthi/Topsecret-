# SkyMind: AI-powered self-balancing, obstacle-avoiding camera drone

SkyMind is the complete software stack for an autonomous quadcopter that:

* **balances itself**: cascaded attitude/rate control with geometric (SO(3)) attitude error, a Mahony
  AHRS that learns gyro bias online, and a mixer that keeps balance authority when the motors saturate;
* **avoids obstacles**: a 360° lidar feeds a reactive swept-corridor avoider, plus an occupancy map with
  an A* planner so it can't get trapped. It treats the geofence as a wall and climbs over obstacles it cannot route around;
* **records and analyses images**: onboard camera video with a HUD, AI object detection (colour targets out of the
  box, YOLOv8 ONNX optionally) and detection snapshots;
* **gets judged**: every flight produces an automatic, reproducible scorecard covering stability, navigation, safety and
  vision, with a flight map and timeline.

It runs entirely in a built-in physics simulator (no hardware needed). The same autonomy code drives a real
drone through a Pixhawk (ArduPilot/PX4) over MAVLink.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                          # safety-critical behaviour tests
python -m skymind sim                                        # fly, record, judge the "Forest Run" mission
python -m skymind sim --config config/windy_city.yaml        # gusty wind, narrow alley, wall to climb
```

A run writes `runs/sim-<timestamp>/`:

```
flight.mp4        onboard camera, HUD + AI boxes
snapshots/        frames where the AI detected something
flight_map.png    top-down path coloured by flight mode
timeline.png      altitude / speed / tilt / nearest obstacle
telemetry.jsonl   full log (25 Hz state + every camera frame)
report.md/.json   the judge's scorecard
```

Example console output (default scenario, seed 7):

```
status=landed waypoints=2/2 collisions=0 min_clearance=0.987 m duration=35.96 s
recorded 349 frames, 18 detection snapshots

=== SCORE 91.2/100  grade A  (PASS) ===
  stability     71.9   (weight 0.30)
  navigation   100.0   (weight 0.25)
  safety       100.0   (weight 0.30)
  vision        97.8   (weight 0.15)
```

Across 12 random seeds of the forest course: 12/12 landed, 0 collisions, mean score 91.4.

## Other commands

```bash
python -m skymind judge runs/sim-...                    # re-score a run (e.g. after changing judge weights)
python -m skymind analyze video.mp4 --model yolov8n.onnx --labels coco.txt   # AI + quality scoring of any footage
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0            # real drone, BENCH MODE (never arms)
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0 --arm      # real autonomous flight
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the pieces fit, with the main design decisions
* [docs/JUDGING.md](docs/JUDGING.md): how flights are scored, and a 5-minute demo script for judges
* [docs/HARDWARE.md](docs/HARDWARE.md): parts list, wiring, autopilot parameters and a safe bring-up sequence

## Project layout

```
skymind/
  control/      PID, mixer, state estimator, self-balancing flight controller
  perception/   reactive obstacle avoidance, occupancy mapping + A*
  vision/       detectors, image quality, recorder, camera input, offline analysis
  sim/          quadrotor physics, world, lidar, synthetic camera
  evaluation/   flight judge, map/timeline plots
  hal/          MAVLink bridge, RPLidar driver, real-flight runner
  mission.py    FlightBrain (autonomy) + SimulationRunner
  cli.py        `python -m skymind ...`
config/         mission scenarios (YAML)
tests/          unit + end-to-end mission tests
```

## Scenarios

Scenarios are plain YAML (`config/*.yaml`): waypoints, geofence, obstacles (cylinders and boxes), targets for
the camera to find, and wind. Any default in `skymind/config.py` can be overridden there, including controller
gains, sensor noise, avoidance parameters, battery size and the judge's thresholds and weights.

## Safety

Real flights default to bench mode. `--arm` asks for confirmation, failsafes (low battery → RTL, geofence,
time limit, telemetry loss → LAND) are always on, and the lidar is forwarded to the autopilot so its own proximity
avoidance acts as a second layer. Always keep an RC pilot ready to take over, and follow local drone regulations.

License: MIT
