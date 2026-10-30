# SkyMind: AI-powered self-balancing, obstacle-avoiding camera drone

SkyMind is the complete software stack for an autonomous quadcopter that:

* **balances itself**: cascaded attitude/rate control with geometric (SO(3)) attitude error, a Mahony
  AHRS that learns gyro bias online, and a mixer that keeps balance authority when the motors saturate;
* **avoids obstacles**: a 360° lidar feeds a reactive swept-corridor avoider, plus an occupancy map with
  an A* planner so it can't get trapped. It treats the geofence as a wall and climbs over obstacles it cannot route around;
* **records and analyses images**: onboard camera video with a HUD, AI object detection (colour targets out of the
  box, YOLOv8 ONNX optionally) and detection snapshots;
* **gets judged**: every flight produces an automatic, reproducible scorecard covering stability, navigation, safety,
  vision and efficiency/noise, with a flight map and timeline;
* **is efficient**: a physics power and noise model, a design calculator to choose props and batteries, and a CPU
  benchmark so the onboard software fits a $15 Raspberry Pi Zero 2 W.

It runs entirely in a built-in physics simulator (no hardware needed). The same autonomy code drives a real
drone through a Pixhawk (ArduPilot/PX4) over MAVLink.

## Two builds

| | **SkyMind Micro Quiet** (college build) | SkyMind (full size) |
|---|---|---|
| Size / weight | 175 mm, 4" props, **240 g** (sub-250 g class) | 450 mm, 10" props, 1.2 kg |
| Parts cost | **≈ $230** | ≈ $700 |
| Brains | Raspberry Pi Zero 2 W + F405 flight controller | Raspberry Pi 5 + Pixhawk |
| Obstacle sensing | 6 × VL53L1X time-of-flight (9 g) with turn-then-go | 360° RPLidar |
| Hover power / endurance | 42 W / 8.7 min | 134 W / 14.3 min |
| Noise (tip speed) | 77 m/s, 2.7 dB quieter than a 3.5" micro | 69 m/s |
| Config | `config/micro_quiet.yaml`, scenario `config/micro_indoor.yaml` | `config/default.yaml` |

Parts list, weight budget, wiring and quiet-flying tips: **[docs/MICRO_BUILD.md](docs/MICRO_BUILD.md)**.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                          # safety-critical behaviour tests
python -m skymind sim                                        # fly, record, judge the "Forest Run" mission
python -m skymind sim --config config/windy_city.yaml        # gusty wind, narrow alley, wall to climb
python -m skymind sim --config config/micro_indoor.yaml      # the $230 micro drone in a college lab
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

Example console output (micro drone, indoor lab):

```
status=landed waypoints=3/3 collisions=0 min_clearance=0.645 m duration=56.64 s
recorded 557 frames, 9 detection snapshots

=== SCORE 90.6/100  grade A  (PASS) ===
  stability     86.1   (weight 0.25)
  navigation    85.2   (weight 0.20)
  safety       100.0   (weight 0.30)
  vision        85.5   (weight 0.10)
  efficiency    89.7   (weight 0.15)
```

Robustness across random seeds (sensor noise, gusts): forest course 8/8 landed, 0 collisions,
mean 90.3; micro indoor course 8/8 (3.5") and 6/6 (Quiet 4") landed, 0 collisions, mean 87.4 / 89.7.

## Other commands

```bash
python -m skymind judge runs/sim-...                    # re-score a run (e.g. after changing judge weights)
python -m skymind analyze video.mp4 --model yolov8n.onnx --labels coco.txt   # AI + quality scoring of any footage
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0            # real drone, BENCH MODE (never arms)
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0 --arm      # real autonomous flight
python -m skymind design config/micro.yaml config/micro_quiet.yaml           # compare builds: power, endurance, noise
python -m skymind design config/micro_quiet.yaml --sweep-battery 4.4,6.7,8.1,9.6   # pick a battery
python -m skymind bench --config config/micro_indoor.yaml                    # onboard CPU budget (run it on the Pi)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the pieces fit, with the main design decisions
* [docs/JUDGING.md](docs/JUDGING.md): how flights are scored, and a 5-minute demo script for judges
* [docs/MICRO_BUILD.md](docs/MICRO_BUILD.md): the small, quiet, ≈ $230 college build
* [docs/HARDWARE.md](docs/HARDWARE.md): the full-size build: parts, wiring, autopilot parameters and bring-up

## Project layout

```
skymind/
  control/      PID, mixer, state estimator, self-balancing flight controller
  perception/   reactive obstacle avoidance, occupancy mapping + A*
  vision/       detectors, image quality, recorder, camera input, offline analysis
  sim/          quadrotor physics, world, lidar, synthetic camera
  evaluation/   flight judge, map/timeline plots
  hal/          MAVLink bridge, RPLidar + VL53L1X ToF-ring drivers, real-flight runner
  power.py      power / endurance / rotor-noise model
  design.py     build comparison calculator
  bench.py      companion-computer CPU benchmark
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
