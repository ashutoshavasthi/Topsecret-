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
* **dodges thrown objects**: infrared ToF + camera "looming" (both see paper and plastic), an onboard
  reflex that predicts the ball's arc and picks the escape with the biggest miss. In simulation: 90–100 %
  dodged at 3–5 m/s, 80 % at 7 m/s from 4 m. See [docs/DODGE.md](docs/DODGE.md);
* **scans a room**: explores by itself and maps obstacles, doorways, exits, tight gaps, tables to fly over,
  and where it saw things (vision triangulation + ranging), then uses the map on later flights.
  See [docs/ROOM_SCAN.md](docs/ROOM_SCAN.md) (example: [docs/images/room_scan.png](docs/images/room_scan.png));
* **talks to you**: a voice and chat co-pilot app. Say "go to the door and tell me what you see";
  a free, local AI (Ollama) plans the flight, looks through the camera and answers, while the safety
  layer stays in charge. See [docs/ASSISTANT.md](docs/ASSISTANT.md) (screenshot: [docs/images/app.jpg](docs/images/app.jpg));
* **is efficient**: a physics power and noise model, a design calculator to choose props and batteries, and a CPU
  benchmark so the onboard software fits a $15 Raspberry Pi Zero 2 W.

It runs entirely in a built-in physics simulator (no hardware needed). The same autonomy code drives a real
drone through a Pixhawk (ArduPilot/PX4) over MAVLink.

## Three builds

| | **SkyMind Nano** (₹5,000 build) | **Micro Quiet** (college build) | SkyMind (full size) |
|---|---|---|---|
| Parts cost | **≈ ₹4,000–5,000** | ≈ $230 (₹19k) | ≈ $700 |
| Size / weight | ~120 mm, **80 g** | 175 mm, 240 g | 450 mm, 1.2 kg |
| Flight control | ESP32-S3 + ESP-Drone firmware | F405 + ArduPilot | Pixhawk + ArduPilot/PX4 |
| Where the AI runs | **your laptop**, over WiFi | Pi Zero 2 W onboard | Pi 5 onboard |
| Obstacle sensing | 3 × VL53L0X, forward | 6 × VL53L1X ring | 360° RPLidar |
| Flight time (est.) | ~4.7 min | ~8.7 min | ~14 min |
| Config | `nano.yaml` / `nano_room.yaml` | `micro_quiet.yaml` / `micro_indoor.yaml` | `default.yaml` |
| Guide | [docs/BUDGET_BUILD.md](docs/BUDGET_BUILD.md) | [docs/MICRO_BUILD.md](docs/MICRO_BUILD.md) | [docs/HARDWARE.md](docs/HARDWARE.md) |


## Quick start

```bash
pip install -r requirements.txt
python -m skymind app                                        # voice/chat control app -> http://localhost:8080
python -m skymind survey                                     # explore + scan a room (obstacles, doorways, exits)
python -m skymind dodge                                      # throw simulated balls at it, measure dodges
python -m pytest -q                                          # safety-critical behaviour tests
python -m skymind sim                                        # fly, record, judge the "Forest Run" mission
python -m skymind sim --config config/windy_city.yaml        # gusty wind, narrow alley, wall to climb
python -m skymind sim --config config/micro_indoor.yaml      # the $230 micro drone in a college lab
python -m skymind sim --config config/nano_room.yaml         # the ₹5,000 nano drone, brain on the laptop over WiFi
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

Robustness across random seeds (sensor noise, gusts, WiFi lag and packet loss), all with 0 collisions:

| Scenario | Runs landed | Mean score |
|---|---:|---:|
| Forest (full size) | 6/6 | 91.0 |
| Micro Quiet, indoor lab | 6/6 | 90.7 |
| Nano, room + doorway, 60 ms lag, 5 % loss | 8/8 | 94.6 |
| Nano, bad WiFi: 150 ms lag, 20 % loss | 6/6 | 93.1 |

## Other commands

```bash
python -m skymind judge runs/sim-...                    # re-score a run (e.g. after changing judge weights)
python -m skymind analyze video.mp4 --model yolov8n.onnx --labels coco.txt   # AI + quality scoring of any footage
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0            # real drone, BENCH MODE (never arms)
python -m skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0 --arm      # real autonomous flight
python -m skymind fly --config config/nano_room.yaml --esp-drone udp://192.168.43.42   # Nano (bench mode)
python -m skymind design config/micro.yaml config/micro_quiet.yaml           # compare builds: power, endurance, noise
python -m skymind design config/micro_quiet.yaml --sweep-battery 4.4,6.7,8.1,9.6   # pick a battery
python -m skymind bench --config config/micro_indoor.yaml                    # onboard CPU budget (run it on the Pi)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the pieces fit, with the main design decisions
* [docs/JUDGING.md](docs/JUDGING.md): how flights are scored, and a 5-minute demo script for judges
* [docs/ASSISTANT.md](docs/ASSISTANT.md): the voice/chat assistant app and its local AI
* [docs/DODGE.md](docs/DODGE.md): dodging thrown objects: sensors, reflex, results, add-on parts in ₹
* [docs/ROOM_SCAN.md](docs/ROOM_SCAN.md): room scanning: exploration, doorways/exits, fusion, using the map
* [docs/BUDGET_BUILD.md](docs/BUDGET_BUILD.md): the ₹5,000 Nano build (ESP32 + laptop brain)
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
  hal/          MAVLink + ESP-Drone (cflib) links, RPLidar / VL53L1X / ESP32-CAM sensor drivers, real-flight runner
  power.py      power / endurance / rotor-noise model
  design.py     build comparison calculator
  bench.py      companion-computer CPU benchmark
  mission.py    FlightBrain (autonomy) + SimulationRunner
  dodge.py      thrown-object sensors, ballistic tracker, model-predictive escape, throw experiments
  survey.py     frontier exploration, room-scan analysis, triangulation, saved maps
  live.py       live command-driven sessions (simulated or real drone)
  assistant/    skills, built-in command parser, Ollama client, conversational agent
  app/          web control app (video, map, chat, voice, STOP)
  cli.py        `python -m skymind ...`
config/         hardware profiles and mission scenarios (YAML)
firmware/       ESP32-CAM sensor-head sketch (Nano build), dodge-reflex C core (compiled + tested)
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
