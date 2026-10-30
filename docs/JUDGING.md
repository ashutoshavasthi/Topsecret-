# How flights are judged

Every flight, simulated or real, ends with an automatic, reproducible
scorecard (`report.md` / `report.json`). Judges can re-run it themselves:

```bash
python -m skymind sim --out runs/judge-demo      # fly + record + score
python -m skymind judge runs/judge-demo          # re-score any run
python -m skymind analyze some_video.mp4         # score footage from any camera
```

## Scoring method

Every metric has a **good** threshold (100 points) and a **bad** threshold (0
points), and scores linearly in between. A category's score is the mean of its
metrics. The overall score is the weighted mean of the categories. All
thresholds and weights are in `judge:` in the config, so they are transparent
and can be tuned to the competition's rules.

| Category | Weight | Metrics (good → bad) |
|---|---:|---|
| **Stability** (self-balancing) | 0.30 | RMS roll+pitch rate (0.2 → 1.5 rad/s), max tilt (20° → 45°), attitude-estimate error vs truth (1° → 6°, sim only), RMS altitude error (0.1 → 0.8 m), velocity tracking error (0.2 → 1.5 m/s) |
| **Navigation** | 0.25 | Waypoints reached (100 % → 0 %), path efficiency = straight-line / flown (0.85 → 0.4) |
| **Safety** (obstacle avoidance) | 0.30 | Collisions (any = 0 points), closest approach to an obstacle (0.8 → 0.3 m from the vehicle centre), failsafes triggered, safe landing |
| **Vision** (camera + AI) | 0.15 | Median sharpness (var. of Laplacian), exposure error, clipped pixels, detection recall and precision against ground truth (sim) |

Hard rules:

* **Any collision caps the overall score at 50** and the verdict becomes `FAIL — collision`.
* Not landing (crash, timeout) is a `FAIL`. Missing waypoints is `PARTIAL`.
* Grades: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F below.

## What the run directory contains

| File | What it shows |
|---|---|
| `flight.mp4` | Onboard camera with HUD (time, mode, altitude, speed, battery, nearest obstacle) and AI detection boxes |
| `snapshots/*.jpg` | Raw frames saved whenever the AI sees something new |
| `flight_map.png` | Top-down map: obstacles, targets, waypoints and the flown path coloured by flight mode |
| `timeline.png` | Altitude, speed, tilt and nearest-obstacle distance over time |
| `telemetry.jsonl` | Every state (25 Hz) and vision record, for independent analysis |
| `mission.json` | Mission outcome, event log and failsafes |
| `report.md` / `report.json` | The scorecard |

## Suggested live demo (about 5 minutes)

1. `python -m pytest -q`: shows the safety-critical behaviour under test
   (tilt recovery, disturbance rejection, narrow-gap rejection, low-battery RTL, climbing a wall, …).
2. `python -m skymind sim`: the forest mission. Open `flight_map.png` and `flight.mp4`.
3. `python -m skymind sim --config config/windy_city.yaml`: strong gusty wind, a
   3 m alley between buildings that the mapper and A* planner find, and a wall that
   blocks the whole site, so the drone decides to climb over it.
4. Open `report.md` and walk through the scorecard.
5. (Hardware) Bench run with props off: move a board around the lidar and show the
   avoidance mode change in the recorded HUD.
