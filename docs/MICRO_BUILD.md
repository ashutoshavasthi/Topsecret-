# SkyMind Micro: small, quiet, cheap build (~$230, under 250 g)

A palm-sized version of SkyMind for college projects. It runs the same
software as the big drone (self-balancing, obstacle avoidance, mapping, camera AI
and flight judging) on about $230 of parts.

| | Big build (`default.yaml`) | **Micro** (`micro.yaml`) | **Micro Quiet** (`micro_quiet.yaml`) |
|---|---|---|---|
| Size | 450 mm, 10" props | 150 mm, 3.5" props | 175 mm, 4" props |
| All-up weight | 1200 g | 235 g | 240 g |
| Approx. parts cost | ~$700 | ~$230 | ~$235 |
| Obstacle sensing | RPLidar 360° (170 g, ~$100) | 6 × VL53L1X ToF (9 g, ~$35) | same |
| Companion computer | Raspberry Pi 5 | Raspberry Pi Zero 2 W | same |
| Hover power | 134 W | 51 W | **42 W** |
| Hover endurance (to 25 % reserve) | 14.3 min | 8.5 min | **8.7 min** |
| Rotor tip speed (noise) | 69 m/s | 87 m/s | 77 m/s (**−2.7 dB** vs Micro) |
| Thrust/weight | 2.7 | 4.5 | 4.4 |

Numbers come from `python -m skymind design config/micro.yaml config/micro_quiet.yaml config/default.yaml`
(momentum-theory power model, `skymind/power.py`). Treat them as design estimates and
measure the real ones on your first flights.

**Recommendation:** build **Micro Quiet**. Bigger, slower props are quieter *and*
more efficient. The 3.5" version is only worth it if 175 mm is too big for your space.

## Parts list

Prices are typical online prices in USD and vary by store and region.

| Part | Example | Price | Weight |
|---|---|---:|---:|
| Frame | 4" (Quiet) or 3.5" carbon frame, 3 mm arms | $15–25 | 25–30 g |
| Flight controller + ESC | SpeedyBee F405 Mini stack (F405 FC + 4-in-1 ESC) | $55–65 | 12 g |
| Motors ×4 | 1404 2750–3000 KV (4") or 1404 3800 KV (3.5") | $35–45 | 36 g |
| Props | Gemfan 4024 bi-blade (4") or 3520 bi-blade (3.5"), 2 sets | $5 | 4 g |
| Battery | 4S 550 mAh (Quiet) / 4S 650 mAh, XT30 | $12–15 | 62–75 g |
| Companion computer | Raspberry Pi Zero 2 W, headers removed | $15 | 11 g |
| Camera | Pi Camera OV5647 (v1.3 clone) or Camera Module 3 | $8–25 | 3 g |
| Obstacle sensors | 6 × VL53L1X breakout (4 m ToF) | ~$35 | 9 g |
| Position + height | MATEK 3901-L0X optical flow + ToF | ~$30 | 3 g |
| RC receiver | ExpressLRS nano | $12–15 | 1 g |
| microSD | 32 GB, A1 class | $6 | — |
| Wire, XT30, standoffs, zip ties | | $5 | 10 g |
| **Total** | | **≈ $230** | **≈ 190–205 g** + optional 15 g prop guards |

You'll also need an ELRS transmitter (~$50–60, often borrowable from a campus
drone club), a LiPo charger and a LiPo-safe bag.

### Why these parts

* **VL53L1X instead of a lidar**: about 25× lighter and 3× cheaper. Each sensor sees a
  27° cone up to ~4 m indoors (~2 m in bright sunlight). Three face forward
  (−30°, 0°, +30°, covering ±43° almost without gaps), plus one left, one right and one rear.
  The software knows where the blind gaps are and uses **turn-then-go**: it only
  moves fast in directions a sensor is looking at, and turns to face anything else first.
* **Optical flow instead of GPS**: holds position indoors (classrooms, labs, hallways),
  where college projects are usually demonstrated.
* **Pi Zero 2 W**: 11 g, about 1.5 W. `python -m skymind bench --config config/micro_indoor.yaml`
  measures the load. The full onboard stack (mapping, A*, avoidance, vision) takes ~1.5 % of one
  core on a desktop CPU, so there's plenty of headroom on the Zero's four cores. Run the bench on
  the Pi itself to get its real numbers.

## Weight budget (Micro Quiet)

```
frame 30 + stack 12 + motors 36 + props 4 + battery 62 + Pi Zero 11 + camera 3
+ ToF 9 + flow 3 + receiver 1 + wiring 10                           = 181 g
+ prop guards (recommended indoors)                                 =  15 g
+ margin for tape, mounts, heavier parts                           ≈  44 g
                                                                     ≤ 240 g
```

## Wiring

```
4S LiPo ─► XT30 ─► 4-in-1 ESC ─► FC (8-pin harness)
FC 5 V (2 A BEC) ─► Pi Zero 2 W 5 V + GND
FC UART2 TX/RX  ◄─► Pi GPIO15/14 (RX/TX)        MAVLink 921600
FC UART3        ◄─► MATEK 3901-L0X              MSP
FC UART1        ◄─► ELRS receiver
Pi I2C (GPIO2 SDA, GPIO3 SCL, 3.3 V, GND) ─► all six VL53L1X
Pi GPIO17,27,22,23,24,25 ─► XSHUT of sensors at −90°,−30°,0°,30°,90°,180°
Pi CSI ─► camera (Pi Zero needs the narrow 22-pin cable)
```

Keep I2C wires short (< 15 cm) and twisted with GND. Mount the ToF sensors
on the frame edges, in the plane of the props, not looking at them.

## Flight controller setup (ArduPilot Copter)

Flash ArduPilot with the board's target (check that your exact FC is on the
ArduPilot supported-board list; Matek H743-Mini is a well-supported alternative).
Betaflight is **not** suitable: it has no autonomous GUIDED mode.

```
FRAME_CLASS = 1, FRAME_TYPE = 1         # quad X
SERIAL2_PROTOCOL = 2, SERIAL2_BAUD = 921 # companion (Pi Zero)
SERIAL3_PROTOCOL = 32                    # MSP for the 3901-L0X
FLOW_TYPE = 7                            # MSP optical flow
RNGFND1_TYPE = 32                        # MSP rangefinder
EK3_SRC1_POSXY = 0, EK3_SRC1_VELXY = 5   # optical flow for horizontal velocity
EK3_SRC1_POSZ = 2                        # rangefinder height (indoors)
PRX1_TYPE = 2                            # proximity from MAVLink (SkyMind forwards the ToF ring)
GUID_TIMEOUT = 3
```

Parameter numbers change between ArduPilot versions, so double-check each one
against the documentation for the version you flash. Some 1 MB-flash F405
builds leave out features such as proximity. SkyMind's own avoidance runs on
the Pi regardless; the autopilot's proximity layer is a bonus.

## Software on the Pi Zero 2 W

```bash
sudo apt install python3-venv python3-opencv python3-picamera2
python3 -m venv --system-site-packages ~/skymind && source ~/skymind/bin/activate
pip install pymavlink adafruit-blinka adafruit-circuitpython-vl53l1x
pip install -e .
python -m skymind bench --config config/micro_indoor.yaml           # check the CPU budget
python -m skymind fly --config config/micro_indoor.yaml --connect /dev/serial0   # bench mode
```

Efficiency settings already in `micro.yaml`: 320×240 at 10 fps, detection on every
2nd frame, slower A* replanning (1.5 s). For YOLO on the Zero, export a small model
(`yolo export model=yolov8n.pt format=onnx imgsz=320`) and set `vision.onnx_input_size: 320`.
Expect only a few frames per second. For heavier AI, run `skymind analyze` on the
recorded video with a laptop after the flight.

## Making it quiet

Noise is mostly set by **weight** and **prop size**, not by how the drone flies.
The simulator confirms this: flying gently changes rotor tip speed by under 1 %, while
switching from 3.5" to 4" props cuts it by 11 % (≈ −2.7 dB).

1. Use the largest props the frame allows, at lower KV (the Micro Quiet build).
2. Use bi-blade rather than tri-blade props, and balance them.
3. Shave grams: every 10 g costs ~2 % more power and tip speed. Check `skymind design … --sweep-battery`.
4. Enable bidirectional DShot and the RPM notch filter in ArduPilot. That gives a smoother
   motor sound, less vibration and sharper video.
5. Soft-mount the motors and the camera with rubber grommets.

For a low visual profile, use a matte black frame and dark props.
Keep Remote ID and any legally required lights on. "Stealthy" here means
quiet and unobtrusive for demos and wildlife or indoor filming, not hard to identify.
Don't fly over people or record them without consent, and check your country's and
campus's drone rules before flying.

## Bring-up checklist

1. `python -m skymind sim --config config/micro_indoor.yaml` must PASS.
2. Props off, `skymind fly` in bench mode: wave a hand in front of each ToF
   sensor and check the readings in the recorded HUD ("obst").
3. Manual hover in Loiter. Run ArduPilot autotune with the guards on.
4. First autonomous flight in an empty room, with a single waypoint, a pilot on the mode switch,
   and prop guards fitted.
5. Then add obstacles. Cardboard boxes and pool noodles are cheap and soft.
