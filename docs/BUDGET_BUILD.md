# SkyMind Nano: the ₹5,000 build

An 80 g indoor drone that balances itself, avoids obstacles, records video and
runs AI detection, for **about ₹4,000–5,000** in parts. Buy at the top of every price
range and it can reach ₹6,100, so see "Staying under ₹5,000" below.

The trick that makes it this cheap is an efficient **split brain**:

```
            DRONE (₹ ~4,700, 80 g)                                YOUR LAPTOP (₹0 extra)
 ┌──────────────────────────────────────────────┐     WiFi     ┌──────────────────────────────────┐
 │ ESP32-S3 + MPU6050 running ESP-Drone         │◄── hover ────│ SkyMind brain: mapping, A*,      │
 │  · self-balancing (attitude/rate PID, 500 Hz)│  setpoints   │ obstacle avoidance, mission,     │
 │  · optical-flow position + height hold       │── telemetry ►│ failsafes                        │
 │  · watchdog: hovers/stops if link goes quiet │              │                                  │
 │ ESP32-CAM                                    │── MJPEG ────►│ Full-size YOLO / colour AI,      │
 │  · 320×240 video, 3 × VL53L0X distance       │── ToF UDP ──►│ recording, the flight judge      │
 └──────────────────────────────────────────────┘              └──────────────────────────────────┘
```

* The **drone** does only the fast, simple, safety-critical work: staying level and holding position.
* The **laptop** does everything heavy, so the AI is far more powerful than any ₹5,000 onboard chip.
  That costs nothing extra and adds no weight.
* The WiFi delay is budgeted into every braking decision (`avoidance.reaction_time`).
  The simulator models latency and packet loss (`sim.link_latency`, `sim.link_drop_prob`).

## Simulated performance

| Test | Runs | Landed | Collisions | Mean score |
|---|---:|---:|---:|---:|
| `config/nano_room.yaml`: 8 × 5 m room, 1 m doorway, chairs, table, cupboard (60 ms one-way lag, 5 % packet loss) | 8 | 8/8 | 0 | 94.6 |
| Same room, bad WiFi (150 ms one-way, 20 % loss) | 6 | 6/6 | 0 | 93.1 |

Design estimates (`python -m skymind design config/nano.yaml`): 80 g, thrust/weight 2.0,
21 W hover, **~4.7 min** flight to a 25 % reserve on a 1S 600 mAh pack (typical for brushed
micro drones). Measure the real figures on your first flights.

## Parts list (prices in ₹, approximate)

Typical prices at Robu.in, Robocraze, Amazon.in or electronics markets (Lamington Road,
SP Road, Nehru Place, Ritchie Street). They change often, so shop around.

| # | Part | Notes | ₹ |
|---|---|---|---:|
| 1 | ESP32-S3 board (SuperMini / small dev board) | Runs ESP-Drone (flight control) | 450–650 |
| 2 | MPU6050 (GY-521) | IMU for balancing | 120–200 |
| 3 | PMW3901 optical-flow module | Position hold indoors | 1,100–1,500 |
| 4 | VL53L1X breakout (pointing down) | Height for the flow sensor | 450–650 |
| 5 | ESP32-CAM (AI-Thinker, OV2640) | Video + reads the obstacle sensors | 450–650 |
| 6 | 3 × VL53L0X breakout | Obstacle sensors: left, centre, right | 450–700 |
| 7 | 4 × 8520 coreless motors (2 CW + 2 CCW) | Brushed, ~5 g each | 400–650 |
| 8 | 65 mm props, 2 sets | One set is spare | 100–150 |
| 9 | 4 × SI2302 (or AO3400) MOSFET, 4 × 1N5819 diode, 4 × 10 kΩ, 4 × 100 Ω | Motor drivers | 50–100 |
| 10 | 1S 3.7 V 600 mAh LiPo, 20C or more, JST-PH | Battery (buy 2 if you can) | 250–400 |
| 11 | TP4056 USB-C 1S charger | Charging | 30–60 |
| 12 | MT3608 boost (5 V for the ESP32-CAM) + ME6211/HT7333 3.3 V LDO | Power | 60–100 |
| 13 | 470 µF low-ESR capacitor, wire, JST plugs, headers, tape | | 100–150 |
| 14 | Frame: 3D-printed PLA, ~120 mm, with prop guards | Free in most college labs | 0–150 |
| | **Total** | | **≈ 4,000–6,100** |

**Staying under ₹5,000:**

* Buy near the low end of each range. Combo listings (e.g. PMW3901 sold with a VL53L1X) are often cheaper.
* Borrow what your lab already has: ESP32 boards, a USB-serial programmer for the ESP32-CAM, the soldering kit.
* Only one battery at first.
* **Staged build**: parts 1, 2, 7–14 (≈ ₹1,600–2,600) already make a self-balancing drone
  you can fly from your phone or laptop. Add the camera and sensors (5, 6: ≈ ₹900–1,350)
  and then optical flow (3, 4: ≈ ₹1,550–2,150) when the budget allows.
  Autonomous avoidance needs all of them.

No RC transmitter is needed (it would cost more than the drone). The laptop is the
controller, and a phone running the ESP-Drone app can fly it manually.

## Wiring

```
1S LiPo (+) ──┬── 470 µF ──┬── 4 × motor (+)
              │            ├── ME6211 3.3 V LDO ──► ESP32-S3 3V3, MPU6050, PMW3901, VL53L1X
              │            └── MT3608 boost 5.0 V ──► ESP32-CAM 5V
1S LiPo (−) ── common GND for everything

Each motor: motor (−) ─► MOSFET drain; source ─► GND; gate ◄─ 100 Ω ◄─ ESP32-S3 motor GPIO
            10 kΩ gate→GND pull-down; 1N5819 across the motor (stripe to +)
ESP32-S3:  I2C ─► MPU6050 ; SPI ─► PMW3901 ; I2C ─► VL53L1X (down)   [pins: your ESP-Drone board config]
ESP32-CAM: GPIO14 SDA, GPIO15 SCL ─► 3 × VL53L0X ; GPIO2 ─► XSHUT left ; GPIO13 ─► XSHUT right
```

Mount the three VL53L0X at −35°, 0° and +35° (right, centre, left), at prop height, looking
out past the prop guards. Mount the flow sensor underneath, lens down, at the centre.

## Software

**1. Flight board: ESP-Drone** (Espressif, open source, GPL-3.0, a port of the Crazyflie firmware)

* Install ESP-IDF, clone `github.com/espressif/esp-drone`, select the ESP32-S3 target,
  set your motor, I2C and SPI pins in its board configuration, then enable the optical-flow
  driver (PMW3901 + VL53L1X). Build and flash it. Follow the ESP-Drone documentation for the
  exact steps for your IDF version.
* The drone starts a WiFi access point (`ESP-DRONE_xxxx`, password `12345678` by default).
  If its access point only accepts one client, raise the station limit in the firmware's WiFi
  settings so both the laptop and the ESP32-CAM can join.
* Tune the PID gains with the ESP-Drone app or cfclient before trying autonomy.

**2. Sensor head: ESP32-CAM.** Flash `firmware/esp32cam_sensors/esp32cam_sensors.ino`
with the Arduino IDE (board "AI Thinker ESP32-CAM", library "VL53L0X" by Pololu).
Put your drone's AP name in `WIFI_SSID`. The serial monitor prints the stream URL.
Copy it into `camera.device` in `config/nano.yaml`.
*This sketch has not been compiled or run on hardware yet. Expect to fix small things
on the first build.*

**3. Laptop: SkyMind**

```bash
pip install -r requirements.txt cflib
python -m skymind sim --config config/nano_room.yaml           # rehearse in the simulator
# join the drone's WiFi, then: PROPS OFF, bench mode (never spins the motors)
python -m skymind fly --config config/nano_room.yaml --esp-drone udp://192.168.43.42
# real autonomous flight
python -m skymind fly --config config/nano_room.yaml --esp-drone udp://192.168.43.42 --arm
```

In bench mode, check:

1. Wave a hand in front of each ToF sensor. The HUD "obst" distance must react, and
   the *right-hand* sensor is the first number in the UDP packet.
2. Tilt the drone and turn it by hand, and watch the logged roll, pitch and yaw. The sign
   constants `PITCH_SIGN` and `YAWRATE_SIGN` in `skymind/hal/esp_drone.py` must match
   your firmware version.
3. Unplug the ESP32-CAM: within 0.3 s every direction should read 0 m ("blocked"),
   so the drone would hold position rather than fly blind.

## Safety

* Fly indoors only, with prop guards, below 1 m at first, in a room with nobody else in it.
* **Ctrl+C** on the laptop lands the drone. If WiFi drops, ESP-Drone's watchdog levels it
  and then cuts the motors after about 2 s. An 80 g drone falling from 1 m onto a floor
  is usually fine, but never fly it over people.
* LiPo safety: charge with the TP4056 only, never leave a charging pack alone, and store
  packs at storage voltage (3.8 V) in a LiPo bag.

## How it compares

| | **Nano** (this) | Micro Quiet | Full size |
|---|---|---|---|
| Cost | **≈ ₹4,700** | ≈ ₹19,000 ($230) | ≈ ₹58,000 ($700) |
| Weight | 80 g | 240 g | 1,200 g |
| Where the AI runs | Laptop over WiFi | Pi Zero 2 W onboard | Pi 5 onboard |
| Obstacle sensing | 3 × VL53L0X, forward only, 1.5 m | 6 × VL53L1X, 3.5 m | 360° lidar, 8 m |
| Flight time | ~4.7 min | ~8.7 min | ~14 min |
| Range from laptop | WiFi (~20–30 m indoors) | Onboard, any range | Onboard, any range |
