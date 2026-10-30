# Hardware build guide

SkyMind is split across two computers, the same way commercial drones are built:

| Layer | Runs on | Rate | Job |
|---|---|---|---|
| Flight controller | Pixhawk-class board running **ArduPilot** or **PX4** | 400–1000 Hz | IMU fusion, attitude and rate control, motor mixing. This is the self-balancing layer. |
| Companion computer | **Raspberry Pi 5** or **Jetson Orin Nano** running SkyMind | 20 Hz autonomy, 10–30 fps vision | Mission logic, lidar mapping and avoidance, camera recording, AI detection, flight judging |

The simulator runs *both* layers in Python (`skymind/control/` stands in for
the firmware), so the algorithms you demonstrate in simulation are the same
ones used on the real drone. The only difference is that the inner balancing
loops are handed to battle-tested firmware on real hardware.

## Reference bill of materials (~450 mm quad, 1.2–1.5 kg AUW)

| Part | Suggested model | Notes |
|---|---|---|
| Frame | Holybro S500 V2 / F450 clone | Plenty of room for the companion computer and lidar |
| Flight controller | Holybro Pixhawk 6C / Matek H743 | Runs ArduPilot Copter 4.5+ or PX4 1.14+ |
| GPS + compass | Holybro M10 GPS | Gives position hold outdoors. For indoors, use an optical-flow + rangefinder module instead |
| Motors | 4 × 2216 880 KV | With 10" props: ~8 N max thrust each, matching `vehicle.max_motor_thrust` |
| ESCs | 4 × 30 A BLHeli_S / 4-in-1 | DShot600 |
| Propellers | 1045 (10 × 4.5) | 2 CW and 2 CCW |
| Battery | 4S 5000 mAh LiPo (~74 Wh) | `battery.capacity_wh` in config |
| Power module | Holybro PM02 | Battery voltage and current reporting to the FC |
| Companion computer | Raspberry Pi 5 (8 GB) or Jetson Orin Nano | Jetson gives about 10× faster YOLO |
| Camera | Raspberry Pi Camera Module 3, or a USB global-shutter camera | Mount on a 2-axis gimbal for sharp footage |
| Gimbal | Any 2-axis brushless gimbal (e.g. Storm32) | Matches `camera.gimbal_stabilized: true` |
| 360° lidar | Slamtec RPLidar A1/C1 (12 m) | Mount on top, 0° mark facing the nose |
| Downward rangefinder | Benewake TFmini-S | Precise altitude hold and landing |
| Telemetry radio | SiK 433/915 MHz pair | Ground station link (Mission Planner / QGroundControl) |
| RC transmitter + receiver | Any ELRS / FrSky set | **Mandatory** manual override for safety |

## Wiring

```
Battery ──► PM02 ──► Pixhawk POWER1
                └──► 5 V BEC (5 A) ──► Raspberry Pi 5
Pixhawk TELEM2 (921600 baud) ◄──► Pi UART (/dev/ttyAMA0)   [TX↔RX, GND]
RPLidar  ──► Pi USB (/dev/ttyUSB0)
Camera   ──► Pi CSI (or USB)
TFmini-S ──► Pixhawk serial 4 (RNGFND1_TYPE = 20)
```

## Flight controller parameters (ArduPilot)

```
SERIAL2_PROTOCOL = 2        # MAVLink2 on TELEM2 for the companion
SERIAL2_BAUD     = 921      # 921600
PRX1_TYPE        = 2        # proximity from MAVLink (SkyMind forwards its lidar)
AVOID_ENABLE     = 7        # FC's own fence + proximity avoidance as a 2nd safety layer
OA_TYPE          = 0        # SkyMind does path planning, not the FC
FENCE_ENABLE     = 1
FS_GCS_ENABLE    = 1
GUID_TIMEOUT     = 3        # if the companion stops sending setpoints, hold position
```

## Companion computer setup

```bash
sudo apt install python3-venv python3-opencv
python3 -m venv ~/skymind-venv && source ~/skymind-venv/bin/activate
pip install -r requirements-hardware.txt
pip install -e .
```

Optional AI model (people, cars, animals and so on):

```bash
pip install ultralytics            # on a laptop, not needed on the drone
yolo export model=yolov8n.pt format=onnx imgsz=640
scp yolov8n.onnx pi@drone:~/skymind/models/
```

Then set `vision.onnx_model: models/yolov8n.onnx` and `vision.onnx_labels`
(the 80 COCO class names) in your config.

## Bring-up sequence (do not skip steps)

1. **Simulation**: `skymind sim` must PASS with your config.
2. **Bench, props OFF**: `skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0`.
   Without `--arm` it never arms. Walk around the drone with a board and check
   in the recording that the lidar ranges and avoidance modes react correctly.
3. **Manual flight**: fly it in Stabilize/Loiter from the RC transmitter. Tune the
   firmware PIDs (ArduPilot autotune) until hover is rock solid.
4. **First autonomous flight**: open field, no obstacles, a single nearby
   waypoint, pilot ready on the mode switch:
   `skymind fly --connect /dev/ttyAMA0 --lidar /dev/ttyUSB0 --arm`.
5. Add obstacles gradually (foam pool noodles on stands are ideal).
6. After every flight: `skymind judge runs/flight-…` and compare the scores.

Know your local drone regulations (registration, line-of-sight, altitude limits)
before flying outdoors.
