# Dodging thrown objects

**Goal:** the drone dodges things thrown at it, including paper, plastic and foam,
which don't conduct electricity.

## Which sensors can see non-conducting objects?

Every sensor that works at a distance detects **reflections**: light, infrared,
radio or sound bouncing back. Paper and plastic reflect all of them, so conductivity doesn't matter.

| Signal | Example part (₹, approx.) | Sees a paper ball? | Speed | Verdict |
|---|---|---|---|---|
| Infrared time of flight, multizone | VL53L5CX / L7CX (₹1,200–2,200) | Yes, white paper reflects IR well | 60 Hz, ~1.4 m for a 8 cm ball | ✅ **used**: precise distance, works in the dark |
| Camera "looming" (vision) | ESP32-CAM you already have + 120° lens (₹150–300) | Yes, in light | 30–40 fps, sees it from ~3.5 m | ✅ **used**: earliest warning, widest view |
| mmWave radar 24/60 GHz | HLK-LD2450 (~₹800–1,200), TI IWR6843 (₹15k+) | Weakly: paper/plastic reflect radar poorly; people, water and metal well | Cheap modules ~10 Hz | ⚠️ good for people approaching, too slow and coarse for a ball |
| Ultrasonic | HC-SR04 (₹60) | Poorly for small soft objects | Slow (speed of sound, ~20 Hz) | ❌ |
| Event camera | Prophesee/iniVation (₹50k+) | Yes, microsecond latency | Fastest available | ✅ best, but research budget |
| Magnetic / capacitive / "resonance" | — | Only at a few cm, or inside a strong magnet (MRI) | — | ❌ physically can't reach across a room |

So the build fuses **two reflection-based inputs**: infrared ToF for exact distance, and
vision looming for early, wide warning. That's the "bouncing-off signals plus vision"
dual-input design.

## How the reflex works

```
VL53L5CX zones ─┐                          ┌─► escape: drop / climb / sidestep
                ├─► threat tracker ──► predict where the ball will be (gravity included)
camera looming ─┘   (weighted ballistic    │   will it pass within the drone's radius + margin?
                     least-squares fit)    └─► pick the escape with the biggest predicted miss
```

1. **Separate thrown objects from the room.** A ToF zone only counts as a threat if its distance
   shrinks much faster than the drone's own motion explains. Walls you fly toward don't trigger it.
2. **Track with physics.** Recent measurements are fitted to a ballistic arc (gravity is known),
   with precise ToF readings weighted more than rough camera-size ranges. Once the arc is known,
   the reflex keeps predicting it even when the ball leaves the sensors' view, e.g. a lob
   dropping in from above.
3. **Choose the escape like a model-predictive controller.** Each candidate (drop, climb, left,
   right, away from the ball's line and diagonals) is simulated with the airframe's real thrust,
   tilt limit, motor lag and top speed. Escapes that would hit the floor or ceiling are excluded,
   and the largest predicted miss wins. Dropping wins ties, because gravity needs no spare thrust.
4. **Run on the drone, not the laptop.** Over WiFi the decision arrives about 150 ms late,
   and that alone halves the dodge rate (see below).

## Simulated results

`python -m skymind dodge` reproduces these. There are 20 throws per speed, aimed to hit, from
within ±45° of the nose, using the ₹5,000 Nano airframe. With 20 throws, each percentage is
roughly ±10 points.

| Setup | 3 m/s | 5 m/s | 7 m/s | 9 m/s |
|---|---:|---:|---:|---:|
| No reflex (thrown from 2.5 m) | 30 % | 5 % | 0 % | 0 % |
| Camera looming only, stock 66° lens | 60 % | 80 % | 55 % | 0 % |
| **ToF + 120° lens looming, reflex onboard** | **100 %** | **90 %** | 30–50 % | 0 % |
| Same, but reflex on the laptop over WiFi | 95 % | 40 % | 0 % | 0 % |
| + brushless motors (more thrust, faster spin-up) | 95 % | 95 % | 55 % | 5 % |
| ToF + wide looming, thrown from 4 m | 70 % | 100 % | 80 % | 60 % |
| + brushless, thrown from 4 m | 70 % | 100 % | 85 % | 65 % |
| ToF + wide looming, throws from any direction | 65 % | 50 % | 15 % | 0 % |
| + second ESP32-CAM facing backwards, any direction | 80 % | 65 % | 35 % | 0 % |

In the live app's simulator, the **Throw ball** button dodged 6 of 6 throws at 4–6 m/s.

What sets the limit is **physics, not software**. A 9 m/s throw from 2.5 m arrives in 0.28 s.
Brushed motors need about 0.1 s just to change thrust, and even free fall only moves the drone
about 15 cm in 0.18 s. More distance helps most (4 m → 60 % at 9 m/s), then more thrust.
Slow lobs are hard in a different way: they drop in from above, outside the sensors' view,
and only the arc prediction catches them.

If a ball *does* hit, a 5 g paper ball only shoves an 80 g drone, and the self-balancing
controller recovers. Use prop guards.

## Parts to add to the Nano (₹)

| Part | ₹ (approx.) | Why |
|---|---:|---|
| VL53L5CX (or VL53L7CX for 90°) multizone ToF, facing forward | 1,200–2,200 | Precise distance to thrown objects, works in the dark |
| 120° M12 wide-angle lens for the ESP32-CAM | 150–300 | Sees throws from the sides (the biggest single improvement) |
| Optional: second ESP32-CAM + lens facing backwards | 600–900 | Covers throws from behind |
| Optional: 1S brushless whoop motors + AIO board | 2,500–4,000 | More thrust, faster response (over the ₹5,000 budget) |

The VL53L5CX shares the ESP32-S3's I2C bus with the flow sensor. Mount it facing forward
at prop height.

## Firmware status

* **Reflex core: done and tested.** `firmware/dodge_reflex/dodge_reflex.c` is plain C99 (no
  dynamic memory, floats only). It contains the tracker, the escape chooser and the ToF
  zone → threat filter. The test suite compiles it with the host C compiler and checks that it
  makes the same decisions as the Python version on 300 random throws.
* **Still to do on hardware: integration into ESP-Drone.** Add a FreeRTOS task on the ESP32-S3
  that reads the VL53L5CX (ST's ULD driver: 4×4 mode, 60 Hz) and receives looming detections
  from the ESP32-CAM over a UART wire (not WiFi), then calls `dr_measure` and `dr_check`. When
  it fires, override the commander with a high-priority velocity setpoint along `escape_dir`
  for `escape_time`, then hold. Raise ESP-Drone's position-controller velocity limits so the
  escape isn't clipped. This glue depends on your ESP-Drone version's APIs, so it isn't written yet.
* **Still to do on hardware: the looming detector on the ESP32-CAM.** Frame differencing on
  QQVGA/QVGA grayscale, the largest moving blob, its growth rate and bearing, sent over UART.
  The simulator models its output (`SENSORS["looming"]` in `skymind/dodge.py`).

Test only with soft objects (paper balls, foam), prop guards on, and nobody close by.
