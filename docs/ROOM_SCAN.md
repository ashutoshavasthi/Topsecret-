# Room scan

Say **"scan the room"** (or press **▦ Scan room** in the app, or run `python -m skymind survey`).
The drone explores by itself and produces a scan of the room:

* **Obstacles:** walls, furniture and poles, each with position and size, plus what the camera saw on them.
* **Doorways and exits:** each with an estimated width and a passable / too-narrow verdict *for this
  drone*. A doorway that leads into unexplored space is reported as an **exit**.
* **Tight gaps** between objects that are worth knowing about.
* **Low obstacles** such as tables, found by the downward sensor, with the minimum safe height to
  fly over them.
* **Objects the camera recognised** (markers, or people, chairs and bags with YOLO), placed on the map.
* **Unexplored areas**, so later flights treat them with caution.

It's saved as `room_scan.png` (picture), `room_scan.json` (data) and `room_map.npz` (map for
later flights). Doorways become places you can name: "go through doorway 1".

## How it works

**Exploration (frontier-based).** The drone looks around (three 120° turns), then flies to the
best viewpoint near the border between mapped free space and the unknown, and repeats until
nothing reachable is left. Viewpoints are always in *known* free space, set back from the
unknown. The drone stops short of space it hasn't seen, faces it, and looks; it never flies into it.
While exploring, routes keep an extra 10 cm from mapped obstacles. A half-mapped chair beside a
narrow gap is invisible to forward-only sensors, so it doesn't squeeze through gaps that barely fit.

**Safety rules learned in testing.** Each of these fixed a real crash in simulation:
* The downward sensor's *cone* (not just the point under the drone) sees a table edge before the props do.
* A cone-shaped ToF also sees things slightly below it, so chairs don't vanish from the map
  when the drone climbs over a table.
* Hovering still keeps its distance. If something shows up inside the safety margin, the hover
  point eases away, slowly and by at most 0.3 m.
* When blocked, it holds the exact spot rather than "zero velocity", which drifts.
* Escaping from something too close is always allowed, even in directions the sensors can't
  see; only goal-seeking motion waits until the nose points where it's going.

**Dual inputs: ranging + vision.**

* *Bounce-off ranging* (ToF / lidar) builds the occupancy map: where things are and how big.
* *Vision* says what things are. Camera sightings are placed by **triangulation**: the bearing
  to an object from two or more positions pins down where it is, as a surveyor does. This works
  beyond the ToF's range and isn't fooled by a door frame between the drone and the object.
  In the test room, three markers were located within 1–3 cm of their true positions.

**Doorway detection.** For every gap between two obstacles, the scan asks: *if this gap were
closed, would the free space split in two?* If so, and the gap is at least 0.5 m wide with a wall
on at least one side, it's a doorway. Anything attached to the room's edge counts as a wall
section, even if only part of it was seen. If one side is still unexplored, it's an exit. Width uses
two estimates: a best estimate bounded only by strongly confirmed walls (cheap ToF cones smear
wall edges), and a conservative one. "Passable" is judged conservatively.

**Low obstacles.** The downward range sensor's footprint sees a table edge before the props reach
it. The drone climbs automatically to keep 30 cm above it (terrain following). The scan records
the table's position and height.

## Results (simulated ₹5,000 Nano, 8 × 5 m room with a 1 m doorway)

| | |
|---|---|
| Area explored | 83–97 % of the room, in about 3 min of flight (about half the battery) |
| Collisions | 0 in 12 runs on the final code (10 surveys + 2 test-suite runs) |
| Doorway | found in every run, at the right place, estimated 0.9–1.05 m (true 1.0 m) |
| Gaps | chair to wall 0.7 m (true 0.65), lamp to cupboard 1.0 m (true 1.1) |
| Table | found as a low obstacle, 0.75 m, so it flies at 1.05 m or higher over it |
| Markers | all 3 triangulated within 1–3 cm |

## Using the scan for later flights

```bash
python -m skymind survey --config config/nano_room.yaml --out runs/my-room
python -m skymind sim --config config/nano_room.yaml --prior-map runs/my-room/room_map.npz
python -m skymind app --prior-map runs/my-room/room_map.npz
```

With the map loaded, the planner routes through doorways and around furniture from the first
second. In the simple test room the measured gain is modest: about 3 % faster, with more
clearance. In the windy-city layout, paths were 6 % shorter but flights 6 % slower. The live
sensors already handle these layouts well. The scan's bigger value is what it *tells* you and the
drone: where the exits are, which gaps fit, how high to fly over furniture, and where objects are.

**Map memory** (`avoidance.map_memory`, off by default) makes mapped obstacles act as virtual
sensor readings in the forward-only sensors' blind spots. In testing it made flights slower and,
with laggy WiFi, less safe, because smeared cone readings made doorways look narrower. The other
fixes made it unnecessary, so it stays off.

## Limits

* The map is 2D at flight height, plus low obstacles from the downward sensor. Things hanging
  from the ceiling aren't mapped.
* Cheap ToF cones blur edges, so widths are estimates (±0.2 m). Treat "passable" as advice, and
  fly slowly through the first time.
* Glass, mirrors and black velvet return little infrared, so they can be missed. Vision helps with some of these.
