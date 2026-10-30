"""High-level drone skills the assistant can use.

Every skill goes through the live session, and so through the flight brain's
avoidance, geofence, battery and link failsafes. None of them can disable those.
Skills block until done (or timed out / cancelled) and return a short,
speakable result.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from skymind.live import LiveSession
from skymind.math3d import wrap_angle


@dataclass
class SkillResult:
    ok: bool
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    observation: bool = False  # True when the result answers a question (worth summarising)


@dataclass
class Skill:
    name: str
    description: str
    args: dict[str, str]  # arg name -> "type: meaning"
    fn: Callable[..., SkillResult]


def relative_direction(rel: float) -> str:
    deg = math.degrees(rel)
    if abs(deg) <= 15:
        return "straight ahead"
    if abs(deg) >= 150:
        return "behind me"
    side = "left" if deg > 0 else "right"
    return f"about {abs(deg):.0f} degrees to my {side}"


def _words(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if w not in {"the", "a", "an", "my"}}


class DroneSkills:
    def __init__(self, session: LiveSession, cancel: threading.Event | None = None):
        self.s = session
        self.cancel = cancel or threading.Event()
        self.cfg = session.cfg
        self.skills: dict[str, Skill] = {}
        for sk in [
            Skill("status", "Report battery, height, position and what is nearby.", {}, self.status),
            Skill("takeoff", "Take off and hover.", {"altitude": "number (m), optional"}, self.takeoff),
            Skill("land", "Land right here.", {}, self.land),
            Skill("return_home", "Fly back to the takeoff point and land.", {}, self.return_home),
            Skill("hold", "Stop and hover in place.", {}, self.hold),
            Skill("go_to", "Fly to a named place.", {"place": f"string, one of: {', '.join(self.cfg.mission.places) or 'none'}"}, self.go_to),
            Skill("go_to_xy", "Fly to map coordinates.", {"x": "number (m)", "y": "number (m)"}, self.go_to_xy),
            Skill("move", "Move relative to where I am facing.", {"direction": "forward|back|left|right", "meters": "number"}, self.move),
            Skill("turn", "Rotate in place. Positive = left.", {"degrees": "number"}, self.turn),
            Skill("set_altitude", "Climb or descend to a height.", {"meters": "number"}, self.set_altitude),
            Skill("climb", "Go up (positive) or down (negative) by some metres.", {"meters": "number"}, self.climb),
            Skill("scan", "Rotate a full circle and report everything the camera recognises.", {}, self.scan),
            Skill("find", "Look for an object; if found, face it.", {"object": "string, e.g. 'red marker', 'person', 'chair'"}, self.find),
            Skill("approach", "Find an object and fly up close to it (stops at a safe distance).", {"object": "string"}, self.approach),
            Skill("describe", "Describe what the camera sees right now.", {}, self.describe),
            Skill("photo", "Save a photo from the camera.", {}, self.photo),
        ]:
            self.skills[sk.name] = sk

    # ------------------------------------------------------------ helpers
    def call(self, name: str, args: dict | None = None) -> SkillResult:
        if name not in self.skills:
            return SkillResult(False, f"I don't know how to '{name}'.")
        st = self.s.status()
        if st.get("mode") == "crashed" and name not in ("status", "describe", "photo"):
            return SkillResult(False, "I've crashed, so I can't fly. Reset the simulation or check the drone.")
        try:
            return self.skills[name].fn(**(args or {}))
        except TypeError as exc:
            return SkillResult(False, f"Bad arguments for {name}: {exc}")
        except (ValueError, TimeoutError) as exc:
            return SkillResult(False, str(exc))

    def _airborne(self) -> bool:
        return self.s.status().get("mode") in ("takeoff", "mission", "hold", "rtl")

    def _ensure_airborne(self) -> SkillResult | None:
        if self._airborne():
            return None
        r = self.takeoff()
        return None if r.ok else r

    def _yaw(self) -> float:
        return math.radians(self.s.status().get("yaw_deg", 0.0))

    def _pos(self) -> np.ndarray:
        return np.array(self.s.status().get("position", [0.0, 0.0, 0.0]), dtype=float)

    def _nearest_place(self, xy: np.ndarray) -> tuple[str | None, float]:
        best, dist = None, float("inf")
        for name, p in self.cfg.mission.places.items():
            d = float(np.linalg.norm(np.asarray(p[:2]) - xy[:2]))
            if d < dist:
                best, dist = name, d
        return best, dist

    def resolve_place(self, place: str) -> tuple[str, np.ndarray]:
        places = self.cfg.mission.places
        q = _words(place)
        for name in places:
            if _words(name) == q or name.lower() == place.lower().strip():
                return name, np.asarray(places[name][:2], dtype=float)
        for name in places:
            if _words(name) & q:
                return name, np.asarray(places[name][:2], dtype=float)
        known = ", ".join(places) or "none configured"
        raise ValueError(f"I don't know a place called '{place}'. Known places: {known}.")

    def _inside_fence(self, xy: np.ndarray, margin: float = 0.2) -> bool:
        xmin, ymin, xmax, ymax = self.cfg.mission.geofence
        return xmin + margin <= xy[0] <= xmax - margin and ymin + margin <= xy[1] <= ymax - margin

    def _face(self, yaw: float, timeout: float = 8.0) -> bool:
        if self.s.status().get("mode") != "hold":
            self.s.hold()
        self.s.turn_to(yaw)
        return self.s.wait_for(lambda st: abs(wrap_angle(math.radians(st.get("yaw_deg", 0.0)) - yaw)) < math.radians(6), timeout, self.cancel)

    @staticmethod
    def _matches(query: str, label: str) -> bool:
        """'red bag' matches 'red_marker' (colour is the distinctive word), 'person' matches 'person'."""
        q = _words(query) - {"object", "thing", "something"}
        lw = _words(label.replace("_", " "))
        generic = {"marker", "ball", "bag", "box", "item", "thing"}
        return bool(q) and (q <= lw or bool((q - generic) & lw))

    # ------------------------------------------------------------ skills
    def status(self) -> SkillResult:
        st = self.s.status()
        pos = np.array(st.get("position", [0, 0, 0]))
        place, d = self._nearest_place(pos)
        where = f"near the {place}" if place and d < 1.5 else f"at x {pos[0]:.1f}, y {pos[1]:.1f}"
        mode = st.get("mode", "?")
        doing = {"preflight": "on the ground, ready", "landed": "landed", "hold": "hovering", "mission": "flying",
                 "takeoff": "taking off", "rtl": "flying home", "land": "landing", "crashed": "crashed"}.get(mode, mode)
        first = f"I'm {doing} {where}"
        if mode not in ("preflight", "landed"):
            first += f" at {pos[2]:.1f} metres"
        second = f"Battery {st.get('battery', 0) * 100:.0f}%"
        if st.get("min_range") is not None:
            second += f", nearest obstacle {st['min_range']:.1f} m"
        text = f"{first}. {second}."
        if st.get("failsafe"):
            text += f" Failsafe active: {st['failsafe']}."
        return SkillResult(True, text, st, observation=True)

    def takeoff(self, altitude: float | None = None) -> SkillResult:
        if self._airborne():
            return SkillResult(True, "I'm already flying.")
        alt = self.cfg.mission.cruise_altitude if altitude is None else float(altitude)
        alt = float(np.clip(alt, 0.3, self.cfg.mission.max_altitude))
        if not self.s.takeoff(alt):
            return SkillResult(False, "Takeoff refused (bench mode, or I'm not on the ground).")
        ok = self.s.wait_for(lambda st: st.get("mode") == "hold", 20.0, self.cancel)
        return SkillResult(ok, f"Airborne at {alt:.1f} metres." if ok else "Takeoff didn't complete.")

    def land(self) -> SkillResult:
        if not self._airborne() and self.s.status().get("mode") != "land":
            return SkillResult(True, "I'm already on the ground.")
        self.s.land()
        ok = self.s.wait_for(lambda st: st.get("mode") == "landed", 30.0, self.cancel)
        return SkillResult(ok, "Landed." if ok else "Still landing.")

    def return_home(self) -> SkillResult:
        if not self._airborne():
            return SkillResult(True, "I'm on the ground already.")
        self.s.return_home()
        ok = self.s.wait_for(lambda st: st.get("mode") == "landed", 120.0, self.cancel)
        return SkillResult(ok, "Back home and landed." if ok else "I couldn't get home yet. I'm still trying.")

    def hold(self) -> SkillResult:
        if not self._airborne():
            return SkillResult(True, "I'm on the ground.")
        self.s.hold()
        return SkillResult(True, "Holding position.")

    def go_to(self, place: str) -> SkillResult:
        name, xy = self.resolve_place(place)
        r = self.go_to_xy(float(xy[0]), float(xy[1]))
        r.text = r.text.replace("the target", f"the {name}")
        return r

    def go_to_xy(self, x: float, y: float) -> SkillResult:
        xy = np.array([float(x), float(y)])
        if not self._inside_fence(xy):
            return SkillResult(False, "That's outside my allowed flying area, so I won't go there.")
        bad = self._ensure_airborne()
        if bad:
            return bad
        self.s.go_to(xy)
        dist = float(np.linalg.norm(xy - self._pos()[:2]))
        timeout = 15.0 + 3.0 * dist / max(self.cfg.mission.cruise_speed, 0.1)
        ok = self.s.wait_for(lambda st: st.get("mode") == "hold", timeout, self.cancel)
        if ok:
            return SkillResult(True, "I've arrived at the target.")
        if self.cancel.is_set():
            return SkillResult(False, "Cancelled.")
        self.s.hold()
        return SkillResult(False, "I couldn't reach the target; the way seems blocked. Holding position.")

    def move(self, direction: str, meters: float = 1.0) -> SkillResult:
        offs = {"forward": 0.0, "ahead": 0.0, "back": math.pi, "backward": math.pi, "left": math.pi / 2, "right": -math.pi / 2}
        d = direction.lower().strip()
        if d not in offs:
            return SkillResult(False, f"I can move forward, back, left or right, not '{direction}'.")
        m = float(np.clip(meters, 0.2, 3.0))
        bad = self._ensure_airborne()
        if bad:
            return bad
        heading = self._yaw() + offs[d]
        target = self._pos()[:2] + m * np.array([math.cos(heading), math.sin(heading)])
        r = self.go_to_xy(*target)
        if r.ok:
            r.text = f"Moved {d} {m:.1f} metres."
        return r

    def turn(self, degrees: float) -> SkillResult:
        if not self._airborne():
            return SkillResult(False, "I need to be flying to turn. Say 'take off' first.")
        ok = self._face(float(wrap_angle(self._yaw() + math.radians(float(degrees)))))
        side = "left" if degrees > 0 else "right"
        return SkillResult(ok, f"Turned {abs(degrees):.0f} degrees {side}." if ok else "I couldn't finish turning.")

    def set_altitude(self, meters: float) -> SkillResult:
        if not self._airborne():
            return SkillResult(False, "I need to be flying first.")
        z = float(np.clip(meters, 0.3, self.cfg.mission.max_altitude))
        if self.s.status().get("mode") not in ("hold", "mission"):
            self.s.hold()
        self.s.set_altitude(z)
        ok = self.s.wait_for(lambda st: abs(st["position"][2] - z) < 0.12, 15.0, self.cancel)
        note = "" if z == float(meters) else f" (limited to {z:.1f} m)"
        return SkillResult(ok, f"Now at {z:.1f} metres{note}." if ok else "Altitude change didn't finish.")

    def climb(self, meters: float) -> SkillResult:
        if not self._airborne():
            return SkillResult(False, "I need to be flying first.")
        return self.set_altitude(self._pos()[2] + float(meters))

    def _look_around(self) -> list[dict]:
        """Rotate in 45-degree steps and collect detections with world bearings."""
        seen: list[dict] = []
        yaw0 = self._yaw()
        steps = 8 if self._airborne() else 1
        for k in range(steps):
            if self.cancel.is_set():
                break
            if steps > 1:
                self._face(float(wrap_angle(yaw0 + k * math.pi / 4)), timeout=6.0)
                self.s.wait_for(lambda st: False, 0.6, self.cancel)  # let a few frames arrive
            for d in self.s.status().get("detections", []):
                same = [e for e in seen if e["label"] == d["label"] and abs(wrap_angle(e["bearing"] - d["bearing"])) < math.radians(25)]
                if same:
                    e = same[0]
                    if d["confidence"] > e["confidence"]:
                        e.update(d)
                else:
                    seen.append(dict(d))
        return seen

    def _describe_list(self, seen: list[dict]) -> str:
        if not seen:
            return "I don't recognise anything around me."
        yaw = self._yaw()
        items = [f"a {d['label'].replace('_', ' ')} {relative_direction(wrap_angle(d['bearing'] - yaw))}" for d in seen]
        return "I can see " + ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] + "."

    def scan(self) -> SkillResult:
        seen = self._look_around()
        return SkillResult(True, self._describe_list(seen), {"seen": seen}, observation=True)

    def _front_range(self) -> float | None:
        r = self.s.ranges
        if not len(r):
            return None
        sens = self.cfg.sensors
        if sens.range_sensor == "tof_ring" and 0.0 in [float(h) for h in sens.tof_headings_deg]:
            i = [float(h) for h in sens.tof_headings_deg].index(0.0)
            front = float(r[i * sens.tof_subrays + sens.tof_subrays // 2])
        else:
            front = float(r[len(r) // 2])  # lidar beams run -pi..pi: the middle one points forward
        return front if front < sens.lidar_max_range * 0.995 else None

    def find(self, object: str) -> SkillResult:  # noqa: A002 - matches the tool schema
        matches = [d for d in self.s.status().get("detections", []) if self._matches(object, d["label"])]
        if not matches:
            matches = [d for d in self._look_around() if self._matches(object, d["label"])]
        if not matches:
            return SkillResult(False, f"I looked around but couldn't find a {object}.", observation=True)
        best = max(matches, key=lambda d: d["confidence"])
        label = best["label"].replace("_", " ")
        if self._airborne():
            self._face(best["bearing"])
            rng = self._front_range()
            dist = f" It's about {rng:.1f} metres away." if rng is not None else ""
            return SkillResult(True, f"Found a {label}. I'm facing it now.{dist}", {"target": best, "range": rng}, observation=True)
        rel = relative_direction(wrap_angle(best["bearing"] - self._yaw()))
        return SkillResult(True, f"I can see a {label} {rel}.", {"target": best}, observation=True)

    def approach(self, object: str) -> SkillResult:  # noqa: A002
        bad = self._ensure_airborne()
        if bad:
            return bad
        found = self.find(object)
        if not found.ok:
            return found
        rng = found.data.get("range")
        stop = self.cfg.avoidance.stop_distance + 0.3
        go = float(np.clip((rng if rng is not None else 1.5) - stop, 0.0, 3.0))
        if go < 0.15:
            return SkillResult(True, f"I'm already close to the {object}.", observation=True)
        r = self.move("forward", go)
        if r.ok:
            r.text = f"I'm right in front of the {object} now."
            r.observation = True
        return r

    def describe(self) -> SkillResult:
        """Without a vision model: summarise detector output and nearby obstacles."""
        st = self.s.status()
        text = self._describe_list(st.get("detections", []))
        rng = self._front_range()
        if rng is not None:
            text += f" There's something {rng:.1f} metres in front of me."
        return SkillResult(True, text, observation=True)

    def photo(self) -> SkillResult:
        name = self.s.snapshot_photo()
        return SkillResult(bool(name), f"Photo saved as {name}." if name else "No camera image yet.")
