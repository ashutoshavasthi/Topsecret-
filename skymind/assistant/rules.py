"""Built-in command understanding: works with no AI model installed.

It covers the everyday commands deterministically, so the drone always responds
to "take off", "stop" or "come back", even if the local language model is missing,
slow or confused. The language model handles everything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NUMBER_WORDS = {"half a": 0.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


@dataclass
class Plan:
    reply: str
    actions: list[dict] = field(default_factory=list)  # [{"tool": name, "args": {...}}]
    source: str = "rules"


STOP_WORDS = re.compile(r"\b(stop|halt|freeze|hover|wait|hold( on| position)?|pause|abort)\b")


def is_stop(text: str) -> bool:
    return bool(STOP_WORDS.search(text.lower()))


def _num(text: str, default: float) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    for w, v in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            return float(v)
    return default


def parse(text: str, places: list[str]) -> Plan | None:
    """Return a plan for simple commands, or None if the request needs real understanding."""
    t = " ".join(text.lower().replace("’", "'").split())
    t = re.sub(r"^(hey |ok |okay )?(skymind|jarvis|drone)[,!. ]*", "", t).strip()
    if not t:
        return Plan("Yes? I'm listening.")

    if is_stop(t) and len(t.split()) <= 4:
        return Plan("Stopping. Holding position.", [{"tool": "hold"}])
    if re.search(r"\b(take ?off|lift ?off|launch|fly up|get up|start flying)\b", t):
        return Plan("Taking off.", [{"tool": "takeoff"}])
    if re.search(r"\b(come back|return( home)?|go home|fly home|come home|rtl)\b", t):
        return Plan("Coming back home.", [{"tool": "return_home"}])
    if re.search(r"\b(land|touch ?down|come down|get down)\b", t):
        return Plan("Landing now.", [{"tool": "land"}])
    if re.search(r"\b(status|battery|how are you|where are you|report)\b", t):
        return Plan("", [{"tool": "status"}])
    if re.search(r"\b(photo|picture|snapshot|capture)\b", t):
        return Plan("Taking a photo.", [{"tool": "photo"}])
    if re.search(r"\b(scan|map|explore|survey)\s+(the\s+|this\s+|my\s+)?(room|area|place|house|space|hall|lab)\b"
                 r"|\bexplore\b|\bmake a map\b|\broom scan\b", t):
        return Plan("Starting a room scan. I'll explore and map everything, it takes a few minutes.", [{"tool": "survey"}])
    if re.search(r"\b(exits?|doorways?|doors in|what did you find|room report|what's in the room|blockages?)\b", t):
        return Plan("", [{"tool": "room_report"}])
    if re.search(r"\b(look around|scan|spin around|360)\b", t):
        return Plan("Looking around.", [{"tool": "scan"}])
    if re.search(r"\b(what (do|can) you see|describe|what's (there|in front|ahead)|what is in front)\b", t):
        return Plan("", [{"tool": "describe"}])

    m = re.search(r"\b(go to|fly to|move to|head to|take me to)\s+(the\s+)?(.+?)\s*$", t)
    if m:
        dest = m.group(3).strip(" .!?")
        for p in places:
            if p.lower() in dest or dest in p.lower():
                return Plan(f"Heading to the {p}.", [{"tool": "go_to", "args": {"place": p}}])

    m = re.search(r"\b(go|fly|move)\s+(forward|ahead|back|backward|left|right)\b(.*)", t)
    if m:
        meters = _num(m.group(3), 1.0)
        return Plan(f"Moving {m.group(2)}.", [{"tool": "move", "args": {"direction": m.group(2), "meters": meters}}])

    m = re.search(r"\b(turn|rotate|look)\s+(left|right|around)\b(.*)", t)
    if m:
        if m.group(2) == "around":
            return Plan("Turning around.", [{"tool": "turn", "args": {"degrees": 180}}])
        deg = _num(m.group(3), 90.0)
        return Plan(f"Turning {m.group(2)}.", [{"tool": "turn", "args": {"degrees": deg if m.group(2) == "left" else -deg}}])

    m = re.search(r"\b(find|look for|search for|where is|where's|locate)\s+(the\s+|a\s+|my\s+)?(.+?)\s*\??$", t)
    if m:
        obj = m.group(3).strip(" .!?")
        return Plan(f"Looking for the {obj}.", [{"tool": "find", "args": {"object": obj}}])
    m = re.search(r"\b(approach|go near|fly near|go up to|get close to)\s+(the\s+|a\s+)?(.+?)\s*$", t)
    if m:
        obj = m.group(3).strip(" .!?")
        return Plan(f"Approaching the {obj}.", [{"tool": "approach", "args": {"object": obj}}])

    m = re.search(r"\b(up|higher|climb|rise|down|lower|descend)\b(.*)", t)
    if m:
        up = m.group(1) in ("up", "higher", "climb", "rise")
        meters = _num(m.group(2), 0.5)
        return Plan("Going up." if up else "Going down.", [{"tool": "climb", "args": {"meters": meters if up else -meters}}])
    return None
