"""Automatic flight judge.

Turns a flight log (and, optionally, the mission summary) into a scorecard:

    stability   how well the drone balanced itself and tracked commands
    navigation  did it get where it was told to, efficiently
    safety      collisions, closest approach to obstacles, failsafes
    vision      footage quality and how reliably the AI found the targets
    efficiency  energy per km, endurance, and rotor tip speed (noise / stealth)

Each metric is mapped linearly from a "good" value (100 points) to a "bad"
value (0 points). Category scores are the mean of their metrics and the
overall score is the weighted mean of categories. A collision caps the
overall score at 50 regardless of everything else.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from skymind.config import JudgeConfig
from skymind.math3d import wrap_angle
from skymind.vision.analyzer import iou

FLYING_MODES = {"takeoff", "mission", "rtl", "land"}
CRUISE_MODES = {"mission", "rtl"}


def linear_score(value: float, good: float, bad: float) -> float:
    if good == bad:
        return 100.0 if value == good else 0.0
    frac = (value - bad) / (good - bad)
    return float(100.0 * min(1.0, max(0.0, frac)))


@dataclass
class Metric:
    name: str
    value: float
    score: float
    unit: str = ""
    note: str = ""


@dataclass
class Category:
    name: str
    score: float
    weight: float
    metrics: list[Metric] = field(default_factory=list)


@dataclass
class Scorecard:
    overall: float
    grade: str
    verdict: str
    categories: list[Category]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# SkyMind Flight Scorecard",
            "",
            f"**Overall: {self.overall:.1f} / 100 — Grade {self.grade}**  ",
            f"Verdict: {self.verdict}",
            "",
            "| Category | Weight | Score |",
            "|---|---:|---:|",
        ]
        for c in self.categories:
            lines.append(f"| {c.name} | {c.weight:.2f} | {c.score:.1f} |")
        for c in self.categories:
            lines += ["", f"## {c.name.title()} — {c.score:.1f}", "", "| Metric | Value | Score | Note |", "|---|---:|---:|---|"]
            for m in c.metrics:
                lines.append(f"| {m.name} | {m.value:.3f} {m.unit} | {m.score:.1f} | {m.note} |")
        if self.summary:
            lines += ["", "## Flight summary", ""]
            for k, v in self.summary.items():
                if k == "events":
                    continue
                lines.append(f"- **{k}**: {v}")
            events = self.summary.get("events") or []
            if events:
                lines += ["", "## Event log", ""]
                lines += [f"- `{t:7.2f}s` {e}" for t, e in events]
        return "\n".join(lines) + "\n"


def _grade(score: float) -> str:
    for g, lo in (("A", 90), ("B", 80), ("C", 70), ("D", 60)):
        if score >= lo:
            return g
    return "F"


class FlightJudge:
    def __init__(self, cfg: JudgeConfig | None = None):
        self.cfg = cfg or JudgeConfig()

    def _metric(self, name: str, value: float, unit: str = "", note: str = "") -> Metric:
        good, bad = self.cfg.thresholds[name]
        return Metric(name, float(value), linear_score(value, good, bad), unit, note)

    # ------------------------------------------------------------- categories
    def stability(self, states: list[dict]) -> Category:
        airborne = [s for s in states if s["mode"] in FLYING_MODES and s["pos"][2] > 0.5]
        cruise = [s for s in airborne if s["mode"] in CRUISE_MODES]
        metrics: list[Metric] = []
        if airborne:
            # Roll/pitch only: yaw turns are deliberate (camera pointing), not wobble.
            rates = np.array([s["rate"][:2] for s in airborne])
            metrics.append(self._metric("rms_body_rate", math.sqrt(np.mean(np.sum(rates**2, axis=1))), "rad/s", "roll+pitch"))
            tilt = [math.degrees(math.acos(min(1.0, math.cos(s["rpy"][0]) * math.cos(s["rpy"][1])))) for s in airborne]
            metrics.append(self._metric("max_tilt_deg", max(tilt), "deg"))
            if all("est_rpy" in s for s in airborne):
                err = np.array([wrap_angle(np.array(s["est_rpy"][:2]) - np.array(s["rpy"][:2])) for s in airborne])
                rms = math.degrees(math.sqrt(np.mean(np.sum(err**2, axis=1))))
                metrics.append(self._metric("rms_attitude_estimate_error_deg", rms, "deg", "AHRS vs truth (sim only)"))
        if cruise:
            alt_err = [s["pos"][2] - s["alt_sp"] for s in cruise if s.get("alt_sp") is not None]
            if alt_err:
                metrics.append(self._metric("rms_altitude_error", math.sqrt(np.mean(np.square(alt_err))), "m"))
            vel_err = [np.linalg.norm(np.array(s["vel"][:2]) - np.array(s["vel_sp"][:2])) for s in cruise if s.get("vel_sp")]
            if vel_err:
                metrics.append(self._metric("rms_velocity_tracking_error", math.sqrt(np.mean(np.square(vel_err))), "m/s"))
        return self._category("stability", metrics)

    def navigation(self, states: list[dict], mission: dict | None) -> Category:
        metrics: list[Metric] = []
        if mission:
            total = max(mission["waypoints_total"], 1)
            metrics.append(
                self._metric("waypoint_completion", mission["waypoints_reached"] / total, "", f"{mission['waypoints_reached']}/{total}")
            )
            if mission["distance_flown"] > 0:
                eff = min(1.0, mission["straight_line_distance"] / mission["distance_flown"])
                metrics.append(self._metric("path_efficiency", eff, "", "straight-line / flown"))
        elif states:
            reached = max(s.get("wp", 0) for s in states)
            metrics.append(Metric("waypoints_reached", reached, 100.0 if reached else 0.0))
        return self._category("navigation", metrics)

    def safety(self, states: list[dict], mission: dict | None) -> tuple[Category, int]:
        metrics: list[Metric] = []
        collisions = int(mission["collisions"]) if mission else 0
        metrics.append(Metric("collisions", collisions, 100.0 if collisions == 0 else 0.0))
        clearances = [s["clearance"] for s in states if s["mode"] in FLYING_MODES and s["pos"][2] > 0.3 and s.get("clearance") is not None]
        clearances = [c for c in clearances if c < 1e8]
        if clearances:
            metrics.append(self._metric("min_clearance", min(clearances), "m"))
        else:
            min_range = [s["min_range"] for s in states if s["mode"] in FLYING_MODES and s.get("min_range") is not None]
            if min_range:
                metrics.append(self._metric("min_clearance", min(min_range), "m", "from lidar"))
        if mission:
            fs = mission.get("failsafe")
            # Low-battery RTL is the system working as designed; geofence or timeout is a planning failure.
            penalty = {"geofence": 0.0, "timeout": 40.0}.get(fs or "", 100.0)
            metrics.append(Metric("failsafe", 0.0 if fs else 1.0, penalty, note=fs or "none triggered"))
            landed = mission["status"] == "landed"
            metrics.append(Metric("safe_landing", 1.0 if landed else 0.0, 100.0 if landed else 0.0, note=mission["status"]))
        return self._category("safety", metrics), collisions

    def vision(self, frames: list[dict]) -> Category:
        metrics: list[Metric] = []
        if not frames:
            return self._category("vision", metrics)  # no camera: category is left out of the total
        q = [f["quality"] for f in frames if f.get("quality")]
        if q:
            metrics.append(self._metric("sharpness", float(np.median([x["sharpness"] for x in q])), "", "median var(Laplacian)"))
            metrics.append(self._metric("exposure_error", float(np.mean([abs(x["brightness"] - 128) for x in q])), "luma"))
            clipped = float(np.mean([x["overexposed"] + x["underexposed"] for x in q]))
            metrics.append(self._metric("clipped_fraction", clipped))
        with_gt = [f for f in frames if "ground_truth" in f]
        if with_gt:
            tp = fn = fp = 0
            for f in with_gt:
                gts = [g for g in f["ground_truth"] if g["bbox"][2] * g["bbox"][3] >= self.cfg.min_gt_area]
                dets = list(f["detections"])
                matched = set()
                for g in gts:
                    best, best_i = 0.0, -1
                    for i, d in enumerate(dets):
                        if i in matched or d["label"] != g["label"]:
                            continue
                        v = iou(tuple(g["bbox"]), tuple(d["bbox"]))
                        if v > best:
                            best, best_i = v, i
                    if best >= 0.3:
                        tp += 1
                        matched.add(best_i)
                    else:
                        fn += 1
                # Only count unmatched detections that do not overlap any (even tiny) ground truth.
                for i, d in enumerate(dets):
                    if i in matched:
                        continue
                    if not any(g["label"] == d["label"] and iou(tuple(g["bbox"]), tuple(d["bbox"])) > 0.05 for g in f["ground_truth"]):
                        fp += 1
            if tp + fn:
                metrics.append(self._metric("detection_recall", tp / (tp + fn), "", f"{tp} hits / {tp + fn} visible targets"))
            if tp + fp:
                metrics.append(self._metric("detection_precision", tp / (tp + fp), "", f"{fp} false alarms"))
        detected = sum(len(f["detections"]) for f in frames)
        metrics.append(Metric("detections_logged", detected, 100.0 if detected else 0.0, note="total boxes over the flight"))
        return self._category("vision", metrics)

    def efficiency(self, states: list[dict], mission: dict | None) -> Category:
        metrics: list[Metric] = []
        if mission and mission.get("energy_per_km_wh", -1) > 0:
            metrics.append(self._metric("energy_per_km_wh", mission["energy_per_km_wh"], "Wh/km", f"{mission['energy_wh']:.2f} Wh used"))
        if mission and mission.get("est_endurance_min"):
            metrics.append(
                self._metric("est_endurance_min", mission["est_endurance_min"], "min", f"avg {mission['avg_power_w']:.0f} W")
            )
        tips = [s["tip_speed"] for s in states if s.get("tip_speed") and s["pos"][2] > 0.3]
        if tips:
            metrics.append(self._metric("mean_tip_speed", float(np.mean(tips)), "m/s", "rotor noise proxy: lower = quieter"))
        return self._category("efficiency", metrics)

    def _category(self, name: str, metrics: list[Metric]) -> Category:
        score = float(np.mean([m.score for m in metrics])) if metrics else 0.0
        return Category(name, round(score, 2), float(self.cfg.weights.get(name, 0.0)), metrics)

    # ------------------------------------------------------------------ main
    def judge(self, records: list[dict], mission: dict | None = None) -> Scorecard:
        states = [r for r in records if r.get("type") == "state"]
        frames = [r for r in records if r.get("type") == "vision"]
        safety, collisions = self.safety(states, mission)
        cats = [self.stability(states), self.navigation(states, mission), safety, self.vision(frames), self.efficiency(states, mission)]
        weights = sum(c.weight for c in cats if c.metrics) or 1.0
        overall = sum(c.score * c.weight for c in cats if c.metrics) / weights
        verdict = "PASS"
        if collisions:
            overall = min(overall, 50.0)
            verdict = "FAIL — collision"
        elif mission and mission["status"] != "landed":
            verdict = f"FAIL — ended in '{mission['status']}'"
        elif mission and mission["waypoints_reached"] < mission["waypoints_total"]:
            verdict = "PARTIAL — not all waypoints reached"
        overall = round(overall, 2)
        return Scorecard(overall, _grade(overall), verdict, cats, mission or {})


def write_report(card: Scorecard, run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    jp, mp = run_dir / "report.json", run_dir / "report.md"
    jp.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    mp.write_text(card.to_markdown(), encoding="utf-8")
    return jp, mp
