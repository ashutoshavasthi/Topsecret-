"""Design calculator: compare builds on weight, power, endurance and noise
before buying parts.

    skymind design config/micro.yaml config/default.yaml
    skymind design config/micro.yaml --sweep-battery 5,7.4,9.6,13,17
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from skymind.config import SkyMindConfig
from skymind.power import PowerModel, relative_noise_db

SUB_250_LIMIT = 0.250  # kg, the registration-free class in many countries (check your local rules)


@dataclass
class DesignReport:
    name: str
    mass_g: float
    battery_wh: float
    battery_mass_g: float
    thrust_to_weight: float
    hover_power_w: float
    hover_current_a: float
    hover_endurance_min: float
    hover_tip_speed: float
    noise_vs_reference_db: float
    sub_250g: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(cfg: SkyMindConfig, name: str = "", reference_tip_speed: float | None = None) -> DesignReport:
    pm = PowerModel(cfg.vehicle)
    b = cfg.battery
    hover = pm.hover_power()
    tip = pm.hover_tip_speed()
    ref = reference_tip_speed or tip
    twr = pm.thrust_to_weight()
    warnings = []
    if twr < 2.0:
        warnings.append("thrust/weight < 2: sluggish, poor wind and disturbance rejection")
    if twr > 8.0:
        warnings.append("thrust/weight > 8: motors oversized, wasting weight")
    if tip > 120:
        warnings.append("tip speed > 120 m/s: loud and inefficient, use bigger/slower props")
    return DesignReport(
        name=name,
        mass_g=round(cfg.vehicle.mass * 1000, 1),
        battery_wh=b.capacity_wh,
        battery_mass_g=round(b.capacity_wh / b.specific_energy, 1),
        thrust_to_weight=round(twr, 2),
        hover_power_w=round(hover, 1),
        hover_current_a=round(hover / (3.7 * b.cells), 2),
        hover_endurance_min=round(pm.hover_endurance_min(b), 1),
        hover_tip_speed=round(tip, 1),
        noise_vs_reference_db=round(relative_noise_db(tip, ref), 1),
        sub_250g=cfg.vehicle.mass <= SUB_250_LIMIT,
        warnings=warnings,
    )


def with_battery(cfg: SkyMindConfig, capacity_wh: float) -> SkyMindConfig:
    """Same airframe with a different pack: all-up mass changes with the pack's weight."""
    new = copy.deepcopy(cfg)
    b = new.battery
    new.vehicle.mass = cfg.vehicle.mass + (capacity_wh - b.capacity_wh) / b.specific_energy / 1000.0
    b.capacity_wh = capacity_wh
    return new


def format_table(reports: list[DesignReport]) -> str:
    cols = [
        ("build", "name", "{}"),
        ("mass g", "mass_g", "{:.0f}"),
        ("pack Wh", "battery_wh", "{:.1f}"),
        ("T/W", "thrust_to_weight", "{:.1f}"),
        ("hover W", "hover_power_w", "{:.0f}"),
        ("hover A", "hover_current_a", "{:.1f}"),
        ("endur. min", "hover_endurance_min", "{:.1f}"),
        ("tip m/s", "hover_tip_speed", "{:.0f}"),
        ("noise dB", "noise_vs_reference_db", "{:+.1f}"),
        ("<250 g", "sub_250g", "{}"),
    ]
    rows = [[h for h, _, _ in cols]]
    for r in reports:
        d = r.to_dict()
        rows.append([fmt.format(d[k]) if not isinstance(d[k], bool) else ("yes" if d[k] else "no") for _, k, fmt in cols])
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    for r in reports:
        for w in r.warnings:
            lines.append(f"! {r.name}: {w}")
    return "\n".join(lines)
