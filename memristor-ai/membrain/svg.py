"""Tiny dependency-free SVG line charts for reports."""
from __future__ import annotations

import math
from html import escape

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.0e}"
    return f"{v:g}"


def line_chart(series: dict[str, tuple[list[float], list[float]]], path: str, title: str,
               x_label: str, y_label: str, log_x: bool = False, y_range: tuple[float, float] | None = None,
               x_tick_labels: dict[float, str] | None = None, width: int = 720, height: int = 420) -> str:
    """Write a multi-series line chart. ``series`` maps label -> (xs, ys)."""
    rows = math.ceil(len(series) / 3)
    ml, mr, mt, mb = 70, 36, 44, 62 + 18 * rows
    pw, ph = width - ml - mr, height - mt - mb
    tx = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    xs_all = [tx(x) for xs, _ in series.values() for x in xs]
    ys_all = [y for _, ys in series.values() for y in ys if y == y]
    x0, x1 = min(xs_all), max(xs_all)
    if x1 == x0:
        x1 = x0 + 1
    y0, y1 = y_range or (min(ys_all), max(ys_all))
    if y_range is None:
        pad = 0.05 * (y1 - y0 or 1)
        y0, y1 = y0 - pad, y1 + pad

    def px(x: float) -> float:
        return ml + (tx(x) - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        return mt + (1 - (y - y0) / (y1 - y0)) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{ml}" y="24" font-size="15" font-weight="bold" fill="#111827">{escape(title)}</text>']
    for t in _nice_ticks(y0, y1):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{ml - 8}" y="{py(t) + 4:.1f}" text-anchor="end" fill="#4b5563">{_fmt(t)}</text>')
    if x_tick_labels:
        xticks = list(x_tick_labels.items())
    elif log_x:
        xticks = [(10 ** e, _fmt(10 ** e)) for e in range(math.floor(x0), math.ceil(x1) + 1)
                  if x0 - 1e-9 <= e <= x1 + 1e-9]
    else:
        xticks = [(t, _fmt(t)) for t in _nice_ticks(x0, x1)]
    for t, lab in xticks:
        out.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{mt}" y2="{mt + ph}" stroke="#f3f4f6"/>')
        out.append(f'<text x="{px(t):.1f}" y="{mt + ph + 18}" text-anchor="middle" fill="#4b5563">{escape(lab)}</text>')
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#9ca3af"/>')
    out.append(f'<text x="{ml + pw / 2}" y="{mt + ph + 38}" text-anchor="middle" fill="#111827">{escape(x_label)}</text>')
    out.append(f'<text transform="translate(18,{mt + ph / 2}) rotate(-90)" text-anchor="middle" fill="#111827">'
               f'{escape(y_label)}</text>')
    for k, (label, (xs, ys)) in enumerate(series.items()):
        color = PALETTE[k % len(PALETTE)]
        pts = " ".join(f"{px(x):.1f},{py(min(max(y, y0), y1)):.1f}" for x, y in zip(xs, ys) if y == y)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        if len(xs) <= 40:
            for x, y in zip(xs, ys):
                if y == y:
                    out.append(f'<circle cx="{px(x):.1f}" cy="{py(min(max(y, y0), y1)):.1f}" r="3.2" fill="{color}"/>')
        lx = ml + (k % 3) * (pw / 3)
        ly = mt + ph + 66 + (k // 3) * 18
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly - 4}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{lx + 28}" y="{ly}" fill="#111827">{escape(label)}</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg
