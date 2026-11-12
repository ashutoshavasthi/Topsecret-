"""Procedural hand-drawn digits, so the project needs no dataset download.

Each digit is a set of strokes (polylines in a unit square). Every sample gets its own
random rotation, anisotropic scale, shear, shift, stroke width, per-vertex wobble, pixel
noise and (sometimes) a stray pen stroke, then is rasterised with anti-aliasing.
16 x 16 pixels, values in [0, 1].
"""
from __future__ import annotations

import numpy as np


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int = 12) -> list[tuple[float, float]]:
    # Image coordinates: y points down, so increasing angle runs clockwise on screen.
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


GLYPHS: dict[int, list[list[tuple[float, float]]]] = {
    0: [_arc(0.5, 0.5, 0.27, 0.40, 0, 360, 20)],
    1: [[(0.33, 0.27), (0.52, 0.10), (0.52, 0.90)]],
    2: [_arc(0.5, 0.33, 0.24, 0.22, -165, 30, 10) + [(0.24, 0.90), (0.78, 0.90)]],
    3: [_arc(0.48, 0.30, 0.22, 0.19, -150, 90, 10), _arc(0.48, 0.70, 0.25, 0.21, -90, 150, 10)],
    4: [[(0.64, 0.90), (0.64, 0.10), (0.20, 0.64), (0.82, 0.64)]],
    5: [[(0.75, 0.10), (0.32, 0.10), (0.29, 0.45)] + _arc(0.5, 0.66, 0.25, 0.23, -125, 150, 12)],
    6: [[(0.70, 0.10), (0.46, 0.20), (0.32, 0.42)] + _arc(0.5, 0.68, 0.20, 0.21, 180, 540, 16)],
    7: [[(0.20, 0.10), (0.80, 0.10), (0.42, 0.90)]],
    8: [_arc(0.5, 0.29, 0.19, 0.18, 0, 360, 16), _arc(0.5, 0.70, 0.23, 0.21, 0, 360, 16)],
    9: [_arc(0.5, 0.32, 0.21, 0.21, 0, 360, 16), [(0.71, 0.32), (0.66, 0.60), (0.42, 0.90)]],
}


def _segments(strokes: list[list[tuple[float, float]]], rng: np.random.Generator, wobble: float) -> np.ndarray:
    segs = []
    for stroke in strokes:
        p = np.asarray(stroke, dtype=float)
        p = p + rng.normal(0.0, wobble, p.shape)
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
    return np.concatenate(segs, axis=0)  # (K, 2, 2)


def render_digit(label: int, rng: np.random.Generator, size: int = 16, noise: float = 0.08,
                 clutter: float = 0.3) -> np.ndarray:
    """Render one randomly distorted digit as a (size*size,) float array.

    ``clutter`` is the probability of a stray pen stroke somewhere in the image.
    """
    segs = _segments(GLYPHS[label], rng, wobble=0.04)
    if rng.random() < clutter:
        start = rng.uniform(0.05, 0.95, 2)
        end = start + rng.normal(0, 0.25, 2)
        segs = np.concatenate([segs, np.stack([start, end])[None]], axis=0)

    # Random affine transform about the image centre.
    th = np.radians(rng.uniform(-18, 18))
    sx, sy = rng.uniform(0.78, 1.05), rng.uniform(0.82, 1.05)
    sh = rng.uniform(-0.25, 0.25)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    m = rot @ np.array([[sx, sh], [0.0, sy]])
    shift = rng.uniform(-0.1, 0.1, 2)
    segs = (segs - 0.5) @ m.T + 0.5 + shift

    a, b = segs[:, 0], segs[:, 1]  # (K, 2)
    c = (np.arange(size) + 0.5) / size
    px = np.stack(np.meshgrid(c, c), axis=-1).reshape(-1, 2)  # (P, 2) as (x, y)
    ab = b - a
    t = np.einsum("pkd,kd->pk", px[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    d = np.sqrt(((px[:, None, :] - nearest) ** 2).sum(-1)).min(axis=1)

    width = rng.uniform(0.045, 0.10)
    img = np.clip((width - d) / (1.5 / size) + 0.5, 0.0, 1.0)
    img *= rng.uniform(0.75, 1.0)  # pen pressure
    img += rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0.0, 1.0)


def make_digits(n: int, seed: int = 0, size: int = 16, noise: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """n class-balanced samples: X (n, size*size) in [0, 1], y (n,) int labels 0-9."""
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 10)
    X = np.stack([render_digit(int(k), rng, size, noise) for k in y])
    return X, y.astype(np.int64)


def train_test(n_train: int = 6000, n_test: int = 2000, seed: int = 0) -> tuple[np.ndarray, ...]:
    Xtr, ytr = make_digits(n_train, seed=seed)
    Xte, yte = make_digits(n_test, seed=seed + 10_000)
    return Xtr, ytr, Xte, yte


def ascii_digit(x: np.ndarray, size: int = 16) -> str:
    ramp = " .:-=+*#%@"
    img = x.reshape(size, size)
    return "\n".join("".join(ramp[min(int(v * 9.999), 9)] * 2 for v in row) for row in img)
