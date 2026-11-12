"""Memristor device physics.

A memristor is a resistor whose conductance G can be nudged up (SET / potentiation) or down
(RESET / depression) by voltage pulses and then *keeps* its value. That makes it an analog,
non-volatile weight. Real devices are imperfect, and every imperfection is modelled here:

* **finite states**: about ``levels`` pulses take a device from g_min to g_max;
* **nonlinear, asymmetric response**: a pulse moves G a lot near one end of the range and very
  little near the other, and potentiation and depression follow different curves;
* **cycle-to-cycle noise**: every pulse lands a little differently;
* **device-to-device variation**: no two devices share exactly the same g_min, g_max or curve;
* **read noise**: every read of G is noisy;
* **stuck devices**: some fraction is stuck fully OFF (g_min) or fully ON (g_max) and ignores pulses;
* **drift**: conductance decays after programming as G(t) = G0 * ((t + t0) / t0) ** -nu (phase-change memory).

Pulse response (normalised state g = (G - g_min) / (g_max - g_min) in [0, 1], ``nu`` = nonlinearity):

    potentiation by n pulses:  g' = (1 - exp(-nu n / L)) / (1 - exp(-nu)) + g exp(-nu n / L)
    depression   by n pulses:  same formula applied to h = 1 - g with the depression nonlinearity

This is the closed form of stepping along G(p) = (1 - exp(-nu p)) / (1 - exp(-nu)), the saturating
exponential widely used to fit measured synaptic devices (it reduces to linear as nu -> 0, and
exactly L pulses span the full range).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DeviceSpec:
    name: str = "ideal"
    g_min: float = 1.0          # microsiemens
    g_max: float = 100.0        # microsiemens
    levels: int = 4096          # pulses to traverse g_min -> g_max
    nl_pot: float = 0.0         # potentiation nonlinearity (0 = linear)
    nl_dep: float = 0.0         # depression nonlinearity
    c2c: float = 0.0            # cycle-to-cycle noise, std per pulse in units of the nominal step
    d2d: float = 0.0            # device-to-device relative spread of g_min, g_max and nonlinearity
    read_noise: float = 0.0     # relative std of a single read
    stuck_off: float = 0.0      # fraction stuck at g_min
    stuck_on: float = 0.0       # fraction stuck at g_max
    drift_nu: float = 0.0       # mean drift exponent
    drift_nu_std: float = 0.0
    drift_t0: float = 1.0       # seconds after programming that drift is measured from
    write_energy_pj: float = 1.0  # energy of one programming pulse
    description: str = ""

    @property
    def g_range(self) -> float:
        return self.g_max - self.g_min

    @property
    def step(self) -> float:
        """Nominal conductance change of one pulse (microsiemens)."""
        return self.g_range / self.levels

    def with_(self, **kw) -> "DeviceSpec":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


# Representative parameter sets. They follow the qualitative behaviour reported for each
# technology in the literature; they are not calibrated to any particular chip.
PRESETS: dict[str, DeviceSpec] = {
    "ideal": DeviceSpec(
        name="ideal", description="Perfect analog weight: linear, noiseless, 4096 levels, no faults."),
    "rram": DeviceSpec(
        name="rram", g_min=2.0, g_max=40.0, levels=64, nl_pot=3.0, nl_dep=4.0, c2c=0.3, d2d=0.05,
        read_noise=0.01, stuck_off=0.005, stuck_on=0.005, write_energy_pj=1.0,
        description="Filamentary oxide RRAM (HfOx-like): ~64 levels, strongly nonlinear, some stuck cells."),
    "pcm": DeviceSpec(
        name="pcm", g_min=0.1, g_max=25.0, levels=100, nl_pot=2.0, nl_dep=12.0, c2c=0.5, d2d=0.10,
        read_noise=0.02, stuck_off=0.005, stuck_on=0.002, drift_nu=0.05, drift_nu_std=0.01,
        write_energy_pj=10.0,
        description="Phase-change memory: gradual SET, abrupt RESET, read noise and conductance drift."),
    "ecram": DeviceSpec(
        name="ecram", g_min=1.0, g_max=10.0, levels=1000, nl_pot=0.2, nl_dep=0.2, c2c=0.05, d2d=0.02,
        read_noise=0.005, write_energy_pj=0.1,
        description="Electrochemical RAM: ~1000 near-linear, symmetric steps; built for on-chip training."),
}


def get_device(spec: str | DeviceSpec, **overrides) -> DeviceSpec:
    base = PRESETS[spec] if isinstance(spec, str) else spec
    return base.with_(**overrides) if overrides else base


def _pot_step(g: np.ndarray, dp: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Move normalised state g up by dp = n / L pulses on the saturating-exponential curve."""
    nu = np.maximum(nu, 1e-9)
    return np.expm1(-nu * dp) / np.expm1(-nu) + g * np.exp(-nu * dp)


class DeviceArray:
    """A 2-D array of memristors with per-device variation, faults and pulse programming."""

    def __init__(self, spec: DeviceSpec, shape: tuple[int, ...], rng: np.random.Generator):
        self.spec = spec
        self.shape = tuple(shape)
        self.rng = rng
        s = spec

        def spread(value: float) -> np.ndarray:
            if s.d2d > 0:
                return value * (1.0 + s.d2d * rng.standard_normal(self.shape))
            return np.full(self.shape, float(value))

        self.g_min = np.maximum(spread(s.g_min), 0.0)
        self.g_max = np.maximum(spread(s.g_max), self.g_min + 0.2 * s.g_range)
        self.nl_pot = np.maximum(spread(s.nl_pot), 0.0)
        self.nl_dep = np.maximum(spread(s.nl_dep), 0.0)

        u = rng.random(self.shape)
        self.stuck = np.zeros(self.shape, np.int8)          # -1 stuck OFF, +1 stuck ON, 0 healthy
        self.stuck[u < s.stuck_off] = -1
        self.stuck[(u >= s.stuck_off) & (u < s.stuck_off + s.stuck_on)] = 1
        self.healthy = self.stuck == 0

        self.nu = np.maximum(s.drift_nu + s.drift_nu_std * rng.standard_normal(self.shape), 0.0) \
            if s.drift_nu > 0 else None
        self.g = self.g_min.copy()
        self._apply_stuck()
        self.age = 0.0               # seconds since last programming (drift)
        self.pulses = 0              # every programming pulse ever sent to this array
        self.resets = 0

    # ----------------------------------------------------------------- state
    def _apply_stuck(self) -> None:
        self.g = np.where(self.stuck == -1, self.g_min, self.g)
        self.g = np.where(self.stuck == 1, self.g_max, self.g)

    def normalized(self) -> np.ndarray:
        return np.clip((self.g - self.g_min) / (self.g_max - self.g_min), 0.0, 1.0)

    def conductance(self) -> np.ndarray:
        """Noise-free conductance right now (including drift since programming)."""
        if self.nu is not None and self.age > 0:
            t0 = self.spec.drift_t0
            return self.g * ((self.age + t0) / t0) ** (-self.nu)
        return self.g

    def read(self, noisy: bool = True) -> np.ndarray:
        g = self.conductance()
        if noisy and self.spec.read_noise > 0:
            g = g * (1.0 + self.spec.read_noise * self.rng.standard_normal(self.shape))
        return g

    # ----------------------------------------------------------- programming
    def apply_pulses(self, n: np.ndarray) -> int:
        """Send n[i] pulses to each device (n > 0 potentiate, n < 0 depress). Returns pulses sent."""
        n = np.rint(np.broadcast_to(n, self.shape)).astype(np.int64)
        sent = int(np.abs(n).sum())
        if sent == 0:
            return 0
        self.pulses += sent
        self.age = 0.0
        n = np.where(self.healthy, n, 0)         # stuck devices ignore pulses
        L = self.spec.levels
        g = self.normalized()
        new = g.copy()
        up, down = n > 0, n < 0
        if up.any():
            new[up] = _pot_step(g[up], n[up] / L, self.nl_pot[up])
        if down.any():
            new[down] = 1.0 - _pot_step(1.0 - g[down], -n[down] / L, self.nl_dep[down])
        if self.spec.c2c > 0:
            moved = up | down
            new[moved] += self.spec.c2c / L * np.sqrt(np.abs(n[moved])) * self.rng.standard_normal(int(moved.sum()))
        np.clip(new, 0.0, 1.0, out=new)
        self.g = self.g_min + new * (self.g_max - self.g_min)
        self._apply_stuck()
        return sent

    def reset(self, mask: np.ndarray | None = None) -> None:
        """One strong RESET pulse: drive the selected devices to their g_min."""
        mask = np.ones(self.shape, bool) if mask is None else mask
        self.g = np.where(mask, self.g_min, self.g)
        self._apply_stuck()
        count = int(mask.sum())
        self.pulses += count
        self.resets += count
        self.age = 0.0

    def _nominal_pulses(self, g_from: np.ndarray, g_to: np.ndarray) -> np.ndarray:
        """Signed pulse count that moves a *nominal* device of this technology from g_from to g_to."""
        s = self.spec

        def position(h: np.ndarray, nu: float) -> np.ndarray:  # inverse of the response curve, in pulses / L
            if nu < 1e-6:
                return h
            return -np.log1p(np.expm1(-nu) * h) / nu  # -ln(1 - h (1 - e^-nu)) / nu

        a = np.clip((g_from - s.g_min) / s.g_range, 0, 1)
        b = np.clip((g_to - s.g_min) / s.g_range, 0, 1)
        up = (position(b, s.nl_pot) - position(a, s.nl_pot)) * s.levels
        down = -(position(1 - b, s.nl_dep) - position(1 - a, s.nl_dep)) * s.levels
        return np.where(b >= a, up, down)

    def program(self, target: np.ndarray, tol: float | None = None, max_iter: int = 20,
                mask: np.ndarray | None = None, gain: float = 0.85) -> dict:
        """Closed-loop write-verify: read, compare to target, pulse, repeat.

        The controller knows the technology's *nominal* response curve, not each device's true
        one (variation, noise and faults are invisible to it), like real programming circuits.
        ``tol`` is a fraction of the nominal range (default: half a nominal step, at least 0.2 %).
        Devices with an abrupt RESET (strong depression nonlinearity, e.g. PCM) are never depressed
        partially: if they overshoot they are RESET and approached from below again.
        Stuck devices are not known to the controller; it keeps trying until ``max_iter``.
        """
        s = self.spec
        tol = self._default_tol(target) if tol is None else tol
        set_only = s.nl_dep >= 6.0
        pending = np.ones(self.shape, bool) if mask is None else mask.copy()
        sent0 = self.pulses
        it = 0
        for it in range(1, max_iter + 1):
            g = self.read()
            err = target - g
            pending &= np.abs(err) > tol * s.g_range
            if not pending.any():
                break
            if set_only:  # abrupt RESET: never depress partially; RESET and climb back from g_min
                over = pending & (err < 0)
                if over.any():
                    self.reset(over)
                    g = np.where(over, s.g_min, g)
                    err = target - g
            n = np.rint(gain * self._nominal_pulses(g, target))
            n = np.where(n == 0, np.sign(err), n)
            self.apply_pulses(np.where(pending, n, 0))
        return {"pulses": self.pulses - sent0, "iterations": it}

    def _default_tol(self, target: np.ndarray) -> np.ndarray:
        """Per-device tolerance (fraction of range): half of one nominal SET step at the target,
        plus the pulse noise, so the loop does not chase precision one pulse cannot deliver."""
        s = self.spec
        b = np.clip((target - s.g_min) / s.g_range, 0, 1)
        nu = max(s.nl_pot, 1e-6)
        local_step = nu * (1 + b * np.expm1(-nu)) / (-np.expm1(-nu) * s.levels)
        return np.maximum(0.5 * local_step + 0.5 * s.c2c / s.levels, 0.002)

    def program_open_loop(self, target: np.ndarray) -> dict:
        """Reset, then send the pulse count a *linear* device would need. No verification."""
        sent0 = self.pulses
        self.reset()
        self.apply_pulses(np.rint(np.clip(target - self.spec.g_min, 0, None) / self.spec.step))
        return {"pulses": self.pulses - sent0, "iterations": 1}


def pulse_response(spec: DeviceSpec, cycles: int = 1, seed: int = 0) -> np.ndarray:
    """Conductance of one (nominal, variation-free) device during `levels` up then `levels` down pulses."""
    one = DeviceArray(spec.with_(d2d=0.0, stuck_off=0.0, stuck_on=0.0), (1,), np.random.default_rng(seed))
    trace = [one.g[0]]
    for _ in range(cycles):
        for direction in (1, -1):
            for _ in range(spec.levels):
                one.apply_pulses(np.array([direction]))
                trace.append(one.g[0])
    return np.asarray(trace)
