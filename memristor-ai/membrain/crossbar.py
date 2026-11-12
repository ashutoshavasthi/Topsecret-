"""Memristor crossbar: a matrix-vector multiply done by Ohm's and Kirchhoff's laws.

Inputs are applied as voltages on the rows; each device passes a current I = G * V (Ohm), and
the currents add up on each column wire (Kirchhoff). One read of the array therefore computes
y = x @ W for the whole matrix at once, where the weights are stored as conductances.

Signed weights use a differential pair per weight, W ~ (G+ - G-). Two mappings:

* ``dual``: W > 0 lives on G+, W < 0 on G-, the other device stays at g_min. Training only ever
  *potentiates* (on G+ or G-); when a device fills up the pair is "refreshed" (both RESET, the
  weight is re-written). This sidesteps the asymmetry between SET and RESET.
* ``ref``: G- is a fixed mid-range reference; G+ moves up and down. Simpler, but exposes the
  device's potentiation/depression asymmetry during training.

Periphery that real chips need, modelled here:

* DAC: inputs are scaled per sample by their abs-max (digital) and quantised to ``dac_bits``;
  negative inputs are applied in a second read phase.
* Tiling: physical arrays have at most ``tile_rows`` rows, so long dot products are split; each
  tile's column current is digitised separately and the partial sums are added digitally.
* ADC: column currents are quantised to ``adc_bits`` over a range calibrated on sample data.
* IR drop: wire resistance along rows and columns lowers the voltage each cell actually sees
  (solved iteratively for the whole array). Cells far from the drivers and sense amps lose most.
* Read noise: every read of every device is noisy (applied exactly, as the equivalent output noise).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .device import DeviceArray, DeviceSpec


@dataclass(frozen=True)
class Periphery:
    dac_bits: int = 8           # 0 = ideal
    adc_bits: int = 8           # 0 = ideal
    tile_rows: int = 128        # 0 = one unlimited array
    wire_ohm: float = 0.0       # wire resistance between neighbouring cells
    v_read: float = 0.2         # full-scale read voltage (V)
    t_read_ns: float = 10.0     # integration time per read
    refresh_at: float = 0.9     # dual mapping: refresh a pair when a device passes this fraction of its range

    def with_(self, **kw) -> "Periphery":
        return replace(self, **kw)


IDEAL_PERIPHERY = Periphery(dac_bits=0, adc_bits=0, tile_rows=0)


def _quant_unsigned(x: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return x
    q = 2 ** bits - 1
    return np.rint(np.clip(x, 0.0, 1.0) * q) / q


def _stochastic_round(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo = np.floor(x)
    return lo + (rng.random(x.shape) < (x - lo))


def ir_drop_currents(v: np.ndarray, g: np.ndarray, wire_ohm: float, iters: int = 4) -> np.ndarray:
    """Column currents (uA) of one array with wire resistance, for row voltages v (B, n) and G (n, m) in uS.

    Rows are driven at column 0; columns are sensed (virtual ground) at row n-1. Every wire
    segment between neighbouring cells has resistance ``wire_ohm``. Solved by fixed-point
    iteration on the node voltages, which converges quickly for realistic r * G * size.
    """
    r = wire_ohm * 1e-6  # ohm * uA -> V
    vr = np.broadcast_to(v[:, :, None], v.shape + (g.shape[1],)).copy()
    vc = np.zeros_like(vr)
    for _ in range(iters):
        cell = g * (vr - vc)                                       # (B, n, m) uA
        row_seg = np.flip(np.cumsum(np.flip(cell, -1), -1), -1)    # current entering each cell along the row
        vr = v[:, :, None] - r * np.cumsum(row_seg, -1)
        col_seg = np.cumsum(cell, 1)                               # current flowing down past each cell
        vc = r * np.flip(np.cumsum(np.flip(col_seg, 1), 1), 1)
    return (g * (vr - vc)).sum(axis=1)


class Crossbar:
    def __init__(self, n_in: int, n_out: int, device: DeviceSpec, periphery: Periphery = Periphery(),
                 w_max: float = 1.0, mapping: str = "dual", rng: np.random.Generator | None = None):
        if mapping not in ("dual", "ref"):
            raise ValueError(f"unknown mapping {mapping!r}")
        self.rng = rng or np.random.default_rng()
        self.n_in, self.n_out = n_in, n_out
        self.device, self.periphery = device, periphery
        self.w_max, self.mapping = float(w_max), mapping
        self.gp = DeviceArray(device, (n_in, n_out), self.rng)
        self.gn = DeviceArray(device, (n_in, n_out), self.rng)
        # Weight represented by one microsiemens of (G+ - G-).
        self.w_per_g = w_max / device.g_range if mapping == "dual" else 2 * w_max / device.g_range
        self.g_ref = 0.5 * (device.g_min + device.g_max)
        rows = periphery.tile_rows or n_in
        self.tiles = [(a, min(a + rows, n_in)) for a in range(0, n_in, rows)]
        self.adc_range: np.ndarray | None = None   # per tile, uA (None = dynamic per batch)
        self.refreshes = 0
        self.program(np.zeros((n_in, n_out)))

    # ------------------------------------------------------------- weights
    def target_conductances(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.clip(w, -self.w_max, self.w_max)
        g0 = self.device.g_min
        if self.mapping == "dual":
            return g0 + np.maximum(w, 0) / self.w_per_g, g0 + np.maximum(-w, 0) / self.w_per_g
        return self.g_ref + w / self.w_per_g, np.full(w.shape, self.g_ref)

    def program(self, w: np.ndarray, method: str = "write_verify", compensate: bool = True,
                tol: float | None = None, max_iter: int = 20) -> dict:
        """Write a weight matrix into the array (RESET everything first).

        ``open_loop``: pulse counts for an ideal linear device, no verification.
        ``write_verify``: closed-loop programming of every device. With ``compensate``, pairs that
        still miss their weight afterwards (stuck or badly-behaved devices) get their *partner*
        device re-targeted so the difference G+ - G- is right, a standard fault-tolerance trick.
        """
        w = np.clip(w, -self.w_max, self.w_max)
        tp, tn = self.target_conductances(w)
        before = self.pulses
        if method == "open_loop":
            self.gp.program_open_loop(tp)
            self.gn.program_open_loop(tn)
        else:
            self.gp.reset()
            self.gn.reset()
            self.gp.program(tp, tol=tol, max_iter=max_iter)
            self.gn.program(tn, tol=tol, max_iter=max_iter)
            if compensate:
                self._compensate(w, tol, max_iter)
        self.adc_range = None
        err = self.weights() - w
        return {"pulses": self.pulses - before,
                "weight_rms_error": float(np.sqrt((err ** 2).mean()) / self.w_max),
                "bad_weights": float((np.abs(err) > 0.1 * self.w_max).mean())}

    def _compensate(self, w: np.ndarray, tol: float | None, max_iter: int) -> None:
        gp, gn = self.gp.read(), self.gn.read()
        bad = np.abs((gp - gn) * self.w_per_g - w) > 0.05 * self.w_max
        if not bad.any():
            return
        tp, tn = self.target_conductances(w)
        lo, hi = self.device.g_min, self.device.g_max
        p_off = np.abs(gp - tp) >= np.abs(gn - tn)          # which device of the pair missed
        want_n = gp - w / self.w_per_g                       # move G- to match the stuck G+
        want_p = gn + w / self.w_per_g                       # or G+ to match the stuck G-
        new_tn, new_tp = np.clip(want_n, lo, hi), np.clip(want_p, lo, hi)
        # Only worth doing when the (possibly clipped) partner target actually shrinks the error.
        fix_n = bad & p_off & (np.abs(want_n - new_tn) < np.abs(gn - want_n))
        fix_p = bad & ~p_off & (np.abs(want_p - new_tp) < np.abs(gp - want_p))
        if fix_n.any():
            self.gn.reset(fix_n)
            self.gn.program(np.where(fix_n, new_tn, tn), tol=tol, max_iter=max_iter, mask=fix_n)
        if fix_p.any():
            self.gp.reset(fix_p)
            self.gp.program(np.where(fix_p, new_tp, tp), tol=tol, max_iter=max_iter, mask=fix_p)

    def weights(self, noisy: bool = False) -> np.ndarray:
        return (self.gp.read(noisy) - self.gn.read(noisy)) * self.w_per_g

    def set_age(self, seconds: float) -> None:
        self.gp.age = self.gn.age = float(seconds)

    @property
    def pulses(self) -> int:
        return self.gp.pulses + self.gn.pulses

    # -------------------------------------------------------------- reading
    def _tile_currents(self, v: np.ndarray) -> list[np.ndarray]:
        """Differential column current (uA) of every tile for normalised row inputs v in [0, 1]."""
        p = self.periphery
        gp, gn = self.gp.conductance(), self.gn.conductance()
        volts = v * p.v_read
        out = []
        for a, b in self.tiles:
            va = volts[:, a:b]
            if p.wire_ohm > 0:
                chunks = [ir_drop_currents(va[i:i + 128], gp[a:b], p.wire_ohm)
                          - ir_drop_currents(va[i:i + 128], gn[a:b], p.wire_ohm) for i in range(0, len(va), 128)]
                cur = np.concatenate(chunks)
            else:
                cur = va @ (gp[a:b] - gn[a:b])
            rn = self.device.read_noise
            if rn > 0:  # sum of independent per-device read noise == Gaussian output noise with this std
                var = (va ** 2) @ (gp[a:b] ** 2 + gn[a:b] ** 2)
                cur = cur + rn * np.sqrt(var) * self.rng.standard_normal(cur.shape)
            out.append(cur)
        return out

    def _adc(self, cur: np.ndarray, tile: int) -> np.ndarray:
        bits = self.periphery.adc_bits
        if bits <= 0:
            return cur
        rng = self.adc_range[tile] if self.adc_range is not None else max(np.abs(cur).max(), 1e-12)
        q = 2 ** (bits - 1) - 1
        return np.clip(np.rint(cur / rng * q), -q, q) * (rng / q)

    def _phases(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        scale = np.abs(x).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        xn = x / scale
        bits = self.periphery.dac_bits
        phases = [_quant_unsigned(np.maximum(xn, 0), bits)]
        if (xn < 0).any():
            phases.append(-_quant_unsigned(np.maximum(-xn, 0), bits))
        return scale, phases

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """y = x @ W computed by the analog array, x (B, n_in) -> (B, n_out)."""
        scale, phases = self._phases(x)
        total = np.zeros((x.shape[0], self.n_out))
        for ph in phases:
            sign = 1.0 if ph.min() >= 0 else -1.0
            for t, cur in enumerate(self._tile_currents(np.abs(ph))):
                total += sign * self._adc(cur, t)
        return total / self.periphery.v_read * self.w_per_g * scale

    def calibrate(self, x: np.ndarray, headroom: float = 1.0) -> None:
        """Set each tile's ADC full-scale from representative inputs (done once after programming)."""
        _, phases = self._phases(x)
        peak = np.zeros(len(self.tiles))
        for ph in phases:
            for t, cur in enumerate(self._tile_currents(np.abs(ph))):
                peak[t] = max(peak[t], np.abs(cur).max())
        self.adc_range = np.maximum(peak * headroom, 1e-12)

    def matvec_t(self, d: np.ndarray) -> np.ndarray:
        """Transposed read d @ W.T (inputs on the columns, read on the rows), used for backprop."""
        scale = np.abs(d).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        dn = d / scale
        bits = self.periphery.dac_bits
        gp, gn = self.gp.conductance(), self.gn.conductance()
        q = 2 ** (self.periphery.adc_bits - 1) - 1 if self.periphery.adc_bits > 0 else 0
        total = np.zeros((d.shape[0], self.n_in))
        for sign in (1.0, -1.0):
            v = _quant_unsigned(np.maximum(sign * dn, 0), bits) * self.periphery.v_read
            if not v.any():
                continue
            cur = v @ (gp - gn).T
            if self.device.read_noise > 0:
                var = (v ** 2) @ (gp ** 2 + gn ** 2).T
                cur = cur + self.device.read_noise * np.sqrt(var) * self.rng.standard_normal(cur.shape)
            if q:
                rng = max(np.abs(cur).max(), 1e-12)
                cur = np.clip(np.rint(cur / rng * q), -q, q) * (rng / q)
            total += sign * cur
        return total / self.periphery.v_read * self.w_per_g * scale

    # ------------------------------------------------------- in-situ update
    def update(self, dw: np.ndarray) -> int:
        """Apply a desired weight change as programming pulses (stochastically rounded). Returns pulses."""
        n = _stochastic_round(dw / (self.w_per_g * self.device.step), self.rng)
        if self.mapping == "ref":
            return self.gp.apply_pulses(n)
        sent = self.gp.apply_pulses(np.maximum(n, 0)) + self.gn.apply_pulses(np.maximum(-n, 0))
        return sent + self._refresh()

    def _refresh(self) -> int:
        """Dual mapping: re-write pairs whose devices are nearly full so they can keep learning."""
        full = (self.gp.normalized() > self.periphery.refresh_at) | (self.gn.normalized() > self.periphery.refresh_at)
        full &= self.gp.healthy & self.gn.healthy
        if not full.any():
            return 0
        w = self.weights(noisy=True)
        tp, tn = self.target_conductances(w)
        before = self.pulses
        self.gp.reset(full)
        self.gn.reset(full)
        self.gp.program(tp, mask=full & (tp > self.device.g_min))
        self.gn.program(tn, mask=full & (tn > self.device.g_min))
        self.refreshes += int(full.sum())
        return self.pulses - before
