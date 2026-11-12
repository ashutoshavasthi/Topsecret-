"""Order-of-magnitude energy model: memristor crossbar inference vs a digital 8-bit accelerator.

The crossbar energy is *measured* from the simulation (actual voltages and conductances of every
read); the peripheral costs (DAC, ADC, digital accumulation) and the digital baseline use
textbook per-operation figures. Treat the comparison as a rough guide, not a chip measurement.

Digital baseline (45 nm figures from M. Horowitz, "Computing's energy problem", ISSCC 2014):
8-bit multiply 0.2 pJ, 32-bit add 0.1 pJ, 32-bit read from an 8 KB SRAM 5 pJ (so 1.25 pJ per
8-bit weight). The baseline fetches every weight from SRAM once per inference (no batching).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import MLP, _with_bias


@dataclass(frozen=True)
class EnergyModel:
    dac_pj: float = 0.1          # one input conversion (per row, per read phase)
    adc_pj_8bit: float = 1.0     # one 8-bit SAR conversion (~4 fJ per conversion step); scales as 2**bits
    digital_add_pj: float = 0.1  # accumulate one tile's partial sum
    mac_pj: float = 0.2 + 0.1    # 8-bit multiply + 32-bit accumulate
    sram_weight_pj: float = 1.25 # fetch one 8-bit weight from on-chip SRAM


def inference_energy(model: MLP, x: np.ndarray, em: EnergyModel = EnergyModel()) -> dict:
    """Mean energy per inference (pJ) of the analog network, broken down, plus the digital baseline."""
    parts = {"array": 0.0, "dac": 0.0, "adc": 0.0, "digital": 0.0}
    macs = 0
    h = x
    for i, layer in enumerate(model.layers):
        xin = _with_bias(h)
        macs += xin.shape[1] * layer.w.shape[1]
        if layer.analog:
            xb = layer.xbar
            p = xb.periphery
            scale = np.abs(xin).max(axis=1, keepdims=True)
            scale[scale == 0] = 1.0
            xn = xin / scale
            g = xb.gp.conductance() + xb.gn.conductance()
            phases = 1 + int((xn < 0).any())
            v2 = (xn * p.v_read) ** 2                      # V^2, both polarities
            # V^2 * uS * ns = 1e-15 J = 1e-3 pJ
            parts["array"] += float((v2 @ g).sum(axis=1).mean() * p.t_read_ns * 1e-3)
            parts["dac"] += phases * xin.shape[1] * em.dac_pj
            bits = p.adc_bits or 8
            parts["adc"] += phases * len(xb.tiles) * xb.n_out * em.adc_pj_8bit * 2 ** (bits - 8)
            parts["digital"] += phases * len(xb.tiles) * xb.n_out * em.digital_add_pj
        h = layer.forward(h)
        if i < len(model.layers) - 1:
            h = np.maximum(h, 0.0)
    analog = sum(parts.values())
    digital = macs * (em.mac_pj + em.sram_weight_pj)
    return {"analog_pj": analog, "breakdown_pj": parts, "digital_pj": digital, "macs": macs,
            "advantage": digital / analog if analog else float("inf")}


def write_energy_uj(pulses: int, write_pj: float) -> float:
    return pulses * write_pj * 1e-6
