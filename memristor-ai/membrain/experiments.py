"""Experiments: deploy vs train-on-chip, non-ideality sweeps, drift over time, device curves."""
from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache

import numpy as np

from . import svg
from .crossbar import Periphery
from .data import train_test
from .device import PRESETS, DeviceSpec, pulse_response
from .energy import inference_energy, write_energy_uj
from .network import MLP, DenseLayer, TrainConfig, analog_mlp, default_w_max, float_mlp, to_analog, train

SIZES = [256, 64, 10]
HW_NOISE = 0.1          # hardware-aware training: weight noise std as a fraction of the weight range


@lru_cache(maxsize=4)
def load_data(n_train: int = 6000, n_test: int = 2000, seed: int = 0):
    return train_test(n_train, n_test, seed)


def _hw_aware_mlp(rng: np.random.Generator, noise: float = HW_NOISE) -> MLP:
    return MLP([DenseLayer(a, b, rng, default_w_max(a), noise) for a, b in zip(SIZES[:-1], SIZES[1:])])


def _quiet(_):
    pass


# --------------------------------------------------------------------------------------- demo
def run_demo(device: DeviceSpec, periphery: Periphery = Periphery(), epochs: int = 15, seed: int = 0,
             n_train: int = 6000, n_test: int = 2000, insitu: bool = True, log=print, chart: str | None = None) -> dict:
    """Train the same network several ways and compare them on one device technology."""
    xtr, ytr, xte, yte = load_data(n_train, n_test, seed)
    cfg = TrainConfig(epochs=epochs, seed=seed)
    cal = xtr[:500]
    rows, curves = [], {}

    def add(name: str, model: MLP, t0: float, kind: str, **extra) -> None:
        acc = model.accuracy(xte, yte)
        row = {"method": name, "kind": kind, "accuracy": acc, "seconds": round(time.time() - t0, 1), **extra}
        if any(layer.analog for layer in model.layers):
            e = inference_energy(model, xte[:200])
            row["inference_pj"] = e["analog_pj"]
            row["digital_pj"] = e["digital_pj"]
        if "pulses" in row:
            row["write_uj"] = write_energy_uj(row["pulses"], device.write_energy_pj)
        rows.append(row)
        log(f"  {name:<48} accuracy {acc * 100:5.1f} %")

    log(f"device: {device.name} - {device.description}")
    t0 = time.time()
    fm = float_mlp(SIZES, np.random.default_rng(seed))
    h = train(fm, xtr, ytr, cfg, xte, yte)
    curves["digital float"] = h
    add("digital float32 (reference)", fm, t0, "digital")

    t0 = time.time()
    hw = _hw_aware_mlp(np.random.default_rng(seed))
    h = train(hw, xtr, ytr, cfg, xte, yte)
    curves["digital, hardware-aware"] = h
    add("digital, hardware-aware training", hw, t0, "digital")

    for name, src, method, comp in [("deploy: open-loop programming", fm, "open_loop", False),
                                    ("deploy: write-verify", fm, "write_verify", False),
                                    ("deploy: write-verify + fault compensation", fm, "write_verify", True),
                                    ("deploy: hardware-aware + verify + compensation", hw, "write_verify", True)]:
        t0 = time.time()
        am, st = to_analog(src, device, periphery, np.random.default_rng(seed + 1), method=method, compensate=comp)
        am.calibrate(cal)
        add(name, am, t0, "ex-situ", pulses=st["pulses"], weight_rms_error=st["weight_rms_error"])

    if insitu:
        for mapping, label in [("dual", "on-chip training, dual (SET-only + refresh)"),
                               ("ref", "on-chip training, reference column")]:
            t0 = time.time()
            am = analog_mlp(SIZES, device, periphery, np.random.default_rng(seed + 2), mapping=mapping)
            h = train(am, xtr, ytr, cfg, xte, yte)
            am.calibrate(cal)
            curves[f"on-chip ({mapping})"] = h
            add(label, am, t0, "in-situ", pulses=am.pulses,
                refreshes=int(sum(layer.xbar.refreshes for layer in am.layers)))

    if chart:
        svg.line_chart({k: ([r["epoch"] for r in v], [100 * r["val_acc"] for r in v]) for k, v in curves.items()},
                       chart, f"Learning curves on {device.name}", "epoch", "test accuracy (%)")
    return {"device": device.as_dict(), "periphery": periphery.__dict__, "epochs": epochs, "rows": rows,
            "curves": {k: v for k, v in curves.items()}}


# -------------------------------------------------------------------------------------- sweep
DEVICE_KNOBS = {
    "levels": lambda d, v: d.with_(levels=int(v)),
    "nonlinearity": lambda d, v: d.with_(nl_pot=v, nl_dep=v),
    "c2c": lambda d, v: d.with_(c2c=v),
    "d2d": lambda d, v: d.with_(d2d=v),
    "read_noise": lambda d, v: d.with_(read_noise=v),
    "stuck": lambda d, v: d.with_(stuck_off=v / 2, stuck_on=v / 2),
}
PERIPHERY_KNOBS = {"adc_bits", "dac_bits", "wire_ohm", "tile_rows"}
KNOB_LABELS = {
    "levels": "pulses (levels) across the conductance range", "nonlinearity": "pulse-response nonlinearity",
    "c2c": "cycle-to-cycle noise (steps)", "d2d": "device-to-device variation", "read_noise": "read noise (relative)",
    "stuck": "fraction of stuck devices", "adc_bits": "ADC resolution (bits)", "dac_bits": "DAC resolution (bits)",
    "wire_ohm": "wire resistance per cell (ohm)", "tile_rows": "rows per physical array",
}


def run_sweep(knob: str, values: list[float], device: DeviceSpec, periphery: Periphery = Periphery(),
              epochs: int = 15, seed: int = 0, repeats: int = 2, insitu: bool = False,
              log=print, chart: str | None = None) -> dict:
    """Accuracy as one non-ideality is varied, for deployment (and optionally on-chip training)."""
    if knob not in DEVICE_KNOBS and knob not in PERIPHERY_KNOBS:
        raise ValueError(f"unknown knob {knob!r}; choose from {sorted(DEVICE_KNOBS) + sorted(PERIPHERY_KNOBS)}")
    xtr, ytr, xte, yte = load_data(seed=seed)
    cfg = TrainConfig(epochs=epochs, seed=seed)
    fm = float_mlp(SIZES, np.random.default_rng(seed))
    train(fm, xtr, ytr, cfg)
    hw = _hw_aware_mlp(np.random.default_rng(seed))
    train(hw, xtr, ytr, cfg)
    ref = fm.accuracy(xte, yte)
    log(f"digital float reference: {ref * 100:.1f} %")
    results = {"deploy (write-verify)": [], "deploy (hardware-aware)": []}
    if insitu:
        results["on-chip training"] = []
    for v in values:
        dev, per = device, periphery
        if knob in DEVICE_KNOBS:
            dev = DEVICE_KNOBS[knob](device, v)
        else:
            per = periphery.with_(**{knob: int(v) if knob != "wire_ohm" else float(v)})
        line = []
        for label, src in [("deploy (write-verify)", fm), ("deploy (hardware-aware)", hw)]:
            accs = []
            for r in range(repeats):
                am, _ = to_analog(src, dev, per, np.random.default_rng(1000 * r + 7))
                am.calibrate(xtr[:500])
                accs.append(am.accuracy(xte, yte))
            results[label].append(float(np.mean(accs)))
            line.append(f"{label} {np.mean(accs) * 100:5.1f} %")
        if insitu:
            mapping = "dual" if dev.nl_dep >= 6 else "ref"
            am = analog_mlp(SIZES, dev, per, np.random.default_rng(seed + 2), mapping=mapping)
            train(am, xtr, ytr, cfg)
            am.calibrate(xtr[:500])
            results["on-chip training"].append(am.accuracy(xte, yte))
            line.append(f"on-chip {results['on-chip training'][-1] * 100:5.1f} %")
        log(f"  {knob} = {v:<8g} " + " | ".join(line))
    if chart:
        series = {k: (list(values), [100 * a for a in accs]) for k, accs in results.items()}
        series["digital float"] = (list(values), [100 * ref] * len(values))
        log_x = knob in ("levels", "read_noise", "stuck", "c2c") and min(values) > 0 and max(values) / min(values) >= 20
        svg.line_chart(series, chart, f"{device.name}: accuracy vs {knob}", KNOB_LABELS.get(knob, knob),
                       "test accuracy (%)", log_x=log_x)
    return {"knob": knob, "values": list(values), "device": device.as_dict(), "reference": ref, "results": results}


# -------------------------------------------------------------------------------------- drift
TIMES = [("1 s", 1.0), ("1 min", 60.0), ("1 h", 3600.0), ("1 day", 86400.0), ("1 month", 2.6e6), ("1 year", 3.15e7),
         ("10 years", 3.15e8)]


def run_drift(device: DeviceSpec, periphery: Periphery = Periphery(), epochs: int = 15, seed: int = 0,
              log=print, chart: str | None = None) -> dict:
    """Program once, then read the same chip later: raw vs with global drift compensation."""
    xtr, ytr, xte, yte = load_data(seed=seed)
    hw = _hw_aware_mlp(np.random.default_rng(seed))
    train(hw, xtr, ytr, TrainConfig(epochs=epochs, seed=seed))
    am, _ = to_analog(hw, device, periphery, np.random.default_rng(seed + 1))
    am.calibrate(xtr[:500])
    cal = xtr[500:756]
    am.drift_compensate(cal, reference=True)
    raw, comp = [], []
    for label, t in TIMES:
        am.set_age(t)
        for layer in am.layers:
            layer.out_scale = 1.0
        raw.append(am.accuracy(xte, yte))
        am.drift_compensate(cal)
        comp.append(am.accuracy(xte, yte))
        g_ratio = float(np.mean([layer.xbar.gp.conductance().mean() / layer.xbar.gp.g.mean() for layer in am.layers]))
        log(f"  {label:>9}: conductance x{g_ratio:.2f}  raw {raw[-1] * 100:5.1f} %   compensated {comp[-1] * 100:5.1f} %")
    if chart:
        xs = [t for _, t in TIMES]
        svg.line_chart({"no compensation": (xs, [100 * a for a in raw]),
                        "global drift compensation": (xs, [100 * a for a in comp])},
                       chart, f"{device.name}: accuracy after programming", "time since programming", "test accuracy (%)",
                       log_x=True, x_tick_labels={t: lab for lab, t in TIMES})
    return {"device": device.as_dict(), "times": [lab for lab, _ in TIMES], "raw": raw, "compensated": comp}


# ------------------------------------------------------------------------------------ devices
def device_curves(names: list[str] | None = None, chart: str | None = None) -> dict:
    """Pulse response of each preset: `levels` SET pulses then `levels` RESET pulses (normalised)."""
    out = {}
    for name in names or list(PRESETS):
        spec = PRESETS[name]
        tr = pulse_response(spec)
        out[name] = {"pulses": np.linspace(0, 2, len(tr)).tolist(),
                     "g_norm": ((tr - spec.g_min) / spec.g_range).tolist()}
    if chart:
        svg.line_chart({k: (v["pulses"], v["g_norm"]) for k, v in out.items()}, chart,
                       "Device response: full-range SET pulses, then RESET pulses",
                       "pulses / levels (0-1 potentiate, 1-2 depress)", "normalised conductance", y_range=(-0.02, 1.02))
    return out


def with_periphery(base: Periphery, **kw) -> Periphery:
    return replace(base, **{k: v for k, v in kw.items() if v is not None})
