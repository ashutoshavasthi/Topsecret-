"""Command line: python -m membrain <command>."""
from __future__ import annotations

import argparse
import json
import os
import time

from . import experiments as ex
from .crossbar import Periphery
from .device import PRESETS, get_device


def _out_dir(arg: str | None, name: str) -> str:
    path = arg or os.path.join("runs", f"{name}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(path, exist_ok=True)
    return path


def _periphery(a) -> Periphery:
    return ex.with_periphery(Periphery(), adc_bits=a.adc_bits, dac_bits=a.dac_bits, wire_ohm=a.wire_ohm,
                             tile_rows=a.tile_rows)


def _dump(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)


def _pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def demo_report(res: dict) -> str:
    d = res["device"]
    lines = [f"# MemBrain run on `{d['name']}`", "", d["description"], "",
             f"Network 256-64-10, {res['epochs']} epochs, procedural digits (6000 train / 2000 test).", "",
             "| Method | Test accuracy | Programming pulses | Write energy | Inference energy | vs digital 8-bit |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in res["rows"]:
        pulses = f"{r['pulses']:,}" if "pulses" in r else "-"
        wu = f"{r['write_uj']:.1f} uJ" if "write_uj" in r else "-"
        ie = f"{r['inference_pj'] / 1000:.2f} nJ" if "inference_pj" in r else "-"
        adv = f"{r['digital_pj'] / r['inference_pj']:.0f}x less" if "inference_pj" in r else "-"
        lines.append(f"| {r['method']} | {_pct(r['accuracy'])} | {pulses} | {wu} | {ie} | {adv} |")
    lines += ["", "![learning curves](learning_curves.svg)", ""]
    return "\n".join(lines)


def cmd_demo(a) -> None:
    out = _out_dir(a.out, "demo")
    names = list(PRESETS) if a.device == "all" else a.device.split(",")
    allres = {}
    for name in names:
        dev = get_device(name)
        sub = out if len(names) == 1 else os.path.join(out, name)
        os.makedirs(sub, exist_ok=True)
        res = ex.run_demo(dev, _periphery(a), epochs=a.epochs, seed=a.seed, insitu=not a.no_insitu,
                          chart=os.path.join(sub, "learning_curves.svg"))
        with open(os.path.join(sub, "report.md"), "w", encoding="utf-8") as f:
            f.write(demo_report(res))
        _dump(os.path.join(sub, "results.json"), res)
        allres[name] = res
        print()
    if len(names) > 1:
        lines = ["| Device | " + " | ".join(r["method"] for r in allres[names[0]]["rows"]) + " |",
                 "|---|" + "---:|" * len(allres[names[0]]["rows"])]
        for name, res in allres.items():
            lines.append(f"| {name} | " + " | ".join(_pct(r["accuracy"]) for r in res["rows"]) + " |")
        table = "\n".join(lines)
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
            f.write("# Accuracy by device and method\n\n" + table + "\n")
        print(table)
    print(f"\nwrote {out}/")


def cmd_sweep(a) -> None:
    out = _out_dir(a.out, f"sweep-{a.knob}")
    values = [float(v) for v in a.values.split(",")]
    res = ex.run_sweep(a.knob, values, get_device(a.device), _periphery(a), epochs=a.epochs, seed=a.seed,
                       repeats=a.repeats, insitu=a.insitu, chart=os.path.join(out, f"sweep_{a.knob}.svg"))
    _dump(os.path.join(out, "results.json"), res)
    print(f"wrote {out}/")


def cmd_drift(a) -> None:
    out = _out_dir(a.out, "drift")
    res = ex.run_drift(get_device(a.device), _periphery(a), epochs=a.epochs, seed=a.seed,
                       chart=os.path.join(out, "drift.svg"))
    _dump(os.path.join(out, "results.json"), res)
    print(f"wrote {out}/")


def cmd_devices(a) -> None:
    out = _out_dir(a.out, "devices")
    print(f"{'device':<7} {'G range (uS)':>14} {'levels':>6} {'NL +/-':>9} {'c2c':>5} {'read':>5} {'stuck':>6} {'drift':>6}"
          f" {'pJ/pulse':>8}")
    for name, s in PRESETS.items():
        print(f"{name:<7} {s.g_min:>6g} - {s.g_max:<6g} {s.levels:>6} {s.nl_pot:>4g}/{s.nl_dep:<4g} {s.c2c:>5g} "
              f"{s.read_noise:>5g} {s.stuck_off + s.stuck_on:>6.3f} {s.drift_nu:>6g} {s.write_energy_pj:>8g}")
        print(f"        {s.description}")
    ex.device_curves(chart=os.path.join(out, "device_curves.svg"))
    print(f"wrote {out}/device_curves.svg")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="membrain", description="Neural networks on simulated memristor crossbars")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, device="rram"):
        sp.add_argument("--device", default=device, help=f"one of {', '.join(PRESETS)}" + (" or 'all'" if sp.prog.endswith("demo") else ""))
        sp.add_argument("--epochs", type=int, default=15)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", help="output directory (default runs/<command>-<time>)")
        sp.add_argument("--adc-bits", type=int)
        sp.add_argument("--dac-bits", type=int)
        sp.add_argument("--wire-ohm", type=float)
        sp.add_argument("--tile-rows", type=int)

    sp = sub.add_parser("demo", help="train one network several ways (digital, deployed, trained on-chip) and compare")
    common(sp)
    sp.add_argument("--no-insitu", action="store_true", help="skip on-chip training (faster)")
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("sweep", help="accuracy vs one non-ideality")
    common(sp)
    sp.add_argument("--knob", required=True, choices=sorted(ex.DEVICE_KNOBS) + sorted(ex.PERIPHERY_KNOBS))
    sp.add_argument("--values", required=True, help="comma-separated values")
    sp.add_argument("--repeats", type=int, default=2, help="chips programmed per value (averaged)")
    sp.add_argument("--insitu", action="store_true", help="also train on-chip at every value (slow)")
    sp.set_defaults(fn=cmd_sweep)

    sp = sub.add_parser("drift", help="accuracy over time after programming, with and without drift compensation")
    common(sp, device="pcm")
    sp.set_defaults(fn=cmd_drift)

    sp = sub.add_parser("devices", help="list device presets and plot their pulse response")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_devices)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
