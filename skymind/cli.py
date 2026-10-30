"""Command line interface.

    skymind sim     [--config FILE] [--out DIR]   fly a simulated mission, record, judge
    skymind judge   RUN_DIR [--config FILE]       re-score an existing run
    skymind analyze VIDEO [--out DIR]             AI analysis + quality scoring of any video
    skymind fly     --connect URL [--arm]         run on a real drone (bench mode unless --arm)
    skymind design  CONFIG... [--sweep-battery]   compare builds: weight, power, endurance, noise
    skymind bench   [--config FILE]               companion-computer CPU budget
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from skymind.config import config_to_dict, load_config


def _default_out(prefix: str) -> Path:
    return Path("runs") / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _print_card(card) -> None:
    print(f"\n=== SCORE {card.overall:.1f}/100  grade {card.grade}  ({card.verdict}) ===")
    for c in card.categories:
        print(f"  {c.name:<11} {c.score:6.1f}   (weight {c.weight:.2f})")


def cmd_sim(args: argparse.Namespace) -> int:
    from skymind.evaluation.judge import FlightJudge, write_report
    from skymind.evaluation.plots import flight_map, timeline
    from skymind.mission import SimulationRunner
    from skymind.vision.recorder import load_telemetry

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.sim.seed = args.seed
    out = Path(args.out) if args.out else _default_out("sim")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")

    print(f"Flying simulated mission -> {out}")
    result = SimulationRunner(cfg, out, record_video=not args.no_video, verbose=True).run()
    (out / "mission.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    records = load_telemetry(out)
    card = FlightJudge(cfg.judge).judge(records, result.to_dict())
    write_report(card, out)
    flight_map(records, cfg, out / "flight_map.png")
    timeline(records, out / "timeline.png")

    print(
        f"\nstatus={result.status} waypoints={result.waypoints_reached}/{result.waypoints_total} "
        f"collisions={result.collisions} min_clearance={result.min_clearance} m "
        f"duration={result.duration} s (simulated in {result.wall_time} s)"
    )
    print(f"recorded {result.frames_recorded} frames, {result.snapshots} detection snapshots")
    _print_card(card)
    print(f"\nreport: {out / 'report.md'}\nvideo:  {out / 'flight.mp4'}\nmap:    {out / 'flight_map.png'}")
    return 0 if card.verdict == "PASS" else 1


def cmd_judge(args: argparse.Namespace) -> int:
    from skymind.evaluation.judge import FlightJudge, write_report
    from skymind.vision.recorder import load_telemetry

    run = Path(args.run_dir)
    cfg = load_config(args.config)
    mission_file = run / "mission.json"
    mission = json.loads(mission_file.read_text()) if mission_file.exists() else None
    card = FlightJudge(cfg.judge).judge(load_telemetry(run), mission)
    write_report(card, run)
    _print_card(card)
    print(f"\nreport: {run / 'report.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from skymind.evaluation.judge import FlightJudge, write_report
    from skymind.vision.offline import analyze_video

    cfg = load_config(args.config)
    if args.model:
        cfg.vision.onnx_model = args.model
    if args.labels:
        cfg.vision.onnx_labels = Path(args.labels).read_text().split()
    out = Path(args.out) if args.out else _default_out("analysis")
    records = analyze_video(args.video, cfg.vision, out, every=args.every)
    card = FlightJudge(cfg.judge).judge(records)
    write_report(card, out)
    counts: dict[str, int] = {}
    for r in records:
        for d in r["detections"]:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
    print(f"analysed {len(records)} frames; detections by label: {counts or 'none'}")
    vision = next(c for c in card.categories if c.name == "vision")
    print(f"vision score {vision.score:.1f}/100")
    for m in vision.metrics:
        print(f"  {m.name:<20} {m.value:10.3f}  -> {m.score:5.1f}")
    print(f"\nannotated video: {out / 'flight.mp4'}\nreport: {out / 'report.md'}")
    return 0


def cmd_fly(args: argparse.Namespace) -> int:
    from skymind.hal.mavlink_bridge import MavlinkVehicle
    from skymind.hal.real_flight import RealFlightRunner
    from skymind.vision.camera import open_camera

    cfg = load_config(args.config)
    if bool(args.connect) == bool(args.esp_drone):
        print("give exactly one of --connect (MAVLink autopilot) or --esp-drone (ESP-Drone over WiFi)", file=sys.stderr)
        return 2
    if args.arm and not args.yes:
        reply = input("ARM MOTORS AND FLY AUTONOMOUSLY? Area clear, someone ready to stop it? type 'fly': ")
        if reply.strip() != "fly":
            print("aborted")
            return 1
    if args.esp_drone:
        from skymind.hal.esp_drone import EspDroneLink

        vehicle = EspDroneLink(args.esp_drone, max_altitude=cfg.mission.max_altitude)
    else:
        vehicle = MavlinkVehicle(args.connect, args.baud)

    if args.esp_drone:
        from skymind.hal.esp_sensors import EspCamSensors

        lidar = EspCamSensors(cfg.sensors, port=args.sensor_port)  # ToF readings from the ESP32-CAM
    elif cfg.sensors.range_sensor == "tof_ring":
        from skymind.hal.tof_ring import ToFRing

        lidar = ToFRing(cfg.sensors, args.xshut.split(",") if args.xshut else None)
    elif args.lidar:
        from skymind.hal.lidar_rplidar import RPLidarSource

        lidar = RPLidarSource(args.lidar, cfg.sensors.lidar_beams, cfg.sensors.lidar_max_range)
    else:
        print("WARNING: no range sensor configured -- obstacle avoidance is blind", file=sys.stderr)
        import numpy as np

        class _NoLidar:
            body_angles = np.linspace(-np.pi, np.pi, cfg.sensors.lidar_beams, endpoint=False)

            def scan(self):
                return np.full(len(self.body_angles), cfg.sensors.lidar_max_range), self.body_angles

        lidar = _NoLidar()
    camera = None if args.no_camera else open_camera(cfg.camera)
    out = Path(args.out) if args.out else _default_out("flight")
    RealFlightRunner(cfg, vehicle, lidar, camera, out, arm=args.arm).run()
    print(f"log written to {out}; score it with: skymind judge {out}")
    return 0


def cmd_design(args: argparse.Namespace) -> int:
    from skymind.design import evaluate, format_table, with_battery

    cfgs = [(Path(c).stem, load_config(c)) for c in args.configs]
    ref_cfg = load_config(args.reference) if args.reference else cfgs[0][1]
    ref_tip = evaluate(ref_cfg).hover_tip_speed
    reports = [evaluate(cfg, name, ref_tip) for name, cfg in cfgs]
    if args.sweep_battery:
        name, base = cfgs[0]
        for wh in [float(x) for x in args.sweep_battery.split(",")]:
            reports.append(evaluate(with_battery(base, wh), f"{name}@{wh:g}Wh", ref_tip))
    print(format_table(reports))
    print(f"\nnoise dB is relative to {'the reference' if args.reference else cfgs[0][0]} hovering (rotor tonal noise ~ 50 log10 tip speed)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from skymind.bench import format_bench, run_bench

    print(format_bench(run_bench(load_config(args.config), args.iterations)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skymind", description="AI-powered self-balancing, obstacle-avoiding camera drone")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sim", help="fly a simulated mission, record video and judge it")
    s.add_argument("--config", default="config/default.yaml")
    s.add_argument("--out")
    s.add_argument("--seed", type=int)
    s.add_argument("--no-video", action="store_true")
    s.set_defaults(func=cmd_sim)

    j = sub.add_parser("judge", help="score an existing run directory")
    j.add_argument("run_dir")
    j.add_argument("--config", default="config/default.yaml")
    j.set_defaults(func=cmd_judge)

    a = sub.add_parser("analyze", help="run AI detection and quality scoring on a video file")
    a.add_argument("video")
    a.add_argument("--config", default="config/default.yaml")
    a.add_argument("--out")
    a.add_argument("--model", help="YOLOv8 ONNX model")
    a.add_argument("--labels", help="text file with one class name per line")
    a.add_argument("--every", type=int, default=1, help="analyse every Nth frame")
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("fly", help="run on a real drone via MAVLink (bench mode unless --arm)")
    f.add_argument("--connect", help="MAVLink autopilot, e.g. /dev/ttyAMA0, udp:127.0.0.1:14550")
    f.add_argument("--esp-drone", help="ESP-Drone / Crazyflie URI, e.g. udp://192.168.43.42 (Nano build)")
    f.add_argument("--sensor-port", type=int, default=2399, help="UDP port of the ESP32-CAM ToF broadcasts")
    f.add_argument("--baud", type=int, default=921600)
    f.add_argument("--lidar", help="RPLidar serial port, e.g. /dev/ttyUSB0")
    f.add_argument("--xshut", help="ToF ring XSHUT GPIOs, e.g. D17,D27,D22,D23,D24,D25 (config range_sensor: tof_ring)")
    f.add_argument("--config", default="config/default.yaml")
    f.add_argument("--out")
    f.add_argument("--no-camera", action="store_true")
    f.add_argument("--arm", action="store_true", help="actually arm and fly (default: bench/monitor only)")
    f.add_argument("--yes", action="store_true", help="skip the interactive arming confirmation")
    f.set_defaults(func=cmd_fly)
    d = sub.add_parser("design", help="compare builds: weight, power, endurance, noise")
    d.add_argument("configs", nargs="+")
    d.add_argument("--reference", help="config whose hover noise is the 0 dB reference")
    d.add_argument("--sweep-battery", help="comma-separated pack sizes in Wh for the first config")
    d.set_defaults(func=cmd_design)

    b = sub.add_parser("bench", help="measure companion-computer CPU load for a config")
    b.add_argument("--config", default="config/default.yaml")
    b.add_argument("--iterations", type=int, default=60)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
