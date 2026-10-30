"""Analyse an existing video file (e.g. footage pulled off a real drone's SD card)."""

from __future__ import annotations

from pathlib import Path

import cv2

from skymind.config import VisionConfig
from skymind.vision.analyzer import FrameAnalyzer, draw_overlay
from skymind.vision.recorder import FlightRecorder


def analyze_video(video: str | Path, cfg: VisionConfig, out_dir: str | Path, every: int = 1) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analyzer = FrameAnalyzer(cfg)
    records: list[dict] = []
    last_labels: set[str] = set()
    last_snap_t = -1e9
    with FlightRecorder(out_dir, fps / every, (w, h)) as rec:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                t = i / fps
                analysis = analyzer.analyze(frame)
                labels = {d.label for d in analysis.detections}
                snap = None
                if cfg.snapshot_on_detection and labels and (labels != last_labels or t - last_snap_t > 3.0):
                    snap = rec.snapshot(frame, t, list(labels))
                    last_snap_t = t
                last_labels = labels
                rec.write_frame(draw_overlay(frame, analysis, [f"t {t:6.2f}s  {len(analysis.detections)} det"]))
                record = {"type": "vision", "t": t, **analysis.to_dict(), "snapshot": snap}
                rec.log_record(record)
                records.append(record)
            i += 1
    cap.release()
    return records
