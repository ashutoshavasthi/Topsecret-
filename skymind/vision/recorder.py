"""Flight recorder: annotated video, raw detection snapshots and a JSONL log.

Run directory layout::

    <run>/
      flight.mp4          annotated video (HUD + detection boxes)
      snapshots/          raw JPEG frames where something was detected
      telemetry.jsonl     one JSON object per control/vision record
      report.json/.md     written by the judge after the flight
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [round(float(x), 4) for x in o.ravel()]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "value"):
        return o.value
    return o


class FlightRecorder:
    def __init__(self, run_dir: str | Path, fps: float, size: tuple[int, int], record_video: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(exist_ok=True)
        self.video_path = self.dir / "flight.mp4"
        self.writer = None
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, size)
            if not self.writer.isOpened():
                self.writer = None
        self.log = open(self.dir / "telemetry.jsonl", "w", encoding="utf-8")
        self.frames = 0
        self.snapshots = 0

    def write_frame(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)
        self.frames += 1

    def snapshot(self, frame: np.ndarray, t: float, labels: list[str]) -> str:
        name = f"t{t:07.2f}_{'-'.join(sorted(set(labels)))}.jpg"
        cv2.imwrite(str(self.snap_dir / name), frame)
        self.snapshots += 1
        return name

    def log_record(self, record: dict[str, Any]) -> None:
        self.log.write(json.dumps(_jsonable(record)) + "\n")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.log.close()

    def __enter__(self) -> "FlightRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_telemetry(run_dir: str | Path) -> list[dict[str, Any]]:
    with open(Path(run_dir) / "telemetry.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
