"""Onboard image analysis.

Two detector back-ends share one interface:

* :class:`OnnxYoloDetector` runs a YOLOv8 ONNX model through OpenCV DNN
  (people, vehicles, animals, ... whatever the model was trained on). It runs
  on a Raspberry Pi 5 / Jetson without PyTorch installed.
* :class:`ColorTargetDetector` finds coloured markers by HSV thresholding.
  It needs no model file, so the full pipeline works out of the box and in
  the simulator.

:class:`FrameAnalyzer` combines detection with image-quality metrics
(sharpness, exposure, contrast) that the judge uses to grade the footage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from skymind.config import VisionConfig


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityMetrics:
    sharpness: float  # variance of Laplacian, higher = crisper
    brightness: float  # mean luma 0..255
    contrast: float  # std of luma
    overexposed: float  # fraction of pixels >= 250
    underexposed: float  # fraction of pixels <= 5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameAnalysis:
    detections: list[Detection] = field(default_factory=list)
    quality: QualityMetrics | None = None

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "quality": self.quality.to_dict() if self.quality else None,
        }


def quality_metrics(frame: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    # A light pre-blur keeps sensor noise from masquerading as detail.
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
    return QualityMetrics(
        sharpness=float(lap.var()),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        overexposed=float(np.mean(gray >= 250)),
        underexposed=float(np.mean(gray <= 5)),
    )


class ColorTargetDetector:
    def __init__(self, targets: dict[str, list], min_area: int = 12):
        # Each label maps to one [lo, hi] HSV range or a list of them (red wraps around hue 0/180).
        self.targets = {}
        for label, spec in targets.items():
            arr = np.array(spec, dtype=np.uint8)
            self.targets[label] = arr.reshape(-1, 2, 3)
        self.min_area = min_area
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for label, ranges in self.targets.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                fill = area / max(w * h, 1)
                # Discs fill ~78% of their box; confidence grows with size and roundness.
                conf = float(np.clip(0.5 + 0.4 * min(area / 400.0, 1.0) + 0.1 * (1 - abs(fill - 0.785) * 2), 0, 1))
                out.append(Detection(label, round(conf, 3), (int(x), int(y), int(w), int(h))))
        return out


class OnnxYoloDetector:
    """YOLOv8 ONNX (exported with ``yolo export format=onnx imgsz=N``), NxN input."""

    def __init__(self, model_path: str, labels: list[str], conf_threshold: float = 0.4, nms_threshold: float = 0.45, input_size: int = 640):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.labels = labels
        self.conf = conf_threshold
        self.nms = nms_threshold
        self.size = input_size  # must match the size the model was exported with

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 4 + classes, N)
        preds = np.squeeze(out, 0).T
        scores = preds[:, 4:]
        cls = np.argmax(scores, axis=1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        preds, cls, conf = preds[keep], cls[keep], conf[keep]
        sx, sy = w / self.size, h / self.size
        boxes = [
            [int((p[0] - p[2] / 2) * sx), int((p[1] - p[3] / 2) * sy), int(p[2] * sx), int(p[3] * sy)] for p in preds
        ]
        idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.conf, self.nms) if boxes else []
        result = []
        for i in np.array(idx).flatten():
            label = self.labels[cls[i]] if cls[i] < len(self.labels) else f"class_{cls[i]}"
            result.append(Detection(label, float(conf[i]), tuple(boxes[i])))
        return result


class FrameAnalyzer:
    def __init__(self, cfg: VisionConfig):
        self.detectors = [ColorTargetDetector(cfg.color_targets, cfg.min_blob_area)]
        if cfg.onnx_model:
            if not Path(cfg.onnx_model).exists():
                raise FileNotFoundError(f"ONNX model not found: {cfg.onnx_model}")
            self.detectors.append(
                OnnxYoloDetector(cfg.onnx_model, cfg.onnx_labels, cfg.onnx_conf_threshold, input_size=cfg.onnx_input_size)
            )

    def analyze(self, frame: np.ndarray) -> FrameAnalysis:
        dets: list[Detection] = []
        for d in self.detectors:
            dets.extend(d.detect(frame))
        return FrameAnalysis(dets, quality_metrics(frame))


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def draw_overlay(frame: np.ndarray, analysis: FrameAnalysis, hud: list[str] | None = None) -> np.ndarray:
    img = frame.copy()
    for d in analysis.detections:
        x, y, w, h = d.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1)
    if hud:
        box_h = 6 + 13 * len(hud)
        region = img[:box_h, :170]
        region[:] = (region * 0.45).astype(np.uint8)
        for i, line in enumerate(hud):
            cv2.putText(img, line, (4, 13 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img
