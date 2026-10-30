"""Real camera input (USB / CSI / RTSP) via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from skymind.config import CameraConfig


class OpenCVCamera:
    """Works with a USB webcam index, a GStreamer pipeline string for the
    Raspberry Pi CSI camera, or an ``rtsp://`` URL from a gimbal camera."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {cfg.device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        return frame

    def close(self) -> None:
        self.cap.release()
