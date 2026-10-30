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


class Picamera2Camera:
    """Raspberry Pi CSI camera (Camera Module 3 / OV5647) via libcamera.

    Use ``camera.device: picamera2`` in the config. Needs ``sudo apt install python3-picamera2``.
    Capturing straight at the working resolution keeps the Pi Zero 2 W's CPU free.
    """

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"}, controls={"FrameRate": float(cfg.fps)}
        )
        self.cam.configure(conf)
        self.cam.start()

    def read(self) -> np.ndarray | None:
        return self.cam.capture_array()  # RGB888 in picamera2 is BGR byte order, as OpenCV expects

    def close(self) -> None:
        self.cam.stop()


def open_camera(cfg: CameraConfig):
    if cfg.device == "picamera2":
        return Picamera2Camera(cfg)
    return OpenCVCamera(cfg)
