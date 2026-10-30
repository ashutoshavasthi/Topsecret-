import cv2
import numpy as np

from skymind.config import CameraConfig, VisionConfig, WorldConfig
from skymind.sim.camera_sim import SimCamera
from skymind.sim.world import World
from skymind.vision.analyzer import FrameAnalyzer, iou, quality_metrics


def _frame_with_marker(colour=(30, 30, 220)):
    img = np.full((240, 320, 3), (90, 140, 100), np.uint8)
    cv2.circle(img, (160, 120), 15, colour, -1)
    return img


def test_detects_red_marker():
    dets = FrameAnalyzer(VisionConfig()).analyze(_frame_with_marker()).detections
    assert len(dets) == 1 and dets[0].label == "red_marker"
    assert iou(dets[0].bbox, (145, 105, 31, 31)) > 0.7


def test_detects_yellow_marker_and_no_false_positive():
    an = FrameAnalyzer(VisionConfig())
    assert [d.label for d in an.analyze(_frame_with_marker((20, 210, 240))).detections] == ["yellow_marker"]
    assert an.analyze(np.full((240, 320, 3), (90, 140, 100), np.uint8)).detections == []


def test_blur_lowers_sharpness():
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320)) > 0.5).astype(np.uint8) * 255
    img = cv2.cvtColor(cv2.resize(img[::8, ::8], (320, 240), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    sharp = quality_metrics(img).sharpness
    blurred = quality_metrics(cv2.GaussianBlur(img, (15, 15), 5)).sharpness
    assert blurred < sharp * 0.3


def test_exposure_metrics():
    q = quality_metrics(np.full((10, 10, 3), 255, np.uint8))
    assert q.overexposed == 1.0 and q.brightness == 255


def test_sim_camera_renders_visible_target_with_ground_truth():
    world = World(WorldConfig(targets=[[8.0, 0.0, 2.0, 0.4, "red_marker"]]))
    cam = SimCamera(CameraConfig(), world, np.random.default_rng(0))
    frame, gt = cam.render(np.array([0.0, 0.0, 2.0]), 0.0)
    assert frame.shape == (240, 320, 3)
    assert len(gt) == 1
    det = FrameAnalyzer(VisionConfig()).analyze(frame).detections
    assert any(d.label == "red_marker" and iou(d.bbox, gt[0].bbox) > 0.3 for d in det)


def test_sim_camera_occlusion():
    world = World(WorldConfig(cylinders=[[4.0, 0.0, 1.0, 6.0]], targets=[[8.0, 0.0, 2.0, 0.4, "red_marker"]]))
    cam = SimCamera(CameraConfig(), world, np.random.default_rng(0))
    _, gt = cam.render(np.array([0.0, 0.0, 2.0]), 0.0)
    assert gt == []
