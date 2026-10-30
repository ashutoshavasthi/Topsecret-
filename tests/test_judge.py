from skymind.evaluation.judge import FlightJudge, linear_score


def _state(t, **kw):
    s = {"type": "state", "t": t, "mode": "mission", "pos": [t, 0, 2.0], "vel": [1, 0, 0], "rpy": [0.0, 0.0, 0.0],
         "est_rpy": [0.0, 0.0, 0.0], "rate": [0.0, 0.0, 0.0], "alt_sp": 2.0, "vel_sp": [1, 0, 0],
         "min_range": 5.0, "clearance": 2.0, "wp": 1}
    s.update(kw)
    return s


MISSION = {"status": "landed", "failsafe": None, "waypoints_reached": 1, "waypoints_total": 1,
           "distance_flown": 10.0, "straight_line_distance": 10.0, "collisions": 0}


def test_linear_score():
    assert linear_score(0.1, 0.1, 1.0) == 100
    assert linear_score(1.0, 0.1, 1.0) == 0
    assert 40 < linear_score(0.5, 0.1, 1.0) < 60
    assert linear_score(0.9, 0.9, 0.2) == 100  # higher-is-better metrics work too


def test_perfect_flight_scores_high():
    card = FlightJudge().judge([_state(t * 0.1) for t in range(100)], MISSION)
    assert card.verdict == "PASS"
    assert card.overall > 90 and card.grade == "A"


def test_wobbly_flight_scores_lower():
    smooth = FlightJudge().judge([_state(t * 0.1) for t in range(100)], MISSION)
    wobbly = FlightJudge().judge(
        [_state(t * 0.1, rate=[1.0 * (-1) ** t, 0.5, 0.0], rpy=[0.5 * (-1) ** t, 0.2, 0.0], pos=[t * 0.1, 0, 2.6]) for t in range(100)],
        MISSION,
    )
    assert wobbly.categories[0].score < smooth.categories[0].score - 30


def test_collision_caps_score():
    card = FlightJudge().judge([_state(t * 0.1) for t in range(100)], {**MISSION, "collisions": 1})
    assert card.overall <= 50 and card.verdict.startswith("FAIL")


def test_vision_recall_and_precision():
    gt = [{"label": "red_marker", "bbox": [10, 10, 20, 20], "distance": 5.0}]
    frames = [
        {"type": "vision", "t": 0.0, "detections": [{"label": "red_marker", "confidence": 0.9, "bbox": [11, 10, 20, 20]}], "quality": None, "ground_truth": gt},
        {"type": "vision", "t": 0.1, "detections": [], "quality": None, "ground_truth": gt},
        {"type": "vision", "t": 0.2, "detections": [{"label": "red_marker", "confidence": 0.9, "bbox": [200, 10, 20, 20]}], "quality": None, "ground_truth": []},
    ]
    vision = FlightJudge().vision(frames)
    m = {x.name: x.value for x in vision.metrics}
    assert m["detection_recall"] == 0.5
    assert m["detection_precision"] == 0.5
