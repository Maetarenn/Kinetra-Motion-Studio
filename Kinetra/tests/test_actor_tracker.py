"""Unit tests for MultiActorTracker and Live Person Re-Identification."""
import numpy as np
import pytest

from kinetra.ai.actor_tracker import MultiActorTracker, color_name_from_rgb, ActorProfile
from kinetra.capture.pipeline import CaptureConfig, CaptureWorker


def test_color_name_from_rgb():
    assert color_name_from_rgb(20, 20, 20) == "Siyah"
    assert color_name_from_rgb(230, 230, 230) == "Beyaz"
    assert color_name_from_rgb(120, 120, 120) == "Gri"
    assert color_name_from_rgb(220, 20, 20) == "Kırmızı"
    assert color_name_from_rgb(20, 50, 220) == "Mavi"
    assert color_name_from_rgb(20, 200, 30) == "Yeşil"
    assert color_name_from_rgb(240, 220, 30) == "Sarı"
    assert color_name_from_rgb(240, 120, 20) == "Turuncu"
    assert color_name_from_rgb(180, 40, 180) == "Mor"


def _make_dummy_body(x_center, y_center, sho_w=0.08, torso_h=0.30):
    """Creates a 33x4 normalized landmark array centered at (x_center, y_center)."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0  # high visibility

    # 11: left shoulder, 12: right shoulder
    lm[11, 0] = x_center + sho_w / 2.0
    lm[11, 1] = y_center - torso_h / 2.0
    lm[12, 0] = x_center - sho_w / 2.0
    lm[12, 1] = y_center - torso_h / 2.0

    # 23: left hip, 24: right hip
    lm[23, 0] = x_center + sho_w / 2.5
    lm[23, 1] = y_center + torso_h / 2.0
    lm[24, 0] = x_center - sho_w / 2.5
    lm[24, 1] = y_center + torso_h / 2.0

    # 3D world joints mock
    world = np.zeros((33, 4), dtype=np.float32)
    world[:, :3] = np.random.randn(33, 3) * 0.1
    world[:, 0] += (x_center - 0.5) * 2.0
    world[:, 3] = 1.0

    return {"image": lm, "world": world}


def _paint_actor(frame, body, color_rgb):
    """Fills the torso region of body in frame with color_rgb."""
    h, w = frame.shape[:2]
    img = body["image"]
    x1 = int(min(img[11, 0], img[12, 0], img[23, 0], img[24, 0]) * w)
    x2 = int(max(img[11, 0], img[12, 0], img[23, 0], img[24, 0]) * w)
    y1 = int(min(img[11, 1], img[12, 1]) * h)
    y2 = int(max(img[23, 1], img[24, 1]) * h)
    frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = color_rgb


def test_extract_appearance():
    tracker = MultiActorTracker(max_actors=2)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    body = _make_dummy_body(0.5, 0.45)
    _paint_actor(frame, body, (220, 20, 20))  # Bright Red in RGB

    app = tracker.extract_appearance(frame, body["image"])
    assert app is not None
    assert "hist" in app
    assert "hex" in app
    assert app["color_name"] == "Kırmızı"
    assert app["sho_ratio"] > 0.0


def test_multi_actor_no_swap_on_crossover():
    """Validates that two actors crossing paths do NOT swap identities due to clothing ReID."""
    tracker = MultiActorTracker(max_actors=2)

    # Frame 0: Actor 1 on left (Red shirt, x=0.20), Actor 2 on right (Blue shirt, x=0.80)
    f0 = np.full((480, 640, 3), 100, dtype=np.uint8)
    b1_f0 = _make_dummy_body(0.20, 0.45, sho_w=0.08)
    b2_f0 = _make_dummy_body(0.80, 0.45, sho_w=0.08)
    _paint_actor(f0, b1_f0, (220, 20, 20))   # Red
    _paint_actor(f0, b2_f0, (20, 40, 220))   # Blue

    actors_0 = tracker.match_and_update(f0, [b1_f0, b2_f0], t_now=0.0)
    assert len(actors_0) == 2
    # Actor 1 should be Red, Actor 2 should be Blue
    assert actors_0[0].actor_id == 1 and actors_0[0].color_name == "Kırmızı"
    assert actors_0[1].actor_id == 2 and actors_0[1].color_name == "Mavi"

    # Frame 1: Crossover point! Both actors pass each other:
    # Actor 1 has moved to x=0.55 (moving right)
    # Actor 2 has moved to x=0.45 (moving left)
    f1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    b_red_mid = _make_dummy_body(0.55, 0.45, sho_w=0.08)
    b_blue_mid = _make_dummy_body(0.45, 0.45, sho_w=0.08)
    _paint_actor(f1, b_red_mid, (220, 20, 20))
    _paint_actor(f1, b_blue_mid, (20, 40, 220))

    # Pass in reversed order in detection list to strictly test ReID matching
    actors_1 = tracker.match_and_update(f1, [b_blue_mid, b_red_mid], t_now=0.5)
    assert len(actors_1) == 2

    # Verify ID persistence: Actor 1 must remain Red, Actor 2 must remain Blue!
    p1 = tracker.profiles[1]
    p2 = tracker.profiles[2]
    assert p1.color_name == "Kırmızı", f"Actor 1 swapped to {p1.color_name}!"
    assert p2.color_name == "Mavi", f"Actor 2 swapped to {p2.color_name}!"

    # Frame 2: Fully crossed!
    # Red actor is now on the right (x=0.80)
    # Blue actor is now on the left (x=0.20)
    f2 = np.full((480, 640, 3), 100, dtype=np.uint8)
    b_red_right = _make_dummy_body(0.80, 0.45, sho_w=0.08)
    b_blue_left = _make_dummy_body(0.20, 0.45, sho_w=0.08)
    _paint_actor(f2, b_blue_left, (20, 40, 220))
    _paint_actor(f2, b_red_right, (220, 20, 20))

    actors_2 = tracker.match_and_update(f2, [b_blue_left, b_red_right], t_now=1.0)
    assert tracker.profiles[1].color_name == "Kırmızı"
    assert tracker.profiles[2].color_name == "Mavi"
    # Position tracking verification
    assert tracker.profiles[1].last_center[0] > 0.6  # Actor 1 is on the right now
    assert tracker.profiles[2].last_center[0] < 0.4  # Actor 2 is on the left now


def test_pipeline_multi_actor_config():
    cfg = CaptureConfig(multi_actor=True)
    assert cfg.multi_actor is True

    worker = CaptureWorker(cfg)
    assert hasattr(worker, "actor_tracker")
    assert worker.actor_tracker is not None

    worker.set_multi_actor(False)
    assert worker.cfg.multi_actor is False
    assert worker.cfg.num_poses == 1

    worker.set_multi_actor(True)
    assert worker.cfg.multi_actor is True
    assert worker.cfg.num_poses == 2
