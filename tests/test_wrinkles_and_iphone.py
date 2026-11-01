"""Unit tests for Dynamic Wrinkle Solver, iPhone TrueDepth Receiver, and Client Detector."""
import json
import time
import numpy as np
import pytest

from kinetra.core.constants import BLENDSHAPE_NAMES
from kinetra.core.wrinkle_solver import WrinkleSolver
from kinetra.core.client_detector import ClientDetector
from kinetra.capture.iphone_receiver import IPhoneTrueDepthReceiver
from kinetra.core.multicam import Rig


def test_wrinkle_solver_forehead_and_frown():
    solver = WrinkleSolver(smoothing=0.0)

    # Neutral face
    neutral_bs = np.zeros(52, dtype=np.float32)
    w_neutral = solver.solve(neutral_bs)
    assert w_neutral["forehead_horizontal"] == 0.0
    assert w_neutral["glabellar_frown"] == 0.0

    # Eyebrow elevation -> Forehead tension
    bs_surprised = np.zeros(52, dtype=np.float32)
    bs_surprised[BLENDSHAPE_NAMES.index("browInnerUp")] = 0.85
    bs_surprised[BLENDSHAPE_NAMES.index("browOuterUpLeft")] = 0.70
    w_surprised = solver.solve(bs_surprised)
    assert w_surprised["forehead_horizontal"] > 0.65
    assert w_surprised["glabellar_frown"] == 0.0

    # Eyebrow furrow -> Glabellar frown tension
    bs_angry = np.zeros(52, dtype=np.float32)
    bs_angry[BLENDSHAPE_NAMES.index("browDownLeft")] = 0.90
    bs_angry[BLENDSHAPE_NAMES.index("browDownRight")] = 0.90
    w_angry = solver.solve(bs_angry)
    assert w_angry["glabellar_frown"] > 0.70

    # MetaHuman curve mapping
    mh_curves = solver.get_metahuman_curves(w_angry)
    assert "CTRL_Wrinkle_Glabella" in mh_curves
    assert mh_curves["CTRL_Wrinkle_Glabella"] == w_angry["glabellar_frown"]


def test_wrinkle_solver_smile_and_crows_feet():
    solver = WrinkleSolver(smoothing=0.0)

    # Big smile with eye squint
    bs_smile = np.zeros(52, dtype=np.float32)
    bs_smile[BLENDSHAPE_NAMES.index("mouthSmileLeft")] = 0.95
    bs_smile[BLENDSHAPE_NAMES.index("mouthSmileRight")] = 0.95
    bs_smile[BLENDSHAPE_NAMES.index("eyeSquintLeft")] = 0.80
    bs_smile[BLENDSHAPE_NAMES.index("cheekSquintLeft")] = 0.60

    w = solver.solve(bs_smile)
    assert w["nasolabial_left"] > 0.50
    assert w["nasolabial_right"] > 0.50
    assert w["crows_feet_left"] > 0.60
    assert w["crows_feet_right"] == 0.0  # right squint was 0


def test_iphone_receiver_json_packet():
    receiver = IPhoneTrueDepthReceiver(port=54321)

    payload = {
        "device_name": "iPhone 16 Pro",
        "blendshapes": {
            "eyeBlinkLeft": 0.88,
            "jawOpen": 0.54,
            "mouthSmileLeft": 0.72
        },
        "head_rotation": [0.98, 0.02, 0.15, -0.05]
    }
    raw_bytes = json.dumps(payload).encode("utf-8")
    receiver._parse_packet(raw_bytes, ("192.168.1.42", 50000))

    bs, head_q, _ = receiver.get_latest()
    assert receiver.device_name == "iPhone 16 Pro"
    assert bs[BLENDSHAPE_NAMES.index("eyeBlinkLeft")] == pytest.approx(0.88, abs=0.01)
    assert bs[BLENDSHAPE_NAMES.index("jawOpen")] == pytest.approx(0.54, abs=0.01)
    assert bs[BLENDSHAPE_NAMES.index("mouthSmileLeft")] == pytest.approx(0.72, abs=0.01)
    assert head_q[0] == pytest.approx(0.98, abs=0.01)


def test_iphone_receiver_ifacialmocap_packet():
    receiver = IPhoneTrueDepthReceiver(port=54322)

    # Pipe-separated format
    packet_str = "eyeBlinkRight-0.92|mouthPucker-0.65|browInnerUp-0.75"
    receiver._parse_packet(packet_str.encode("utf-8"), ("192.168.1.55", 50000))

    bs, _, _ = receiver.get_latest()
    assert bs[BLENDSHAPE_NAMES.index("eyeBlinkRight")] == pytest.approx(0.92, abs=0.01)
    assert bs[BLENDSHAPE_NAMES.index("mouthPucker")] == pytest.approx(0.65, abs=0.01)
    assert bs[BLENDSHAPE_NAMES.index("browInnerUp")] == pytest.approx(0.75, abs=0.01)


def test_client_detector():
    detector = ClientDetector(cache_ttl=0.1)
    status = detector.check(force=True)
    assert "ue5_running" in status
    assert "blender_running" in status
    assert isinstance(status["ue5_running"], bool)
    assert isinstance(status["blender_running"], bool)


def test_rig_default_stereo():
    rig = Rig.default_stereo(size=(1920, 1080), baseline_m=1.2)
    assert len(rig.cameras) == 2
    assert rig.cameras[0].size == (1920, 1080)
    assert rig.cameras[1].size == (1920, 1080)
    # Check cameras are offset horizontally
    assert rig.cameras[0].t[0] < 0.0
    assert rig.cameras[1].t[0] > 0.0


def test_rig_default_array_n_cameras():
    """Rig.default_array creates the correct number of cameras arranged in a semicircle."""
    for n in (2, 3, 5, 8, 20):
        rig = Rig.default_array(n, size=(1280, 720), radius_m=3.0)
        assert len(rig.cameras) == n, f"Expected {n} cameras, got {len(rig.cameras)}"
        for i, cam in enumerate(rig.cameras):
            assert cam.size == (1280, 720)
            assert cam.name == f"CAM_{i}"
        # First and last cameras should be on opposite sides
        if n > 1:
            assert rig.cameras[0].t[0] != rig.cameras[-1].t[0]
