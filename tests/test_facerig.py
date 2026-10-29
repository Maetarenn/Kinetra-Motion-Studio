import tempfile
from pathlib import Path
import pytest
import numpy as np
from kinetra.core.constants import BS_INDEX, NBS
from kinetra.core.facerig import FaceEnrollment, FaceRig
from kinetra.core.exporters import write_points_usd

RNG = np.random.default_rng(0)
L0 = RNG.normal(0, 0.5, (478, 3))
REGIONS = {"lips": list(range(0, 40)), "eyes": list(range(40, 80)), "brows": list(range(80, 100)), "nose": list(range(100, 130))}
TRIS = [(i, i + 1, i + 2) for i in range(0, 120, 3)]
EXPR_BS = {"mouth_open": {"jawOpen": 0.9}, "smile": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8},
           "pucker": {"mouthPucker": 0.8}, "brows_up": {"browInnerUp": 0.8},
           "brows_down": {"browDownLeft": 0.8, "browDownRight": 0.8}, "eyes_closed": {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9},
           "cheeks_puff": {"cheekPuff": 0.7}}
DELTA = {}
for n, k in enumerate(EXPR_BS):
    d = np.zeros((478, 3))
    d[10 * n:10 * n + 10 + 20] = RNG.normal(0, 0.08, (30, 3))          # each expression moves its own vertices
    DELTA[k] = d


def bsvec(d):
    v = np.zeros(NBS)
    for k, x in d.items():
        v[BS_INDEX[k]] = x
    return v


SCRIPT = [("center", 0, 0, {}), ("turn_a", 30, 0, {}), ("turn_b", -30, 0, {}), ("up", 0, 18, {}), ("down", 0, -18, {})] + \
         [(k, 0, 0, v) for k, v in EXPR_BS.items()]


def run_script(enr, script=SCRIPT, frames=40, in_circle=True):
    for key, yaw, pitch, bs in script:
        L = L0 + DELTA.get(key, 0.0)
        for _ in range(frames):
            enr.update(L, bsvec(bs), yaw, pitch, in_circle)


def test_enrollment_walks_through_every_step():
    enr = FaceEnrollment()
    run_script(enr)
    assert enr.done and len(enr.captures) == 12
    assert enr.captures["mouth_open"]["bs"][BS_INDEX["jawOpen"]] > 0.8


def test_wrong_direction_and_outside_circle_do_not_advance():
    enr = FaceEnrollment()
    run_script(enr, SCRIPT[:2])                                  # center, turn_a (positive yaw)
    assert "turn_a" in enr.captures
    run_script(enr, [("turn_b", 30, 0, {})])                      # turns the SAME way again: must not count
    assert "turn_b" not in enr.captures
    e2 = FaceEnrollment()
    run_script(e2, SCRIPT[:1], in_circle=False)
    assert e2.idx == 0                                           # not inside the guide circle -> nothing captured


def test_rig_solves_personal_expression_weights():
    enr = FaceEnrollment()
    run_script(enr)
    rig = FaceRig.from_captures(enr.captures, REGIONS, TRIS)
    live = L0 + 0.7 * DELTA["smile"] + 0.3 * DELTA["mouth_open"] + RNG.normal(0, 0.003, L0.shape)
    w = dict(zip(rig.keys, rig.solve(live)))
    assert abs(w["smile"] - 0.7) < 0.12 and abs(w["mouth_open"] - 0.3) < 0.12
    assert max(v for k, v in w.items() if k not in ("smile", "mouth_open")) < 0.12
    assert rig.gain.max() <= 2.5 and rig.gain.min() >= 0.8


def test_rig_roundtrip_and_usd_point_cache():
    enr = FaceEnrollment()
    run_script(enr)
    rig = FaceRig.from_captures(enr.captures, REGIONS, TRIS)
    p = Path(tempfile.mkdtemp()) / "rig.json"
    rig.save(p)
    r2 = FaceRig.load(p)
    assert np.allclose(r2.identity, rig.identity, atol=1e-4) and r2.keys == rig.keys
    n = 12
    pts = np.tile(rig.neutral_points()[None], (n, 1, 1))
    idx = {int(v): i for i, v in enumerate(rig.feat)}
    tris = [[idx[a] for a in t] for t in rig.tris if all(int(a) in idx for a in t)]
    out = Path(tempfile.mkdtemp()) / "face.usda"
    write_points_usd(out, np.arange(n) / 30.0, pts, tris)
    assert out.exists() and out.stat().st_size > 0
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(out))
    mesh = UsdGeom.Mesh(st.GetPrimAtPath("/FaceFeatures"))
    assert len(mesh.GetPointsAttr().Get(5)) == len(rig.feat) and st.GetEndTimeCode() == n


def test_features_only_scene_builds_items_without_opengl():
    from kinetra.ui.face_view import FaceScene
    enr = FaceEnrollment()
    run_script(enr)
    rig = FaceRig.from_captures(enr.captures, REGIONS, TRIS)
    sc = FaceScene()
    sc.rig = rig
    tk = {"L": L0, "neutral_L": L0, "S": np.eye(3), "slot": 0, "heat_W": None, "muscles": None, "bs": np.zeros(NBS)}
    items = sc.items([tk], 3)
    assert len(items) == 5                                         # feature mesh + two eyeballs + two irises
    for it in items:
        assert it["pos"].shape[1] == 3 and it["nrm"].shape == it["pos"].shape and it["idx"].max() < len(it["pos"])


def _center_then(enr, step_key, L, bs, yaw=0, pitch=0, frames=40):
    for _ in range(frames):
        enr.update(L0, bsvec({}), 0, 0, True) if "center" not in enr.captures else None
    for _ in range(frames):
        enr.update(L, bsvec(bs), yaw, pitch, True)


def _run_to(enr, key):
    for k, yaw, pitch, bs in SCRIPT:
        if k == key:
            return
        for _ in range(40):
            enr.update(L0 + DELTA.get(k, 0.0), bsvec(bs), yaw, pitch, True)


def test_weak_real_world_frown_is_detected_through_blendshapes():
    enr = FaceEnrollment()
    _run_to(enr, "brows_down")
    assert enr.steps[enr.idx].key == "brows_down"
    for _ in range(40):
        enr.update(L0, bsvec({"browDownLeft": 0.2, "browDownRight": 0.18}), 0, 0, True)     # a typical weak MediaPipe frown
    assert "brows_down" in enr.captures


def test_frown_is_detected_from_landmarks_even_if_blendshapes_stay_zero():
    enr = FaceEnrollment()
    _run_to(enr, "brows_down")
    L = L0.copy()
    mid = (L[107] + L[336]) / 2
    L[107] = mid + (L[107] - mid) * 0.8                      # inner brow ends move toward each other
    L[336] = mid + (L[336] - mid) * 0.8
    for _ in range(40):
        enr.update(L, bsvec({}), 0, 0, True)
    assert "brows_down" in enr.captures


def test_skip_and_capture_now_let_the_actor_get_past_a_stubborn_step():
    enr = FaceEnrollment()
    _run_to(enr, "cheeks_puff")
    enr.skip()
    assert enr.done and "cheeks_puff" not in enr.captures
    rig = FaceRig.from_captures(enr.captures, REGIONS, TRIS)
    assert "cheeks_puff" not in rig.keys and len(rig.keys) == 10
    e2 = FaceEnrollment()
    e2.skip()
    assert e2.idx == 0                                        # the neutral face cannot be skipped
    _run_to(e2, "pucker")
    for _ in range(12):
        e2.update(L0, bsvec({}), 0, 0, True)                   # no pucker at all
    e2.capture_now()
    assert "pucker" in e2.captures
