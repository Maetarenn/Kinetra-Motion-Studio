import time, threading
import numpy as np
from PySide6 import QtCore
from kinetra.capture.pipeline import CaptureWorker, CaptureConfig
from kinetra.core.constants import *
from kinetra.core.multicam import Rig
from test_body_solver import synth_pose
from test_multicam import true_cams, SIZE


class SynthSource:
    is_file = False

    def __init__(self, n=200, fps=60):
        self.info, self.fps, self.ended, self.produced = "synthetic", fps, False, 0
        self._n, self._i, self.running = n, 0, False

    def open(self): return True
    def start(self): self.running = True
    def stop(self): self.running = False

    def get(self, timeout=0.25):
        time.sleep(1 / 60)
        if self._i >= self._n:
            self.ended = True
            return None
        self._i += 1
        self.produced += 1
        return np.zeros((SIZE[1], SIZE[0], 3), np.uint8), time.perf_counter()


class L:  # mediapipe-like landmark
    def __init__(self, x, y, z=0.0, v=1.0): self.x, self.y, self.z, self.visibility = x, y, z, v


class Res:
    def __init__(self, img, wld=None, faces=None):
        self.pose_landmarks = [img] if img else []
        self.pose_world_landmarks = [wld] if wld else []
        self.face_landmarks = faces or []
        self.face_blendshapes = []


class FakeModels:
    def __init__(self, truth_fn, cams=None):
        self.truth_fn, self.cams, self.count = truth_fn, cams, 0

    def create_pose(self, variant, log=print):
        outer, i = self, self.count
        self.count += 1

        class P:
            def detect_for_video(self, image, ts):
                X = outer.truth_fn()
                if outer.cams:
                    px = outer.cams[i].project(X) / np.array(SIZE, float)
                    return Res([L(a, b) for a, b in px])
                img = [L(0.5 + x * 0.3, 0.5 - y * 0.4) for x, y, _ in X]
                wld = [L(x, -y, -z) for x, y, z in X - X[[23, 24]].mean(0)]
                return Res(img, wld)

            def close(self): pass
        return P()

    def create_hands(self, n=2, log=print):
        outer = self

        class H:
            def detect_for_video(self, image, ts):
                r = Res(None)
                r.hand_landmarks, r.hand_world_landmarks, r.handedness = [], [], []
                return r

            def close(self): pass
        return H()

    def create_face(self, num_faces=1, prefer_gpu=False, log=print):
        class F:
            def detect_for_video(self, image, ts):
                return Res(None, None, [])
            def close(self): pass
        return F(), "CPU"


def run_worker(cfg, models, n_frames=60, timeout=20):
    w = CaptureWorker(cfg, models=models)
    got, msgs = [], []
    w.frame_ready.connect(lambda r: got.append(r), QtCore.Qt.DirectConnection)
    w.message.connect(lambda k, a: msgs.append((k, a)), QtCore.Qt.DirectConnection)
    w.start()
    t0 = time.time()
    while len(got) < n_frames and time.time() - t0 < timeout and w.isRunning():
        time.sleep(0.02)
    w.stop()
    return got, msgs


def test_single_camera_pipeline():
    truth = synth_pose() + np.array([0, 0.97, 0])
    cfg = CaptureConfig(sources=[SynthSource()], mode="single", track_face=True)
    got, msgs = run_worker(cfg, FakeModels(lambda: truth))
    assert len(got) >= 40, msgs
    r = got[-1]
    assert r["joints"] is not None and r["quats"] is not None and len(r["quats"]) == 13
    assert r["faces"] == [] and r["bs"].shape == (52,)
    assert len(r["quats_full"]) == 52 and r["fk"][0].shape == (52, 3)
    assert not r["calibrating"]


def test_multi_camera_pipeline_triangulates_to_ground_truth():
    cams = true_cams()
    truth = synth_pose() + np.array([0.0, 0.97, 0.0])
    cfg = CaptureConfig(sources=[SynthSource() for _ in cams], mode="multi", rig=Rig(cams), track_face=False)
    got, msgs = run_worker(cfg, FakeModels(lambda: truth, cams))
    assert len(got) >= 40, msgs
    r = got[-1]
    assert r["tri_used"] > 2.5
    J = r["joints"]
    assert np.abs(J[ARMS["L"][2]] - truth[ARMS["L"][2]]).max() < 0.05      # wrist, metres
    assert abs(J[:, 1].min()) < 0.1                                         # absolute, standing on the floor


def test_recording_roundtrip():
    truth = synth_pose() + np.array([0, 0.97, 0])
    cfg = CaptureConfig(sources=[SynthSource(n=150)], mode="single", track_face=False)
    w = CaptureWorker(cfg, models=FakeModels(lambda: truth))
    got = []
    w.frame_ready.connect(lambda r: got.append(r), QtCore.Qt.DirectConnection)
    w.start()
    while len(got) < 10:
        time.sleep(0.02)
    w.recorder.start()
    while len(got) < 60:
        time.sleep(0.02)
    data = w.recorder.stop()
    w.stop()
    assert len(data["t"]) > 20 and data["raw"].shape[1:] == (33, 3) and data["quats"].shape[1:] == (52, 4)


def test_hand_tracking_reaches_the_full_skeleton():
    from kinetra.core.hand import hand_fk, HandSolver, FINGERS
    from kinetra.core.mathx import q_axis_angle
    truth = synth_pose() + np.array([0, 0.97, 0])
    off = HandSolver().rest_offsets()
    fist = {}
    for f in FINGERS:
        fist[f + "_1"], fist[f + "_2"], fist[f + "_3"] = (q_axis_angle([0, 0, 1], np.radians(a)) for a in (80, 90, 60))
    Pl = hand_fk(fist, off)                                        # scene axes, wrist at origin, palm frame = world

    class HandModels(FakeModels):
        def create_hands(self, n=2, log=print):
            class H:
                def detect_for_video(self, image, ts):
                    r = Res(None)
                    X = truth
                    wrist_img = [0.5 + X[15, 0] * 0.3, 0.5 - X[15, 1] * 0.4, 0.0]
                    img = np.tile(wrist_img, (21, 1))
                    r.hand_landmarks = [[L(*p) for p in img]]
                    r.hand_world_landmarks = [[L(*(p * np.array([1, -1, -1]))) for p in Pl]]
                    class C: category_name, score = "Right", 0.9
                    r.handedness = [[C()]]
                    return r

                def close(self): pass
            return H()

    cfg = CaptureConfig(sources=[SynthSource(n=260)], mode="single", track_face=False, use_ai=False)
    got, msgs = run_worker(cfg, HandModels(lambda: truth), n_frames=180, timeout=30)
    r = got[-1]
    assert "L" in r["hands"] and "R" not in r["hands"], r["hands"].keys()
    assert min(v for k, v in r["hands"]["L"]["curl"].items() if k != "thumb") > 0.5
    pos, rot = r["fk"]
    order = list(r["skel"].order)
    tip_mid = pos[order.index("index_3_L")] - pos[order.index("hand_L")]
    assert np.linalg.norm(tip_mid - Pl[7]) < 0.01, (tip_mid, Pl[7])


def test_hidden_hand_keeps_a_probable_pose_instead_of_vanishing():
    from kinetra.core.hand import hand_fk, HandSolver, FINGERS
    from kinetra.core.mathx import q_axis_angle
    truth = synth_pose() + np.array([0, 0.97, 0])
    off = HandSolver().rest_offsets()
    fist = {}
    for f in FINGERS:
        fist[f + "_1"], fist[f + "_2"], fist[f + "_3"] = (q_axis_angle([0, 0, 1], np.radians(a)) for a in (80, 90, 60))
    Pl = hand_fk(fist, off)
    calls = {"n": 0}

    class HandModels(FakeModels):
        def create_hands(self, n=2, log=print):
            class H:
                def detect_for_video(self, image, ts):
                    calls["n"] += 1
                    r = Res(None)
                    r.hand_landmarks, r.hand_world_landmarks, r.handedness = [], [], []
                    if calls["n"] < 70:                                  # hand visible, then it leaves the picture
                        X = truth
                        img = np.tile([0.5 + X[15, 0] * 0.3, 0.5 - X[15, 1] * 0.4, 0.0], (21, 1))
                        r.hand_landmarks = [[L(*p) for p in img]]
                        r.hand_world_landmarks = [[L(*(p * np.array([1, -1, -1]))) for p in Pl]]
                        class C: category_name, score = "Right", 0.9
                        r.handedness = [[C()]]
                    return r

                def close(self): pass
            return H()

    cfg = CaptureConfig(sources=[SynthSource(n=300)], mode="single", track_face=False)
    got, msgs = run_worker(cfg, HandModels(lambda: truth), n_frames=200, timeout=40)
    r = got[-1]
    assert "L" in r["hands"] and r["hands"]["L"]["ai"]["missing"] > 0.3, r["hands"].keys()
    q = r["quats_full"]["index_2_L"]
    assert np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-6
    assert r["hands"]["L"]["ai"]["confidence"] > 0.0


def test_scanned_face_rig_flows_into_take_files(tmp_path):
    from kinetra.core.facerig import FaceEnrollment, FaceRig
    from kinetra.core.project import Project, save_take
    from test_facerig import run_script, REGIONS, TRIS
    enr = FaceEnrollment()
    run_script(enr)
    rig = FaceRig.from_captures(enr.captures, REGIONS, TRIS)
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.08, (478, 3))
    base[33], base[263], base[133], base[362], base[10], base[152] = [0.42, 0.45, 0], [0.58, 0.45, 0], [0.47, 0.45, 0], [0.53, 0.45, 0], [0.5, 0.3, 0], [0.5, 0.7, 0]

    class FaceModels(FakeModels):
        def create_face(self, num_faces=1, prefer_gpu=False, log=print):
            class F:
                def detect_for_video(self, image, ts):
                    r = Res(None)
                    r.face_landmarks = [[L(*p) for p in base]]
                    class Cat:
                        def __init__(self, n): self.category_name, self.score = n, 0.1
                    from kinetra.core.constants import BLENDSHAPE_NAMES
                    r.face_blendshapes = [[Cat(n) for n in BLENDSHAPE_NAMES]]
                    return r
                def close(self): pass
            return F(), "CPU"

    truth = synth_pose() + np.array([0, 0.97, 0])
    cfg = CaptureConfig(sources=[SynthSource(n=260)], mode="single", track_face=True, face_rig=rig)
    w = CaptureWorker(cfg, models=FaceModels(lambda: truth))
    got = []
    w.frame_ready.connect(lambda r: got.append(r), QtCore.Qt.DirectConnection)
    w.start()
    while len(got) < 20:
        time.sleep(0.02)
    w.recorder.start()
    while len(got) < 90:
        time.sleep(0.02)
    data = w.recorder.stop()
    w.stop()
    assert data["face_L"].shape[1:] == (len(rig.feat), 3) and "personal" in got[-1]["faces"][0]
    proj = Project.create(tmp_path / "p", "P")
    d = save_take(proj, data, face_rig=rig)
    assert (d / "face_points.usda").exists() and (d / "face_personal.csv").exists()
