"""MediaPipe model files (downloaded once into the user's data folder) and landmarker factories."""
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = Path.home() / ".kinetra" / "models"
POSE_VARIANTS = ("lite", "full", "heavy")
_BASE = "https://storage.googleapis.com/mediapipe-models/"
FACE_URL = _BASE + "face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def pose_url(variant):
    return f"{_BASE}pose_landmarker/pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"


def ensure_model(url, filename, log=print):
    path = MODEL_DIR / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"DOWNLOADING {filename} (first start only) ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def create_pose(variant="heavy", num_poses=1, log=print):
    path = str(ensure_model(pose_url(variant), f"pose_landmarker_{variant}.task", log))
    return mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path), running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_segmentation_masks=False))


def create_face(mode=mp_vision.RunningMode.VIDEO, num_faces=1, prefer_gpu=False, log=print):
    path = str(ensure_model(FACE_URL, "face_landmarker.task", log))
    last = None
    for name in (("GPU",) if prefer_gpu else ()) + ("CPU",):
        try:
            base = mp_python.BaseOptions(model_asset_path=path, delegate=getattr(mp_python.BaseOptions.Delegate, name))
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base, running_mode=mode, num_faces=num_faces, min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5, min_tracking_confidence=0.5, output_face_blendshapes=True,
                output_facial_transformation_matrixes=False)
            return mp_vision.FaceLandmarker.create_from_options(opts), name
        except Exception as exc:          # GPU delegate is not available on every platform
            last = exc
    raise RuntimeError(f"FaceLandmarker could not start: {last}")


HAND_URL = _BASE + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def create_hands(num_hands=2, log=print):
    path = str(ensure_model(HAND_URL, "hand_landmarker.task", log))
    return mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path), running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands, min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4))


def face_connections(name):
    """Edge list of a FaceMesh region: Tasks API first, legacy solutions as fallback."""
    try:
        return [(c.start, c.end) for c in getattr(mp_vision.FaceLandmarksConnections, f"FACE_LANDMARKS_{name}")]
    except Exception:
        try:
            return list(getattr(mp.solutions.face_mesh, f"FACEMESH_{name}"))
        except Exception:
            return []
