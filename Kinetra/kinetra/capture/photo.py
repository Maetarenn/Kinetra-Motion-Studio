"""Photo analysis: one or many face photos -> proportions, symmetry, expression bias, identity mesh, annotated PNG."""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

from ..core.constants import BLENDSHAPE_NAMES, BS_INDEX, NBS
from ..core.face import to_face_frame, MuscleModel, MUSCLE_NAMES
from .face_tracks import blend_vector
from .models import create_face, face_connections

SYM_PAIRS = ((33, 263), (133, 362), (61, 291), (129, 358), (234, 454), (172, 397), (70, 300),
             (107, 336), (205, 425), (159, 386), (145, 374))


def analyze_photo(path, models=None):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image could not be read")
    h, w = img.shape[:2]
    lmk, _ = (models or create_face)(mp_vision.RunningMode.IMAGE, 1, False)
    try:
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    finally:
        lmk.close()
    if not res.face_landmarks:
        raise ValueError("no face detected")
    pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])
    P = np.column_stack([pts[:, 0] * w, pts[:, 1] * h, pts[:, 2] * w])
    L, R, o, s = to_face_frame(P)
    bs = blend_vector(res.face_blendshapes[0])
    d = lambda a, b: float(np.linalg.norm(L[a] - L[b]))
    asym = np.mean([abs(abs(L[a, 0]) - abs(L[b, 0])) + abs(L[a, 1] - L[b, 1]) for a, b in SYM_PAIRS])
    M = R.T
    yaw = math.degrees(math.atan2(M[0, 2], M[2, 2]))
    pitch = math.degrees(math.atan2(-M[1, 2], math.hypot(M[0, 2], M[2, 2])))
    roll = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    metrics = {"face_height": d(10, 152), "face_width": d(234, 454), "jaw_width": d(172, 397),
               "mouth_width": d(61, 291), "nose_width": d(129, 358), "inner_eye_gap": d(133, 362),
               "brow_to_eye": float(L[133, 1] - L[107, 1]),
               "symmetry_pct": float(np.clip(100 * (1 - asym / 0.12), 0, 100)),
               "yaw": yaw, "pitch": pitch, "roll": roll, "face_px": float(s)}
    warn = []
    if max(abs(yaw), abs(pitch)) > 15:
        warn.append("head not frontal (yaw/pitch > 15 deg)")
    if bs[BS_INDEX["eyeBlinkLeft"]] > .5 or bs[BS_INDEX["eyeBlinkRight"]] > .5:
        warn.append("eyes closed")
    if s < 0.06 * w:
        warn.append("face too small in frame")
    quiet = [i for i, n in enumerate(BLENDSHAPE_NAMES) if not n.startswith(("eyeLook", "eyeBlink"))]
    return {"path": str(path), "img": img, "P": P, "L": L, "bs": bs, "metrics": metrics, "warn": warn,
            "neutral_like": float(bs[quiet].max()) < 0.25}


def build_profile(paths, out_dir, log=print):
    """Analyse several photos in parallel and merge them into one profile (saved as JSON + annotated PNG)."""
    out_dir = Path(out_dir)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 2))) as pool:
        futs = {p: pool.submit(analyze_photo, p) for p in paths}
        for p, f in futs.items():
            try:
                results.append(f.result())
                log(f"PHOTO OK: {Path(p).name}")
            except Exception as exc:
                errors.append(f"{Path(p).name}: {exc}")
                log(f"PHOTO FAILED: {Path(p).name} ({exc})")
    if not results:
        raise ValueError("; ".join(errors) or "no usable photo")
    metrics = {k: float(np.mean([r["metrics"][k] for r in results])) for k in results[0]["metrics"]}
    neutral = [r for r in results if r["neutral_like"]]
    bias = np.median([r["bs"] for r in neutral], axis=0) if neutral else np.zeros(NBS)
    identity = np.median([r["L"] for r in results], axis=0)
    name = Path(paths[0]).stem + "_profile"
    top = np.argsort(-results[0]["bs"])[:3]
    acts = MuscleModel.activations(results[0]["bs"])
    lines = [f"PROFILE  {name}   photos used: {len(results)}/{len(paths)}",
             f"neutral-expression photos: {len(neutral)} (bias {'applied' if neutral else 'not applied'})", ""]
    lines += [f"{k:<14}{v:8.3f}" for k, v in metrics.items()]
    lines += ["", "Strongest expression in first photo: " +
              ", ".join(f"{BLENDSHAPE_NAMES[i]} {results[0]['bs'][i]:.2f}" for i in top),
              "Most active muscles: " + ", ".join(f"{MUSCLE_NAMES[i]} {acts[i]:.2f}" for i in np.argsort(-acts)[:4])]
    for r in results:
        if r["warn"]:
            lines.append(f"! {Path(r['path']).name}: " + ", ".join(r["warn"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "photos": [r["path"] for r in results], "metrics": metrics, "bias": bias.tolist(),
            "identity_L": np.round(identity, 4).tolist()}
    (out_dir / f"{name}.json").write_text(json.dumps(prof))
    ann = results[0]["img"].copy()
    for a, b in face_connections("TESSELATION"):
        pa, pb = results[0]["P"][a, :2], results[0]["P"][b, :2]
        cv2.line(ann, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (255, 240, 0), 1, cv2.LINE_AA)
    png = out_dir / f"{name}_analysis.png"
    cv2.imwrite(str(png), ann)
    lines.append(f"\nsaved: {out_dir / (name + '.json')}\nannotated: {png}")
    return prof, "\n".join(lines)
