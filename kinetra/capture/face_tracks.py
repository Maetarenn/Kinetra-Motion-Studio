"""Per-face tracking state: identity matching, filtering, neutral/ROM calibration, marker fusion, muscles."""
import time

import numpy as np

from ..core.constants import NBS, BLENDSHAPE_NAMES, BS_INDEX
from ..core.face import to_face_frame, head_quat, head_pose_scene, MuscleModel
from ..core.mathx import OneEuroFilter
from .markers import MarkerTracker

# One-Euro parameters: landmarks live in pixels, blendshapes in 1/s units
LM_FMIN, LM_BETA = 1.5, 0.05
BS_FMIN, BS_BETA = 2.5, 10.0


def blend_vector(categories):
    v = np.zeros(NBS)
    for c in categories:
        i = BS_INDEX.get(c.category_name)
        if i is not None:
            v[i] = c.score
    return v


class FaceTrack:
    def __init__(self, tid, bias=None, gain=None, neutral_L=None):
        self.id, self.centroid, self.last_seen = tid, np.zeros(2), time.perf_counter()
        self.lm_filter = OneEuroFilter(LM_FMIN, LM_BETA, 1.0, vector=True)
        self.bs_filter = OneEuroFilter(BS_FMIN, BS_BETA, 1.0, vector=False)
        self.bias = np.zeros(NBS) if bias is None else np.asarray(bias, float).copy()
        self.gain = np.ones(NBS) if gain is None else np.asarray(gain, float).copy()
        self.neutral_L, self.heat_W = (None if neutral_L is None else np.asarray(neutral_L, float).copy()), None
        self.markers = MarkerTracker()
        self.mode, self.buf = None, []


def process_face(tr, lm_norm, bs_raw, size, gray, t, cmds, markers_enabled):
    """One face of one frame -> packet dict (landmarks, blendshapes, muscles, head pose, calibration state)."""
    w, h = size
    ev = []
    P = np.column_stack([lm_norm[:, 0] * w, lm_norm[:, 1] * h, lm_norm[:, 2] * w])
    Pf = tr.lm_filter(P, t)
    if cmds.get("reg") and markers_enabled and gray is not None:
        n = tr.markers.register(gray, Pf)
        ev.append(("markers_found", tr.id, n) if n else ("markers_none", tr.id, 0))
    if markers_enabled and gray is not None and tr.markers.active:
        Pf = tr.markers.update(gray, Pf)
    L, R, origin, s = to_face_frame(Pf)
    bs_f = tr.bs_filter(bs_raw, t)
    if tr.neutral_L is None:
        tr.neutral_L = L.copy()
    if tr.heat_W is None:
        tr.heat_W = MuscleModel.vertex_weights(L)
    if cmds.get("neutral"):
        tr.mode, tr.buf = "neutral", []
    if cmds.get("rom"):
        tr.mode, tr.buf = "rom", []
    if tr.mode == "neutral":
        tr.buf.append((bs_f.copy(), L.copy()))
        if len(tr.buf) >= 45:
            tr.bias = np.median([b[0] for b in tr.buf], axis=0)
            tr.neutral_L = np.median([b[1] for b in tr.buf], axis=0)
            tr.mode = None
            ev.append(("neutral_done", tr.id, 0))
    elif tr.mode == "rom":
        tr.buf.append(bs_f.copy())
        if len(tr.buf) >= 180:
            p95 = np.percentile(np.array(tr.buf), 95, axis=0)
            span = p95 - tr.bias
            tr.gain = np.where(span > 0.2, np.clip(0.9 / np.maximum(span, 1e-3), 0.8, 2.5), 1.0)
            tr.mode = None
            ev.append(("rom_done", tr.id, float(tr.gain.max())))
    bs = np.clip((bs_f - tr.bias) * tr.gain, 0.0, 1.0)
    mk = tr.markers.points
    return {"id": tr.id, "lm": Pf, "L": L, "S": head_pose_scene(R), "quat": head_quat(R), "origin": origin,
            "scale": s, "bs": bs, "bs_raw": bs_raw, "muscles": MuscleModel.activations(bs), "heat_W": tr.heat_W,
            "neutral_L": tr.neutral_L, "markers": None if mk is None else mk.copy(),
            "marker_jitter": tr.markers.jitter, "marker_n": 0 if mk is None else len(mk), "events": ev,
            "calib": None if tr.mode is None else (tr.mode, len(tr.buf)),
            "bbox": (float(Pf[:468, 0].min()), float(Pf[:468, 1].min()),
                     float(Pf[:468, 0].max()), float(Pf[:468, 1].max()))}


class FaceTracker:
    """Keeps FaceTrack objects alive across frames and assigns detections to them by centroid distance."""

    def __init__(self):
        self.tracks, self._next_id = {}, 0
        self.profile_bias, self.profile_gain, self.profile_neutral = None, None, None

    def match(self, centroids, now, on_event=None):
        free, assign = set(self.tracks), {}
        for i, c in enumerate(centroids):
            best, bd = None, 0.2
            for tid in free:
                d = float(np.linalg.norm(self.tracks[tid].centroid - c))
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self._next_id
                self._next_id += 1
                self.tracks[best] = FaceTrack(best, self.profile_bias, self.profile_gain, self.profile_neutral)
                if on_event:
                    on_event(("face_acquired", best, 0))
            else:
                free.discard(best)
            tr = self.tracks[best]
            tr.centroid, tr.last_seen = c, now
            assign[i] = best
        return assign

    def prune(self, now, on_event=None, timeout=1.5):
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > timeout]:
            if on_event:
                on_event(("face_lost", tid, 0))
            del self.tracks[tid]

    def slots(self):
        return {tid: r for r, tid in enumerate(sorted(self.tracks))}

    def clear_markers(self):
        for tr in self.tracks.values():
            tr.markers.clear()
