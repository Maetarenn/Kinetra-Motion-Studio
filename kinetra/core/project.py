"""Projects, takes, raw recording and offline (non-causal) reprocessing.

project/
  project.json            name, created, notes
  calibration.json        multi-camera rig (optional)
  takes/take_0001/
      raw.npz             unfiltered joints, hand landmarks, timestamps, blendshapes (source of truth)
      meta.json           mode, fps, lengths, skeleton (rest offsets), solver settings
      motion.bvh / motion.usda / face_curves.csv
"""
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .body_solver import BodySolver
from .constants import (HIP_MID, SHO_MID, EAR_MID, HIP_L, HIP_R, SHO_L, SHO_R, EAR_L, EAR_R, NBS, JOINT_ORDER)
from .exporters import write_bvh, write_usda, write_curves_csv, quats_to_array
from .fullbody import Skeleton, assemble, build_skeleton, JOINT_ORDER_V2
from .hand import HandSolver
from .mathx import OneEuroFilter

SIDES = ("L", "R")


class TakeRecorder:
    """Thread-safe frame collector. Frames may carry body data, face data, hands or any mix."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = False
        self.order = JOINT_ORDER
        self.face_idx = None
        self._reset()

    def _reset(self):
        self.t, self.raw, self.vis, self.root_raw, self.root_out = [], [], [], [], []
        self.quats, self.bs, self.hands, self.face_L = [], [], [], []
        self.absolute, self.lengths, self.skeleton = False, None, None

    def start(self):
        with self._lock:
            self._reset()
            self.active = True

    def add(self, t, raw, vis, root_raw, root_out, quats, bs, absolute, lengths, hands_raw=None, skeleton=None, face_L=None):
        with self._lock:
            if not self.active:
                return
            nan3, nan2 = np.full((33, 3), np.nan), np.full(2, np.nan)
            self.t.append(t)
            self.raw.append(nan3 if raw is None else np.asarray(raw, float).copy())
            self.vis.append(np.zeros(33) if vis is None else np.asarray(vis, float).copy())
            self.root_raw.append(nan2 if root_raw is None else np.asarray(root_raw, float).copy())
            self.root_out.append(np.full(3, np.nan) if root_out is None else np.asarray(root_out, float).copy())
            self.quats.append(None if quats is None else quats_to_array([quats], self.order)[0])
            self.bs.append(np.full(NBS, np.nan) if bs is None else np.asarray(bs, float).copy())
            self.hands.append(np.full((2, 21, 3), np.nan) if hands_raw is None else np.asarray(hands_raw, float).copy())
            if self.face_idx is not None:
                nf = len(self.face_idx)
                self.face_L.append(np.full((nf, 3), np.nan, np.float32) if face_L is None
                                   else np.asarray(face_L, np.float32)[self.face_idx])
            self.absolute = absolute
            if lengths:
                self.lengths = dict(lengths)
            if skeleton is not None:
                self.skeleton = skeleton

    def __len__(self):
        return len(self.t)

    def stop(self):
        with self._lock:
            self.active = False
            qa = np.array([np.full((len(self.order), 4), np.nan) if q is None else q for q in self.quats])
            return {"t": np.array(self.t), "raw": np.array(self.raw), "vis": np.array(self.vis),
                    "root_raw": np.array(self.root_raw), "root_out": np.array(self.root_out), "quats": qa,
                    "bs": np.array(self.bs), "hands_raw": np.array(self.hands), "absolute": np.array(self.absolute),
                    "face_L": np.array(self.face_L, dtype=np.float32) if self.face_L else np.zeros((len(self.t), 0, 3), np.float32),
                    "lengths": self.lengths or {},
                    "skeleton": self.skeleton.to_meta() if self.skeleton is not None else None}


class Project:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = {"name": self.path.name, "created": datetime.now().isoformat(timespec="seconds"), "notes": ""}

    @classmethod
    def create(cls, path, name=None):
        p = cls(path)
        p.path.mkdir(parents=True, exist_ok=True)
        (p.path / "takes").mkdir(exist_ok=True)
        if name:
            p.meta["name"] = name
        p.save()
        return p

    @classmethod
    def open(cls, path):
        p = cls(path)
        f = p.path / "project.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found")
        p.meta.update(json.loads(f.read_text()))
        (p.path / "takes").mkdir(exist_ok=True)
        return p

    def save(self):
        (self.path / "project.json").write_text(json.dumps(self.meta, indent=1))

    @property
    def rig_path(self):
        return self.path / "calibration.json"

    def next_take_dir(self, label=""):
        n = 1 + max([int(d.name.split("_")[1]) for d in (self.path / "takes").glob("take_[0-9]*") if d.is_dir()] or [0])
        d = self.path / "takes" / f"take_{n:04d}"
        d.mkdir(parents=True)
        return d

    def takes(self):
        out = []
        for d in sorted((self.path / "takes").glob("take_*")):
            f = d / "meta.json"
            if f.exists():
                try:
                    m = json.loads(f.read_text())
                    m["dir"] = str(d)
                    m.setdefault("name", d.name)
                    out.append(m)
                except ValueError:
                    pass
        return out


def export_face_assets(d, rig, times, face_L):
    """Face performance assets from the recorded feature landmarks: USD point cache + personal expression curves."""
    from .exporters import write_points_usd
    from .facerig import IPD_M
    ok = ~np.isnan(face_L).any(axis=(1, 2))
    if ok.sum() < 2:
        return {}
    idx = {int(v): i for i, v in enumerate(rig.feat)}
    tris = [[idx[a] for a in t] for t in rig.tris if all(int(a) in idx for a in t)]
    files = {"usd": write_points_usd(d / "face_points.usda", times[ok], face_L[ok] * IPD_M, tris)}
    w = np.array([rig.solve(L) for L in face_L[ok]])
    import csv
    with open(d / "face_personal.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Timestamp"] + rig.keys)
        for t, row in zip(times[ok] - times[ok][0], w):
            wr.writerow([round(float(t), 4)] + [round(float(v), 4) for v in row])
    files["csv"] = str(d / "face_personal.csv")
    return files


def save_take(project, data, mode="single", settings=None, label="", face_rig=None):
    """Write raw.npz and the standard exports of a finished recording. Returns the take directory or None."""
    if len(data["t"]) < 2:
        return None
    d = project.next_take_dir(label)
    np.savez_compressed(d / "raw.npz", **{k: v for k, v in data.items() if k not in ("lengths", "skeleton")})
    meta = {"name": d.name, "label": label, "created": datetime.now().isoformat(timespec="seconds"), "mode": mode,
            "frames": int(len(data["t"])), "duration_s": float(data["t"][-1] - data["t"][0]),
            "lengths": data["lengths"], "skeleton": data.get("skeleton"), "settings": settings or {},
            "processed": "live"}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    export_take(d)
    if face_rig is not None and data.get("face_L") is not None and data["face_L"].shape[1] == len(face_rig.feat):
        export_face_assets(d, face_rig, data["t"], data["face_L"])
    return d


def _skel_from_meta(meta):
    return Skeleton.from_meta(meta["skeleton"]) if meta.get("skeleton") else None


def export_take(take_dir, fps=None):
    """(Re)write BVH / USD / face CSV from the quaternions stored in motion.npz (if reprocessed) or raw.npz."""
    d = Path(take_dir)
    src = d / "motion.npz" if (d / "motion.npz").exists() else d / "raw.npz"
    z = np.load(src, allow_pickle=False)
    meta = json.loads((d / "meta.json").read_text())
    skel = _skel_from_meta(meta)
    t, q, root, bs = z["t"], z["quats"], z["root_out"], z["bs"]
    ok = ~np.isnan(q).any(axis=(1, 2)) & ~np.isnan(root).any(axis=1)
    files = {}
    if ok.sum() >= 2:
        files["bvh"] = write_bvh(d / "motion.bvh", t[ok], root[ok], q[ok], meta["lengths"], fps, skel)
        files["usd"] = write_usda(d / "motion.usda", t[ok], root[ok], q[ok], meta["lengths"], fps, skel=skel)
    fk = ~np.isnan(bs).any(axis=1)
    if fk.sum() >= 2:
        files["csv"] = write_curves_csv(d / "face_curves.csv", t[fk], bs[fk])
    return files


# ----------------------------------------------------------------------------------------------
def _zero_phase(seq_t, seq, f_min, beta):
    """Forward + backward One-Euro pass averaged -> no phase lag (only possible offline)."""
    fa, fb = OneEuroFilter(f_min, beta), OneEuroFilter(f_min, beta)
    fwd = np.array([fa(x, t) for x, t in zip(seq, seq_t)])
    rt = seq_t[-1] - seq_t[::-1] + seq_t[0]
    bwd = np.array([fb(x, t) for x, t in zip(seq[::-1], rt)])[::-1]
    return (fwd + bwd) / 2.0


def reprocess_take(take_dir, beta=2.0, f_min=1.0, progress=None):
    """Offline pass over raw.npz: whole-take calibration, zero-phase smoothing of body and hands, then the same
    solvers as live. Writes motion.npz and refreshes the exports. Returns the number of body frames solved."""
    d = Path(take_dir)
    z = np.load(d / "raw.npz", allow_pickle=False)
    meta = json.loads((d / "meta.json").read_text())
    absolute = bool(z["absolute"])
    t, raw, vis, root_raw = z["t"], z["raw"], z["vis"], z["root_raw"]
    ok = ~np.isnan(raw).any(axis=(1, 2))
    idx = np.where(ok)[0]
    if len(idx) < 8:
        raise ValueError("too few body frames to reprocess")
    st = t[idx]
    sm = _zero_phase(st, raw[idx].reshape(len(idx), -1), f_min, beta).reshape(-1, 33, 3)
    rr = root_raw[idx]
    sm_root = _zero_phase(st, rr, f_min, beta) if not np.isnan(rr).any() else None

    def full(f):
        J = np.empty((36, 3))
        J[:33] = f
        J[HIP_MID], J[SHO_MID], J[EAR_MID] = (f[HIP_L] + f[HIP_R]) / 2, (f[SHO_L] + f[SHO_R]) / 2, (f[EAR_L] + f[EAR_R]) / 2
        return J

    # hands: zero-phase smoothing per hand over the frames where the hand exists
    v2 = bool(meta.get("skeleton"))
    hands_sm = {}
    if v2 and "hands_raw" in z.files:
        hr = z["hands_raw"]
        for hi, s in enumerate(SIDES):
            have = np.where(~np.isnan(hr[:, hi]).any(axis=(1, 2)))[0]
            if len(have) >= 8:
                sm_h = _zero_phase(t[have], hr[have, hi].reshape(len(have), -1), f_min * 2, beta).reshape(-1, 21, 3)
                hands_sm[s] = {int(i): sm_h[k] for k, i in enumerate(have)}

    solver = BodySolver()
    solver.preset_from_sequence([full(f) for f in sm], absolute=absolute)
    solver.root_filter = lambda x, tt: np.asarray(x, dtype=float)           # root already smoothed
    hand_solvers = {s: HandSolver() for s in SIDES}
    for s, frames in hands_sm.items():                                       # learn each hand's template first
        for i in list(frames)[:200]:
            hand_solvers[s].solve(frames[i], learn=True)
    order = JOINT_ORDER_V2 if v2 else JOINT_ORDER
    q_out = np.full((len(t), len(order), 4), np.nan)
    root_out = np.full_like(z["root_out"], np.nan)
    lengths = None
    for k, i in enumerate(idx):
        r = solver.process(sm[k], float(t[i]), root=None if sm_root is None else sm_root[k], vis=vis[i],
                           absolute=absolute, prefiltered=True)
        if v2:
            hres = {s: (hand_solvers[s].solve(hands_sm[s][int(i)], learn=False) if int(i) in hands_sm.get(s, {}) else None)
                    for s in SIDES}
            bsv = z["bs"][i]
            qd = assemble(r["quats"], r["world"], hres, None if np.isnan(bsv).any() else bsv)
        else:
            qd = r["quats"]
        q_out[i] = [qd[n] for n in order]
        root_out[i] = r["joints"][HIP_MID]
        lengths = r["lengths"]
        if progress and k % 50 == 0:
            progress(k / len(idx))
    np.savez_compressed(d / "motion.npz", t=t, quats=q_out, root_out=root_out, bs=z["bs"])
    meta["lengths"] = lengths or meta["lengths"]
    if v2:
        meta["skeleton"] = build_skeleton(meta["lengths"], hand_solvers).to_meta()
    meta["processed"] = f"offline zero-phase beta={beta} fmin={f_min}"
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    export_take(d)
    return len(idx)
