"""Guided face scan and the personal face rig.

FaceEnrollment turns a live landmark stream into captured key poses (neutral, head turns, expressions) while the
operator is prompted ("keep your head inside the circle", "open your mouth", ...). FaceRig stores the actor's own
neutral face and how each expression moves the eyes, brows, nose and lips; at run time it solves the live face
into personal expression weights - the same idea as the actor-specific facial solves used in film VFX."""
import json
import math
from pathlib import Path

import numpy as np

from .constants import BS_INDEX, NBS

IPD_M = 0.094                      # typical outer-eye distance; face-frame units are multiples of it


def _bs(bs, *names):
    return float(np.mean([bs[BS_INDEX[n]] for n in names]))


class Step:
    """score(st) >= 1 means the requested pose/expression is reached. st carries yaw, pitch, the raw blendshapes bs,
    the face-frame landmarks L and the neutral references bs0 / L0 (None until the neutral face is captured)."""

    def __init__(self, key, kind, score):
        self.key, self.kind, self.score = key, kind, score


def _frontal(st):
    return abs(st["yaw"]) < 15 and abs(st["pitch"]) < 15


def _rel(names, thr):
    """Blendshape rise above the actor's own neutral, in units of the threshold (MediaPipe scores are small)."""
    def f(st):
        if not _frontal(st):
            return 0.0
        v = _bs(st["bs"], *names) - (_bs(st["bs0"], *names) if st["bs0"] is not None else 0.0)
        return v / thr
    return f


def _lm(fn):
    def f(st):
        if st["L0"] is None or not _frontal(st):
            return 0.0
        return float(fn(st["L"], st["L0"]))
    return f


def _brow_gap(L):
    return float(np.linalg.norm(L[107] - L[336]))


def _brow_h(L):
    return float((L[133, 1] - L[107, 1]) + (L[362, 1] - L[336, 1]))


def _frown_lm(L, L0):
    g0, h0 = _brow_gap(L0), _brow_h(L0)
    return max((g0 - _brow_gap(L)) / (0.07 * abs(g0) + 1e-9), (h0 - _brow_h(L)) / (0.06 * abs(h0) + 1e-9))


def _brows_up_lm(L, L0):
    h0 = _brow_h(L0)
    return (_brow_h(L) - h0) / (0.10 * abs(h0) + 1e-9)


def _cheek_lm(L, L0):
    w0 = float(np.linalg.norm(L0[205] - L0[425]))
    return (float(np.linalg.norm(L[205] - L[425])) - w0) / (0.05 * w0 + 1e-9)


def _either(a, b):
    return lambda st: max(a(st), b(st))


def build_steps():
    return [
        Step("center", "pose", lambda st: float(abs(st["yaw"]) < 8 and abs(st["pitch"]) < 8)),
        Step("turn_a", "pose", lambda st: float(abs(st["yaw"]) > 20 and abs(st["pitch"]) < 15)),
        Step("turn_b", "pose", lambda st: float(abs(st["pitch"]) < 15 and st["opposite"]("turn", st["yaw"], 20))),
        Step("up", "pose", lambda st: float(abs(st["pitch"]) > 12 and abs(st["yaw"]) < 15)),
        Step("down", "pose", lambda st: float(abs(st["yaw"]) < 15 and st["opposite"]("tilt", st["pitch"], 12))),
        Step("mouth_open", "expr", _rel(("jawOpen",), 0.35)),
        Step("smile", "expr", _rel(("mouthSmileLeft", "mouthSmileRight"), 0.30)),
        Step("pucker", "expr", _rel(("mouthPucker",), 0.30)),
        Step("brows_up", "expr", _either(_either(_rel(("browInnerUp",), 0.25), _rel(("browOuterUpLeft", "browOuterUpRight"), 0.25)),
                                         _lm(_brows_up_lm))),
        Step("brows_down", "expr", _either(_rel(("browDownLeft", "browDownRight"), 0.15), _lm(_frown_lm))),
        Step("eyes_closed", "expr", _rel(("eyeBlinkLeft", "eyeBlinkRight"), 0.50)),
        Step("cheeks_puff", "expr", _either(_rel(("cheekPuff",), 0.18), _lm(_cheek_lm))),
    ]


class FaceEnrollment:
    STABLE_FRAMES, CAPTURE_FRAMES = 6, 10

    def __init__(self):
        self.steps = build_steps()
        self.idx, self.captures = 0, {}
        self._signs = {}
        self._stable, self._buf, self._roll = 0, [], []
        self.score = 0.0

    @property
    def done(self):
        return self.idx >= len(self.steps)

    def _opposite(self, name, v, lim):
        s = self._signs.get(name)
        return s is not None and v * s < -lim

    def restart(self):
        self.__init__()

    def _finalize(self, frames):
        step = self.steps[self.idx]
        Ls, Bs = np.array([b[0] for b in frames]), np.array([b[1] for b in frames])
        my, mp = float(np.median([b[2] for b in frames])), float(np.median([b[3] for b in frames]))
        self.captures[step.key] = {"L": np.median(Ls, axis=0), "bs": np.median(Bs, axis=0), "yaw": my, "pitch": mp}
        if step.key == "turn_a":
            self._signs["turn"] = math.copysign(1.0, my) if abs(my) > 1e-6 else 1.0
        elif step.key == "up":
            self._signs["tilt"] = math.copysign(1.0, mp) if abs(mp) > 1e-6 else 1.0
        self.idx += 1
        self._stable, self._buf = 0, []

    def skip(self):
        """Skip the current step (not possible for the neutral face, which everything else is measured against)."""
        if not self.done and self.steps[self.idx].key != "center":
            self.idx += 1
            self._stable, self._buf = 0, []

    def capture_now(self):
        """Capture the last few frames immediately (for the actor whose expression never crosses the threshold)."""
        if not self.done and len(self._roll) >= 3:
            self._finalize(self._roll[-self.CAPTURE_FRAMES:])

    def update(self, L, bs, yaw, pitch, in_circle):
        """Feed one frame (L: face-frame landmarks, bs: raw blendshapes, yaw/pitch in degrees, in_circle: the face
        is inside the guide circle). Returns a state dict for the UI."""
        if self.done:
            return self.state(True)
        step = self.steps[self.idx]
        c = self.captures.get("center")
        st = {"yaw": yaw, "pitch": pitch, "bs": bs, "L": L, "opposite": self._opposite,
              "bs0": None if c is None else c["bs"], "L0": None if c is None else c["L"]}
        self.score = float(step.score(st))
        self._roll.append((np.array(L, dtype=np.float64), np.array(bs, dtype=np.float64), yaw, pitch))
        del self._roll[:-self.CAPTURE_FRAMES]
        ok = bool(in_circle) and self.score >= 1.0
        if ok:
            self._stable += 1
            if self._stable > self.STABLE_FRAMES:
                self._buf.append(self._roll[-1])
            if len(self._buf) >= self.CAPTURE_FRAMES:
                self._finalize(self._buf)
        else:
            self._stable, self._buf = 0, []
        return self.state(ok)

    def state(self, ok):
        n = len(self.steps)
        cur = self.steps[min(self.idx, n - 1)]
        prog = 1.0 if self.done else min(1.0, self._stable / (self.STABLE_FRAMES + self.CAPTURE_FRAMES))
        return {"idx": self.idx, "total": n, "key": cur.key, "ok": ok, "progress": prog, "done": self.done,
                "hint": "scan." + cur.key, "score": float(min(max(self.score, 0.0), 1.5)), "skippable": cur.key != "center"}


class FaceRig:
    """Actor-specific face: neutral shape, expression displacements and the vertices that matter for performance."""

    def __init__(self, identity, bias, gain, keys, deltas, bs_peak, regions, tris, meta=None):
        self.identity = np.asarray(identity, float)                 # (478,3) face-frame landmarks of the neutral face
        self.bias, self.gain = np.asarray(bias, float), np.asarray(gain, float)
        self.keys = list(keys)
        self.deltas = np.asarray(deltas, float)                     # (n_expr, 478, 3)
        self.bs_peak = np.asarray(bs_peak, float)
        self.regions = {k: list(map(int, v)) for k, v in regions.items()}
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.meta = meta or {}
        self.feat = np.array(sorted({i for v in self.regions.values() for i in v}), dtype=np.int64)
        self._A = None

    @classmethod
    def from_captures(cls, caps, regions, tris):
        neutral = caps["center"]
        expr = [k for k in caps if k not in ("center",)]
        deltas = [caps[k]["L"] - neutral["L"] for k in expr]
        peaks = [caps[k]["bs"] for k in expr]
        span = np.max([np.maximum(p - neutral["bs"], 0) for p in peaks], axis=0)
        gain = np.where(span > 0.2, np.clip(0.9 / np.maximum(span, 1e-3), 0.8, 2.5), 1.0)
        if not expr:
            raise ValueError("no expression was captured")
        meta = {"yaw": {k: caps[k]["yaw"] for k in caps}, "pitch": {k: caps[k]["pitch"] for k in caps}}
        return cls(neutral["L"], neutral["bs"], gain, expr, deltas, peaks, regions, tris, meta)

    def solve(self, L, iters=25, ridge=1e-3):
        """Personal expression weights (n_expr,) in [0,1.3] that best explain the live landmark displacement."""
        if self._A is None:
            A = self.deltas[:, self.feat].reshape(len(self.keys), -1).T          # (3F, n)
            self._A, self._AtA = A, A.T @ A + ridge * np.eye(A.shape[1])
        L = np.asarray(L, float)
        d = ((L - self.identity[self.feat]) if len(L) == len(self.feat) else (L - self.identity)[self.feat]).reshape(-1)
        c = np.linalg.solve(self._AtA, self._A.T @ d)
        lip = 1.0 / (np.diag(self._AtA).max() + 1e-9)
        for _ in range(iters):                                                   # projected gradient = non-negative LS
            c = np.clip(c - lip * (self._AtA @ c - self._A.T @ d), 0.0, 1.3)
        return c

    def calibrated_blendshapes(self, bs_raw):
        return np.clip((np.asarray(bs_raw) - self.bias) * self.gain, 0.0, 1.0)

    def neutral_points(self, scale=IPD_M):
        return self.identity[self.feat] * scale

    # -- file ---------------------------------------------------------------------------------------
    def save(self, path):
        d = {"version": 1, "identity": np.round(self.identity, 5).tolist(), "bias": self.bias.tolist(),
             "gain": self.gain.tolist(), "keys": self.keys, "deltas": np.round(self.deltas, 5).tolist(),
             "bs_peak": np.array(self.bs_peak).tolist(), "regions": self.regions, "tris": self.tris.tolist(),
             "meta": self.meta}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["identity"], d["bias"], d["gain"], d["keys"], d["deltas"], d["bs_peak"], d["regions"], d["tris"],
                   d.get("meta"))
