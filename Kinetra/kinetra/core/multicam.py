"""Multi-camera geometry: intrinsic calibration, floor-board extrinsics, triangulation.

World frame of a calibrated rig: metres, y up, the calibration board lying on the floor defines the origin
and the floor plane (y = 0). The rig works with any number of cameras >= 2 (a single camera simply does not
need a rig).
"""
import json
from pathlib import Path

import cv2
import numpy as np

# board frame (x right, y along rows, z = board normal) -> scene frame (y up)
M_UP_Z = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # board z points toward the cameras
M_UP_NZ = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # board z points into the floor


class CameraModel:
    def __init__(self, name, K, dist, R, t, size):
        self.name = name
        self.K = np.asarray(K, float).reshape(3, 3)
        self.dist = np.asarray(dist, float).reshape(-1)
        self.R = np.asarray(R, float).reshape(3, 3)      # world -> camera
        self.t = np.asarray(t, float).reshape(3)
        self.size = (int(size[0]), int(size[1]))

    @property
    def P(self):
        return np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self):
        return -self.R.T @ self.t

    @property
    def forward(self):
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    @property
    def focal(self):
        return float(self.K[0, 0])

    def normalize(self, pts_px):
        """Pixels -> undistorted normalised image coordinates (N, 2)."""
        p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist).reshape(-1, 2)

    def project(self, X):
        X = np.asarray(X, float).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        px, _ = cv2.projectPoints(X, rvec, self.t, self.K, self.dist)
        return px.reshape(-1, 2)

    def to_dict(self):
        return {"name": self.name, "K": self.K.tolist(), "dist": self.dist.tolist(), "R": self.R.tolist(),
                "t": self.t.tolist(), "size": list(self.size)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["K"], d["dist"], d["R"], d["t"], d["size"])


def default_intrinsics(size, fov_deg=65.0):
    """Rough pinhole guess for cameras that were never calibrated."""
    w, h = size
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), np.zeros(5)


# ----------------------------------------------------------------------------------------------
def triangulate_dlt(Ps, xs, ws):
    """Weighted DLT. Ps: list of 3x4 (normalised cameras), xs: list of (2,) normalised points."""
    A = []
    for P, x, w in zip(Ps, xs, ws):
        A.append(w * (x[0] * P[2] - P[0]))
        A.append(w * (x[1] * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


class Rig:
    REPROJ_PX = 12.0          # joints whose worst camera exceeds this error lose that camera
    MIN_VIS = 0.5

    def __init__(self, cameras=None, rms=None):
        self.cameras = list(cameras or [])
        self.rms = rms or {}

    def __len__(self):
        return len(self.cameras)

    def triangulate(self, pts_px, vis):
        """pts_px: list (per camera) of (N,2) pixel arrays or None; vis: list of (N,) visibility or None.
        Returns X (N,3) with NaN where < 2 cameras agree, and the number of cameras used per joint."""
        n = next(p.shape[0] for p in pts_px if p is not None)
        norm = [None if p is None else cam.normalize(p) for cam, p in zip(self.cameras, pts_px)]
        X = np.full((n, 3), np.nan)
        used = np.zeros(n, dtype=int)
        for j in range(n):
            idx = [c for c, nrm in enumerate(norm) if nrm is not None and vis[c] is not None and vis[c][j] > self.MIN_VIS]
            if len(idx) < 2:
                continue
            for _ in range(2):
                p = triangulate_dlt([self.cameras[c].P for c in idx], [norm[c][j] for c in idx],
                                    [float(vis[c][j]) for c in idx])
                errs = []
                for c in idx:
                    cam = self.cameras[c]
                    xc = cam.R @ p + cam.t
                    errs.append(np.linalg.norm(xc[:2] / xc[2] - norm[c][j]) * cam.focal if xc[2] > 1e-6 else 1e9)
                worst = int(np.argmax(errs))
                if errs[worst] > self.REPROJ_PX and len(idx) > 2:
                    idx.pop(worst)
                    continue
                break
            X[j], used[j] = p, len(idx)
        return X, used

    def save(self, path):
        Path(path).write_text(json.dumps({"cameras": [c.to_dict() for c in self.cameras], "rms": self.rms}, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls([CameraModel.from_dict(c) for c in d["cameras"]], d.get("rms"))


# ----------------------------------------------------------------------------------------------
def board_points(cols, rows, square_m):
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return obj


def find_board(gray, cols, rows):
    """Inner-corner chessboard detection with sub-pixel refinement. Returns (N,1,2) float32 or None."""
    ok, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners.astype(np.float32) if ok else None


class IntrinsicsCollector:
    """Collects well-spread board views of one camera, then solves K and the distortion."""

    def __init__(self, cols, rows, square_m, min_shift=0.04):
        self.cols, self.rows, self.square = cols, rows, square_m
        self.views, self.size, self._last, self.min_shift = [], None, None, min_shift

    def add(self, gray):
        return self.add_corners(find_board(gray, self.cols, self.rows), (gray.shape[1], gray.shape[0]))

    def add_corners(self, c, size):
        self.size = tuple(size)
        if c is None:
            return False
        diag = float(np.hypot(*self.size))
        if self._last is not None and float(np.linalg.norm(c - self._last, axis=-1).mean()) < self.min_shift * diag:
            return False
        self.views.append(c)
        self._last = c
        return True

    def solve(self):
        if len(self.views) < 8:
            raise ValueError("need at least 8 distinct board views")
        obj = board_points(self.cols, self.rows, self.square)
        rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(self.views), self.views, self.size, None, None)
        return K, dist.reshape(-1), float(rms), self.size


def solve_extrinsics(observations, cols, rows, square_m):
    """observations: list of (name, corners (N,1,2), K, dist, size) of ONE simultaneous shot of the board lying
    on the floor. Returns a Rig whose world is metres, y up, origin at the board's first corner."""
    obj = board_points(cols, rows, square_m)
    poses = []
    for name, corners, K, dist, size in observations:
        ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"pose of {name} could not be solved")
        Rb, _ = cv2.Rodrigues(rvec)
        poses.append((name, K, dist, size, Rb, tvec.reshape(3)))
    _, _, _, _, Rb0, t0 = poses[0]
    M = M_UP_Z if float(np.dot(Rb0[:, 2], -t0)) > 0 else M_UP_NZ
    cams = []
    for name, K, dist, size, Rb, t in poses:
        cam = CameraModel(name, K, dist, Rb @ M.T, t, size)
        if cam.center[1] <= 0:
            raise ValueError(f"{name}: board orientation differs between cameras; show the same board side to all")
        cams.append(cam)
    rms = {}
    for cam, (name, corners, *_rest) in zip(cams, observations):
        world = (M @ obj.T).T
        err = np.linalg.norm(cam.project(world) - corners.reshape(-1, 2), axis=1)
        rms[name] = float(np.sqrt(np.mean(err ** 2)))
    return Rig(cams, rms)
