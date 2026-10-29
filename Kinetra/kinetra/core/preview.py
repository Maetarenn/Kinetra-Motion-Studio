"""Tiny software renderer (painter's algorithm + Lambert) used to preview meshes without OpenGL, e.g. for
documentation, tests and headless machines."""
import math

import cv2
import numpy as np


def render_mesh(V, N, C, T, yaw=0.45, pitch=0.12, size=(900, 900), center=None, extent=None, light=(0.4, 0.55, 0.75)):
    V = np.asarray(V, float)
    center = V.mean(axis=0) if center is None else np.asarray(center, float)
    ext = float(np.ptp(V, axis=0).max()) if extent is None else extent
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R = Rx @ Ry
    P = (V - center) @ R.T
    Nn = np.asarray(N, float) @ R.T
    scale = 0.92 * min(size) / ext
    xy = np.stack([P[:, 0] * scale + size[0] / 2, -P[:, 1] * scale + size[1] / 2], 1)
    tri = np.asarray(T, dtype=np.int64)
    zc = P[tri][:, :, 2].mean(axis=1)
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    front = fn[:, 2] > 0
    L = np.array(light) / np.linalg.norm(light)
    vn = Nn[tri].mean(axis=1)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
    shade = 0.28 + 0.72 * np.clip(vn @ L, 0, 1)
    col = (np.asarray(C)[tri].mean(axis=1) * shade[:, None])[:, ::-1] * 255          # RGB -> BGR
    img = np.full((size[1], size[0], 3), (14, 9, 7), np.uint8)
    for i in np.argsort(zc):
        if front[i]:
            cv2.fillConvexPoly(img, xy[tri[i]].astype(np.int32), tuple(float(c) for c in col[i]), cv2.LINE_AA)
    return img
