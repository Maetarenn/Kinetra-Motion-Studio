"""Rigged head (.glb with ARKit morph targets) and the 468-vertex live face mesh builder."""
import re
from pathlib import Path

import numpy as np

from .core.constants import BLENDSHAPE_NAMES, NBS
from .capture.models import face_connections

_ARKIT_NORM = {re.sub("[^a-z]", "", n.lower()): i for i, n in enumerate(BLENDSHAPE_NAMES)}


def arkit_index(name):
    n = re.sub("[^a-z]", "", name.lower())
    if n in _ARKIT_NORM:
        return _ARKIT_NORM[n]
    if n.endswith("l") and n[:-1] + "left" in _ARKIT_NORM:
        return _ARKIT_NORM[n[:-1] + "left"]
    if n.endswith("r") and n[:-1] + "right" in _ARKIT_NORM:
        return _ARKIT_NORM[n[:-1] + "right"]
    return None


_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_accessor(g, blob, ai):
    a = g.accessors[ai]
    dt, n = np.dtype(_CT[a.componentType]), _NC[a.type]
    if a.bufferView is None:
        return np.zeros((a.count, n), dt)
    bv = g.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or n * dt.itemsize
    raw = np.frombuffer(blob, np.uint8)
    if stride == n * dt.itemsize:
        return np.frombuffer(blob, dt, a.count * n, off).reshape(a.count, n).copy()
    sel = off + np.arange(a.count)[:, None] * stride + np.arange(n * dt.itemsize)[None, :]
    return raw[sel].copy().view(dt).reshape(a.count, n)


def _node_matrix(n):
    if n.matrix:
        return np.array(n.matrix, dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    if n.scale:
        M[:3, :3] = np.diag(n.scale)
    if n.rotation:
        x, y, z, w = n.rotation
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M[:3, :3] = R @ M[:3, :3]
    if n.translation:
        M[:3, 3] = n.translation
    return M


class MorphCharacter:
    """Rigged head loaded from .glb. Morph targets are matched to the ARKit-52 names; deformation runs as one
    (52 x V*3) matrix product per frame."""

    def __init__(self, path):
        from pygltflib import GLTF2
        g = GLTF2().load_binary(str(path))
        blob = g.binary_blob()
        world = {}

        def walk(ni, parent):
            world[ni] = parent @ _node_matrix(g.nodes[ni])
            for c in g.nodes[ni].children or []:
                walk(c, world[ni])
        for sc in g.scenes:
            for r in sc.nodes:
                walk(r, np.eye(4))
        picks = []
        for ni, node in enumerate(g.nodes):
            if node.mesh is None:
                continue
            mesh = g.meshes[node.mesh]
            M = np.eye(4) if node.skin is not None else world.get(ni, np.eye(4))
            for prim in mesh.primitives:
                if prim.targets:
                    names = None
                    for ex in (getattr(prim, "extras", None), getattr(mesh, "extras", None)):
                        if isinstance(ex, dict) and ex.get("targetNames"):
                            names = ex["targetNames"]
                            break
                    picks.append((f"{node.name or ''} {mesh.name or ''}".lower(), M, prim, names))
        if not picks:
            raise ValueError("no mesh with morph targets found in this file")
        pref = [p for p in picks if any(k in p[0] for k in ("head", "face", "teeth", "tongue", "eye"))]
        picks = pref or picks
        bases, idxs, parts, off = [], [], [], 0
        matched, unmatched = set(), []
        for _, M, prim, names in picks:
            pos = _read_accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
            A, t = M[:3, :3], M[:3, 3]
            bases.append(pos @ A.T + t)
            idx = (_read_accessor(g, blob, prim.indices).reshape(-1).astype(np.uint32) if prim.indices is not None
                   else np.arange(len(pos), dtype=np.uint32))
            idxs.append(idx + off)
            deltas = {}
            for ti, tgt in enumerate(prim.targets):
                tname = names[ti] if names and ti < len(names) else f"target{ti}"
                k = arkit_index(tname)
                if k is None:
                    unmatched.append(tname)
                    continue
                matched.add(k)
                deltas[k] = (_read_accessor(g, blob, tgt["POSITION"]).astype(np.float64) @ A.T
                             if "POSITION" in tgt else np.zeros((len(pos), 3)))
            parts.append((off, len(pos), deltas))
            off += len(pos)
        base = np.concatenate(bases)
        lo, hi = base.min(0), base.max(0)
        self.scale = 2.6 / float((hi - lo).max())
        self.center = (lo + hi) / 2
        self.base = ((base - self.center) * self.scale).astype(np.float32)
        self.idx = np.concatenate(idxs).astype(np.uint32)
        D = np.zeros((NBS, off, 3), np.float32)
        for o, n, deltas in parts:
            for k, d in deltas.items():
                D[k, o:o + n] = d * self.scale
        self.D = D.reshape(NBS, -1)
        self.matched, self.unmatched, self.V = len(matched), unmatched, off
        self.name = Path(path).name
        self.tris = self.idx.reshape(-1, 3).astype(np.int64)
        self.col0 = np.zeros((self.V, 4), np.float32)

    def pose(self, weights):
        return (self.base.reshape(-1) + weights.astype(np.float32) @ self.D).reshape(-1, 3)


class MeshBuilder:
    """Triangle list of the 468-vertex face mesh (3-cliques of the tessellation graph), consistently oriented
    outward, plus per-frame smooth normals."""
    CENTER = np.array([0.0, -0.8, -1.2])

    def __init__(self):
        adj = {}
        edges = face_connections("TESSELATION")
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        tris = {tuple(sorted((a, b, c))) for a, b in edges for c in adj[a] & adj[b]}
        self.tris = np.array(sorted(tris), dtype=np.uint32)
        self.oriented = False

    def orient(self, pos):
        p = pos[self.tris]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        flip = (n * (p.mean(1) - self.CENTER)).sum(1) < 0
        self.tris[flip] = self.tris[flip][:, [0, 2, 1]]
        self.oriented = True

    @staticmethod
    def normals(pos, tris):
        p = pos[tris]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        out = np.zeros_like(pos)
        for k in range(3):
            for c in range(3):
                out[:, c] += np.bincount(tris[:, k], weights=fn[:, c], minlength=len(pos))
        return (out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
