"""Detailed procedural human model bound rigidly to the 52-joint skeleton.

Built from the skeleton's own rest offsets, so it automatically follows the user's measured bone lengths and hand
template. Every finger segment, palm, foot with toes, head with eyes (they look around), nose, brows, ears,
lips and a jaw that opens are separate parts. Pure numpy: usable by the OpenGL viewport, by OBJ export and by the
software preview.
"""
import math

import numpy as np

from .fullbody import SIDES
from .hand import FINGERS
from .mathx import unit, q_from_vectors, q_to_mat

SKIN, SUIT, SUIT2 = (0.87, 0.67, 0.57), (0.23, 0.29, 0.40), (0.10, 0.45, 0.52)
SHOE, HAIR, LIP, EYE_W, IRIS, BROW = (0.09, 0.10, 0.13), (0.14, 0.10, 0.08), (0.70, 0.35, 0.36), (0.95, 0.95, 0.93), (0.16, 0.10, 0.07), (0.16, 0.11, 0.08)


def _revolve(profile, nseg):
    prof = np.asarray(profile, float)
    n = len(prof)
    ang = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    x = prof[:, 0, None] * np.cos(ang)[None]
    z = prof[:, 0, None] * np.sin(ang)[None]
    y = np.repeat(prof[:, 1, None], nseg, axis=1)
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    tris = []
    for i in range(n - 1):
        for k in range(nseg):
            a, b = i * nseg + k, i * nseg + (k + 1) % nseg
            c, d = (i + 1) * nseg + k, (i + 1) * nseg + (k + 1) % nseg
            tris += [(a, c, b), (b, c, d)]
    return verts, np.array(tris, dtype=np.int64)


def _sphere_profile(r, nv):
    t = np.linspace(-math.pi / 2, math.pi / 2, nv)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def capsule(r0, r1, length, nseg=10, nv=4):
    t0 = np.linspace(-math.pi / 2, 0, nv, endpoint=False)
    t1 = np.linspace(0, math.pi / 2, nv)
    prof = [(r0 * math.cos(t), r0 * math.sin(t)) for t in t0] + [(r0, 0.0), (r1, length)] + \
           [(r1 * math.cos(t), length + r1 * math.sin(t)) for t in t1[1:]]
    return _revolve(prof, nseg)


def ellipsoid(radii, nseg=14, nv=9):
    v, t = _revolve(_sphere_profile(1.0, nv), nseg)
    return v * np.asarray(radii, float), t


def _normals(v, t):
    p = v[t]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out = np.zeros_like(v)
    for k in range(3):
        for c in range(3):
            out[:, c] += np.bincount(t[:, k], weights=fn[:, c], minlength=len(v))
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def stylized_head_mesh(H=0.225, n_rings=32, n_seg=36):
    """Sculpted mannequin head with clean planar faceplate and smooth cranium."""
    chin_y = 0.010
    chin_z = 0.028
    verts = [[0.0, chin_y, chin_z]]
    rings = []
    for i in range(1, n_rings):
        t = i / float(n_rings)
        y = chin_y + (H - chin_y) * t
        if t < 0.26:
            s = t / 0.26
            f = s * s * (3.0 - 2.0 * s)
            rx = 0.014 + 0.048 * f
            zf = 0.016 + 0.052 * f
            zb = 0.012 + 0.048 * f
            cz = chin_z * (1.0 - 0.72 * f)
            p_front = 2.0 + 1.5 * f
        elif t < 0.65:
            s = (t - 0.26) / 0.39
            rx = 0.062 + 0.014 * math.sin(s * math.pi * 0.95)
            zf = 0.068 + 0.008 * math.sin(s * math.pi)
            zb = 0.060 + 0.030 * s
            cz = (chin_z * 0.28) * (1.0 - s) - 0.012 * s
            p_front = 3.5 - 1.1 * s
        else:
            s = (t - 0.65) / 0.35
            dome = math.sqrt(max(0.0, 1.0 - s * s))
            rx = 0.076 * dome
            zf = 0.076 * dome
            zb = 0.090 * dome
            cz = -0.012 - 0.006 * s
            p_front = 2.4 - 0.4 * s

        ring = []
        for j in range(n_seg):
            theta = 2.0 * math.pi * j / n_seg
            ct, st = math.cos(theta), math.sin(theta)
            if st >= 0:
                p = p_front
                denom = (abs(ct)**p + abs(st)**p)**(1.0 / p) if (abs(ct)**p + abs(st)**p) > 1e-6 else 1.0
                vx = rx * ct / denom
                vz = cz + zf * st / denom
            else:
                vx = rx * ct
                vz = cz + zb * st
            ring.append([vx, y, vz])
        rings.append(ring)
        verts.extend(ring)

    crown_z = -0.018
    verts.append([0.0, H, crown_z])
    V = np.array(verts, dtype=np.float32)
    tris = []
    for k in range(n_seg):
        kn = (k + 1) % n_seg
        tris.append([0, 1 + kn, 1 + k])
    for i in range(len(rings) - 1):
        r0 = 1 + i * n_seg
        r1 = 1 + (i + 1) * n_seg
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            tris.append([r0 + k, r1 + k, r0 + kn])
            tris.append([r0 + kn, r1 + k, r1 + kn])
    top_idx = len(V) - 1
    last_ring = 1 + (len(rings) - 1) * n_seg
    for k in range(n_seg):
        kn = (k + 1) % n_seg
        tris.append([top_idx, last_ring + k, last_ring + kn])
    T = np.array(tris, dtype=np.int64)
    return V, T


class BodyModel:
    def __init__(self, skel):
        self.skel = skel
        self.jidx = {j: i for i, j in enumerate(skel.order)}
        self._parts = []                      # (joint, verts, tris, color)
        self._build()
        self._finish()

    # -- part helpers -----------------------------------------------------------------------------
    def _add(self, joint, verts, tris, color, origin=(0, 0, 0), align=None):
        v = np.asarray(verts, float)
        if align is not None:
            v = v @ q_to_mat(q_from_vectors([0.0, 1.0, 0.0], align)).T
        v = v + np.asarray(origin, float)
        t = np.asarray(tris)
        c = v.mean(axis=0)
        n = _normals(v, t)
        if float(np.mean(np.einsum("ij,ij->i", n, v - c))) < 0:        # make triangles face outwards
            t = t[:, [0, 2, 1]]
        self._parts.append((joint, v, t, np.asarray(color, float)))

    def _ell(self, joint, center, radii, color, nseg=14, nv=9):
        v, t = ellipsoid(radii, nseg, nv)
        self._add(joint, v, t, color, center)

    def _limb(self, joint, vec, r0, r1, color, nseg=10, nv=4):
        L = float(np.linalg.norm(vec))
        if L < 1e-6:
            return
        v, t = capsule(r0, r1, L, nseg, nv)                           # axis runs joint -> child, caps overlap neighbours
        self._add(joint, v, t, color, (0, 0, 0), unit(vec))

    def _child_vec(self, joint):
        kids = [c for c in self.skel.order if self.skel.parent.get(c) == joint]
        if kids:
            return np.array(self.skel.offsets[kids[0]], float)
        return np.array(self.skel.ends.get(joint, (0, 0.05, 0)), float)

    # -- the model --------------------------------------------------------------------------------
    def _build(self):
        sk, o = self.skel, self.skel.offsets
        sp = 2 * o["chest"][1]
        sw, hw = o["upper_arm_L"][0], o["thigh_L"][0]
        # torso
        self._ell("Hips", (0, 0.06, 0), (hw * 1.6, 0.11, 0.115), SUIT)
        self._limb("spine", o["chest"], 0.088, 0.095, SUIT)
        self._ell("chest", (0, 0.24 * sp, 0), (sw * 1.1, 0.34 * sp, 0.125), SUIT)
        self._ell("neck", (0, -0.015, 0), (sw * 0.95, 0.05, 0.085), SUIT)
        self._limb("neck", o["head"], 0.05, 0.045, SKIN)
        # head: clean sculpted stylized mannequin head
        hv, ht = stylized_head_mesh()
        self._add("head", hv, ht, SKIN)
        # arms, hands, fingers
        for s in SIDES:
            self._ell("upper_arm_" + s, (0, 0, 0), (0.054, 0.054, 0.054), SUIT, 12, 7)
            self._limb("upper_arm_" + s, o["forearm_" + s], 0.047, 0.036, SUIT)
            self._limb("forearm_" + s, o["hand_" + s], 0.037, 0.027, SUIT)
            self._ell("hand_" + s, (0, 0, 0), (0.031, 0.031, 0.031), SKIN, 10, 7)                # wrist
            mcp = np.array([o[f"{f}_1_{s}"] for f in FINGERS[1:]])
            c = mcp.mean(axis=0) * 0.55
            self._ell("hand_" + s, c, (0.021, abs(mcp[:, 1]).mean() * 0.62, (mcp[:, 2].max() - mcp[:, 2].min()) / 2 + 0.013), SKIN)
            for f, r in zip(FINGERS, (0.0108, 0.0097, 0.0100, 0.0092, 0.0080)):
                for i in (1, 2, 3):
                    j = f"{f}_{i}_{s}"
                    vec = np.array(sk.offsets[f"{f}_{i + 1}_{s}"]) if i < 3 else np.array(sk.ends[j])
                    rr = r * (1.0, 0.9, 0.8)[i - 1]
                    self._limb(j, vec, rr, rr * 0.88, SKIN, 8, 3)
                    self._ell(j, (0, 0, 0), (rr * 1.02,) * 3, SKIN, 8, 5)
                self._ell(f"{f}_3_{s}", sk.ends[f"{f}_3_{s}"], (r * 0.75,) * 3, SKIN, 8, 5)
        # legs, feet, toes
        for s in SIDES:
            self._ell("thigh_" + s, (0, 0, 0), (0.08, 0.08, 0.08), SUIT, 12, 7)
            self._limb("thigh_" + s, o["shin_" + s], 0.08, 0.058, SUIT, 12)
            self._limb("shin_" + s, o["foot_" + s], 0.057, 0.037, SUIT, 12)
            fl = o["toe_" + s][2] / 0.65
            self._ell("foot_" + s, (0, -0.045, 0.45 * o["toe_" + s][2]), (0.045, 0.042, 0.55 * o["toe_" + s][2] + 0.02), SHOE)
            self._ell("foot_" + s, (0, -0.04, -0.025), (0.045, 0.048, 0.05), SHOE)
            self._ell("toe_" + s, (0, -0.045, 0.12 * fl), (0.042, 0.025, 0.12 * fl), SHOE)

    # -- assembly ---------------------------------------------------------------------------------
    def _finish(self):
        verts, norms, cols, tris, jid, base = [], [], [], [], [], 0
        for joint, v, t, c in self._parts:
            verts.append(v)
            norms.append(_normals(v, t))
            cols.append(np.tile(c, (len(v), 1)))
            tris.append(t + base)
            jid.append(np.full(len(v), self.jidx[joint]))
            base += len(v)
        V, N, C, T, J = (np.concatenate(a) for a in (verts, norms, cols, tris, jid))
        perm = np.argsort(J, kind="stable")                           # contiguous vertex range per joint
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        self.v_local, self.n_local = V[perm].astype(np.float32), N[perm].astype(np.float32)
        self.colors, self.tris = C[perm].astype(np.float32), inv[T].astype(np.uint32)
        Js = J[perm]
        self.slices = {int(j): (int(np.searchsorted(Js, j, "left")), int(np.searchsorted(Js, j, "right")))
                       for j in np.unique(Js)}

    def pose(self, pos, rot):
        """World-space vertices and normals for joint positions (J,3) and rotation matrices (J,3,3)."""
        V, N = np.empty_like(self.v_local), np.empty_like(self.n_local)
        for j, (a, b) in self.slices.items():
            R = rot[j].astype(np.float32)
            V[a:b] = self.v_local[a:b] @ R.T + pos[j].astype(np.float32)
            N[a:b] = self.n_local[a:b] @ R.T
        min_y = float(np.min(V[:, 1]))
        if min_y < 0.0:
            V[:, 1] -= min_y
        return V, N

    def write_obj(self, path, pos, rot):
        V, _ = self.pose(pos, rot)
        with open(path, "w") as f:
            f.write("# Kinetra body model\n")
            for c, v in zip(self.colors, V):
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
            for a, b, c in self.tris + 1:
                f.write(f"f {a} {b} {c}\n")
        return str(path)
