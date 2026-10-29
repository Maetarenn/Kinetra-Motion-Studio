"""Automatic rigging and skinning for arbitrary 3D models (OBJ, GLTF, VRM).

Allows users to load ANY 3D model (e.g. Shrek, anime character, doll, monster)
without a skeleton. Automatically fits Kinetra's 52-joint skeleton, calculates
smooth anatomical bone weights (LBS), and animates it in real-time with motion capture!
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image

from .fullbody import build_skeleton, JOINT_ORDER_V2, PARENT_V2
from .hand import FINGERS
from .mathx import q_to_mat, q_rotate
from .physics_engine import fast_lbs


def _load_obj(path):
    """Parses an OBJ file, returning (verts, normals, uvs, tris, materials)."""
    verts, normals, uvs, tris = [], [], [], []
    v_map = {}
    unique_verts, unique_norms, unique_uvs = [], [], []

    raw_v, raw_vn, raw_vt = [], [], []

    mtl_filename = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == "mtllib" and len(parts) > 1:
                mtl_filename = parts[1]
            elif cmd == "v" and len(parts) >= 4:
                raw_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == "vn" and len(parts) >= 4:
                raw_vn.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == "vt" and len(parts) >= 3:
                raw_vt.append([float(parts[1]), float(parts[2])])
            elif cmd == "f" and len(parts) >= 4:
                # Triangulate face (polygon fan)
                face_verts = parts[1:]
                for i in range(1, len(face_verts) - 1):
                    tri_items = [face_verts[0], face_verts[i], face_verts[i + 1]]
                    tri_indices = []
                    for item in tri_items:
                        if item not in v_map:
                            vals = item.split("/")
                            vi = int(vals[0]) - 1 if vals[0] else 0
                            pos = raw_v[vi] if 0 <= vi < len(raw_v) else [0, 0, 0]
                            ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else -1
                            uv = raw_vt[ti] if 0 <= ti < len(raw_vt) else [0.0, 0.0]
                            ni = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else -1
                            norm = raw_vn[ni] if 0 <= ni < len(raw_vn) else None

                            idx = len(unique_verts)
                            v_map[item] = idx
                            unique_verts.append(pos)
                            unique_uvs.append(uv)
                            unique_norms.append(norm)
                        tri_indices.append(v_map[item])
                    tris.append(tri_indices)

    V = np.array(unique_verts, dtype=np.float32)
    T = np.array(tris, dtype=np.uint32)
    UV = np.array(unique_uvs, dtype=np.float32)

    # Compute normals if missing
    N = np.zeros_like(V)
    has_norms = any(n is not None for n in unique_norms)
    if has_norms:
        for i, n in enumerate(unique_norms):
            N[i] = n if n is not None else [0, 1, 0]
    else:
        # Area-weighted vertex normal computation
        p = V[T]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for k in range(3):
            np.add.at(N, T[:, k], fn)
        norm_len = np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        N /= norm_len

    # Load texture image if MTL and image exist
    img = None
    if mtl_filename:
        mtl_path = Path(path).parent / mtl_filename
        if mtl_path.exists():
            try:
                with open(mtl_path, "r", encoding="utf-8", errors="ignore") as mf:
                    for mline in mf:
                        mparts = mline.strip().split()
                        if len(mparts) > 1 and mparts[0] in ("map_Kd", "map_Ka"):
                            tex_path = Path(path).parent / mparts[1]
                            if tex_path.exists():
                                img = np.array(Image.open(tex_path).convert("RGBA"))
                                break
            except Exception:
                pass

    return V, N, UV, T, img


def _point_to_segment_dist_sq(P, A, B):
    """P: (N, 3), A: (3,), B: (3,). Returns squared distance (N,) and param t (N,)."""
    AB = B - A
    ab_len_sq = float(np.dot(AB, AB))
    if ab_len_sq < 1e-9:
        diff = P - A
        return np.sum(diff * diff, axis=1), np.zeros(len(P))
    AP = P - A
    t = np.clip(np.sum(AP * AB, axis=1) / ab_len_sq, 0.0, 1.0)
    proj = A + t[:, None] * AB
    diff = P - proj
    return np.sum(diff * diff, axis=1), t


class AutoRiggedCharacter:
    """Wraps an unrigged 3D mesh and automatically binds it to Kinetra's 52-joint skeleton."""

    def __init__(self, path):
        path = Path(path)
        self.name = path.stem
        self.path = str(path)

        if path.suffix.lower() == ".obj":
            V, N, UV, T, img = _load_obj(path)
        else:
            # Fallback loader for GLTF/GLB without skin
            from pygltflib import GLTF2
            from ..character import _read_accessor, _image_data
            g = GLTF2().load(str(path))
            blob = g.binary_blob()
            v_list, n_list, uv_list, t_list = [], [], [], []
            base = 0
            img = None
            for mesh in g.meshes:
                for prim in mesh.primitives:
                    pos = _read_accessor(g, blob, prim.attributes.POSITION)
                    nrm = _read_accessor(g, blob, prim.attributes.NORMAL) if prim.attributes.NORMAL is not None else np.zeros_like(pos)
                    uv = _read_accessor(g, blob, prim.attributes.TEXCOORD_0) if prim.attributes.TEXCOORD_0 is not None else np.zeros((len(pos), 2))
                    idx = _read_accessor(g, blob, prim.indices) if prim.indices is not None else np.arange(len(pos))
                    v_list.append(pos)
                    n_list.append(nrm)
                    uv_list.append(uv)
                    t_list.append(idx.reshape(-1, 3) + base)
                    base += len(pos)
            V = np.concatenate(v_list).astype(np.float32)
            N = np.concatenate(n_list).astype(np.float32)
            UV = np.concatenate(uv_list).astype(np.float32)
            T = np.concatenate(t_list).astype(np.uint32)

        self.V = len(V)
        self.tris = T

        # 1. Normalise, scale and fit mesh to standard humanoid dimensions
        min_p, max_p = V.min(axis=0), V.max(axis=0)
        size = max_p - min_p
        mesh_h = float(size[1])
        if mesh_h < 1e-4:
            mesh_h = 1.0

        # Build reference rest skeleton
        skel = build_skeleton()
        self.skel = skel
        self.order = list(JOINT_ORDER_V2)
        self.num_joints = len(self.order)
        self.joint_to_idx = {j: i for i, j in enumerate(self.order)}

        # Reference skeleton height
        ref_h = 1.74  # standard humanoid height ~1.74m
        scale = ref_h / mesh_h
        center_x = (min_p[0] + max_p[0]) / 2.0
        center_z = (min_p[2] + max_p[2]) / 2.0
        min_y = min_p[1]

        # Shift mesh so feet rest at Y = 0, centered on X and Z
        V = V.copy()
        V[:, 0] = (V[:, 0] - center_x) * scale
        V[:, 1] = (V[:, 1] - min_y) * scale
        V[:, 2] = (V[:, 2] - center_z) * scale

        # Calculate bone segments from reference rest skeleton
        bone_segments = self._build_bone_segments()

        # 2. Compute smooth automatic skinning weights
        self.joint_nodes, self.weights = self._compute_skinning_weights(V, bone_segments)

        self.rest_v = V.astype(np.float32)
        self.rest_n = N.astype(np.float32)
        self.uv = UV.astype(np.float32)

        # Build primitives for stage viewport rendering
        self.prims = [{
            "start": 0,
            "range": (0, len(V)),
            "tri_range": None,
            "color": [0.88, 0.88, 0.90, 1.0],
            "image": img,
            "alpha": "OPAQUE",
            "cutoff": 0.5,
            "double": False,
            "tris": self.tris
        }]

    def _build_bone_segments(self):
        """Constructs bone line segments (A, B) for all 52 joints in rest pose."""
        offsets = self.skel.offsets
        pos = {}
        # Forward kinematics for rest pose positions
        pos["Hips"] = np.array([0.0, 0.92, 0.0], dtype=np.float32)
        for j in self.order[1:]:
            p = PARENT_V2.get(j)
            parent_pos = pos.get(p, np.array([0.0, 0.92, 0.0], dtype=np.float32))
            pos[j] = parent_pos + np.array(offsets[j], dtype=np.float32)

        self.rest_joint_pos = np.array([pos[j] for j in self.order], dtype=np.float32)

        segments = []
        for i, j in enumerate(self.order):
            parent = PARENT_V2.get(j)
            p_pos = pos[parent] if parent in pos else pos[j]
            c_pos = pos[j]
            segments.append((i, j, p_pos, c_pos))
        return segments

    def _compute_skinning_weights(self, V, segments):
        """Computes anatomical distance-field skinning weights for each vertex."""
        num_v = len(V)
        all_weights = np.zeros((num_v, self.num_joints), dtype=np.float32)

        for j_idx, j_name, A, B in segments:
            dist_sq, t = _point_to_segment_dist_sq(V, A, B)
            dist = np.sqrt(dist_sq)

            # Anatomical filtering:
            # Prevent Left bones from influencing Right vertices and vice versa
            if j_name.endswith("_L"):
                dist = np.where(V[:, 0] < -0.015, dist + 10.0, dist)
            elif j_name.endswith("_R"):
                dist = np.where(V[:, 0] > 0.015, dist + 10.0, dist)

            # Prevent legs from binding to head/torso and head from binding to legs
            if "thigh" in j_name or "shin" in j_name or "foot" in j_name or "toe" in j_name:
                dist = np.where(V[:, 1] > 0.95, dist + 10.0, dist)
            if j_name in ("head", "jaw", "neck", "eye_L", "eye_R"):
                dist = np.where(V[:, 1] < 1.15, dist + 10.0, dist)

            # Smooth inverse distance weight with cubic falloff
            w = 1.0 / np.power(dist + 0.035, 3.2)
            all_weights[:, j_idx] = w

        # Top 4 bone influences per vertex
        top4_idx = np.argsort(all_weights, axis=1)[:, -4:]
        top4_w = np.take_along_axis(all_weights, top4_idx, axis=1)

        # Normalize weights so they sum to 1.0
        w_sum = top4_w.sum(axis=1, keepdims=True) + 1e-9
        top4_w /= w_sum

        return top4_idx.astype(np.int32), top4_w.astype(np.float32)

    def pose(self, quats, root, bs=None):
        """Deforms the mesh in real-time using Linear Blend Skinning (LBS)."""
        # Compute forward kinematics transform matrix for each joint
        joint_matrices = np.zeros((self.num_joints, 4, 4), dtype=np.float32)

        # World positions and rotations
        world_R = {}
        world_P = {}

        root_pos = np.asarray(root, dtype=np.float32)
        world_P["Hips"] = root_pos
        world_R["Hips"] = q_to_mat(quats.get("Hips", [1, 0, 0, 0]))

        for j in self.order[1:]:
            p = PARENT_V2.get(j)
            q_loc = quats.get(j, [1, 0, 0, 0])
            R_loc = q_to_mat(q_loc)
            p_R = world_R.get(p, np.eye(3, dtype=np.float32))
            p_P = world_P.get(p, root_pos)

            R_w = p_R @ R_loc
            P_w = p_P + p_R @ np.array(self.skel.offsets[j], dtype=np.float32)

            world_R[j] = R_w
            world_P[j] = P_w

        for i, j in enumerate(self.order):
            R = world_R.get(j, np.eye(3, dtype=np.float32))
            P = world_P.get(j, root_pos)
            P0 = self.rest_joint_pos[i]

            # Skinning matrix M = [R | P - R @ P0]
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R
            M[:3, 3] = P - R @ P0
            joint_matrices[i] = M

        # Deform vertices and normals using C++ accelerated LBS
        return fast_lbs(
            self.rest_v,
            self.rest_n,
            joint_matrices,
            self.joint_nodes,
            self.weights
        )

    def vertex_colors(self):
        """Generates vertex colours for previews."""
        col = np.ones((self.V, 3), np.float32) * 0.82
        for pr in self.prims:
            img = pr["image"]
            if img is not None:
                h, w = img.shape[:2]
                u = np.clip((self.uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
                v = np.clip((self.uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
                col[:] = img[v, u, :3].astype(np.float32) / 255.0
        return col
