"""Import any rigged humanoid (.glb / .gltf / .vrm, e.g. Mixamo, Ready Player Me, VRoid) and drive it with Kinetra's
52-joint pose.

How the retarget works
  * Bones are mapped by name (or by the VRM humanoid table).
  * Every mapped character bone gets a constant alignment rotation D that takes the character's rest pose (T-pose,
    A-pose ...) onto Kinetra's rest pose (arms hanging down). D is built from the bone direction and the forward
    direction, so palms, knuckles and feet end up with the right roll.
  * Per frame: world delta of the bone = Kinetra world rotation (x) D. Unmapped bones (twist bones, clavicles, hair)
    inherit their parent's delta. Positions follow by walking the character's own hierarchy, so proportions of the
    character are kept; the root translation is scaled by the height ratio.
  * Skinning is linear blend skinning in rest-world space: v' = sum w_i (R_i (v - p_i) + p'_i).
  * Morph targets named like the 52 ARKit shapes are driven by the live face blendshapes.
"""
import re
from pathlib import Path

import numpy as np

from .character import arkit_index, _read_accessor, _node_matrix
from .core.constants import NBS
from .core.fullbody import JOINT_ORDER_V2, PARENT_V2, build_skeleton
from .core.hand import FINGERS
from .core.mathx import unit, q_mul, q_rotate, q_to_mat, mat_to_quat, IDENTITY_Q


# ----------------------------------------------------------------------------------------------
def _norm(name):
    n = name.split(":")[-1].lower()
    n = n.replace("mixamorig", "")
    n = re.sub(r"bip\d*", "", n)
    n = re.sub(r"bones", "", n)
    n = re.sub(r"_\d+$", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def _side_names(base, side):
    full, ab = ("left", "l") if side == "L" else ("right", "r")
    return [full + base, base + ab, ab + base, base + full, f"jbip{ab}{base}", f"{full}hand{base}", f"{ab}hand{base}"]


_BASES = {"upper_arm": ["arm", "upperarm", "shoulder_"], "forearm": ["forearm", "lowerarm"], "hand": ["hand"],
          "thigh": ["upleg", "upperleg", "thigh"], "shin": ["leg", "lowerleg", "shin", "calf"],
          "foot": ["foot"], "toe": ["toebase", "toes", "toe", "toe0"], "eye": ["eye"]}


def name_candidates():
    """our joint -> list of normalised character bone names."""
    c = {"Hips": ["hips", "pelvis", "hip"], "spine": ["spine", "spine0"], "chest": ["spine1", "chest"],
         "neck": ["neck", "neck1"], "head": ["head"], "jaw": ["jaw"]}
    FINGER_MAP = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
    for s in "LR":
        for ours, bases in _BASES.items():
            key = "eye_" + s if ours == "eye" else f"{ours}_{s}"
            lst = []
            for b in bases:
                lst += _side_names(b, s)
            c[key] = lst
        for f in FINGERS:
            for i in (1, 2, 3):
                lst = []
                for fmt in (f"{f}{i}",):
                    lst += _side_names(fmt, s)
                f_idx = FINGER_MAP.get(f)
                if f_idx is not None:
                    suffix = "" if i == 1 else str(i - 1)
                    lst += [f"{s.lower()}finger{f_idx}{suffix}", f"finger{f_idx}{suffix}{s.lower()}"]
                c[f"{f}_{i}_{s}"] = lst
    return c


# VRM humanoid names -> our joints
_VRM = {"hips": "Hips", "spine": "spine", "chest": "chest", "neck": "neck", "head": "head", "jaw": "jaw"}
for _s, _p in (("L", "left"), ("R", "right")):
    _VRM.update({f"{_p}UpperArm": f"upper_arm_{_s}", f"{_p}LowerArm": f"forearm_{_s}", f"{_p}Hand": f"hand_{_s}",
                 f"{_p}UpperLeg": f"thigh_{_s}", f"{_p}LowerLeg": f"shin_{_s}", f"{_p}Foot": f"foot_{_s}",
                 f"{_p}Toes": f"toe_{_s}", f"{_p}Eye": f"eye_{_s}"})
    for _f in ("Thumb", "Index", "Middle", "Ring", "Little"):
        ours = "pinky" if _f == "Little" else _f.lower()
        for _i, _n in enumerate(("Metacarpal", "Proximal", "Intermediate", "Distal") if _f == "Thumb" else ("Proximal", "Intermediate", "Distal")):
            idx = _i if _f == "Thumb" else _i + 1
            if _f == "Thumb":                       # VRM1 thumb has 4 bones, VRM0 has 3 (no metacarpal)
                continue
            _VRM[f"{_p}{_f}{_n}"] = f"{ours}_{idx}_{_s}"
        _VRM[f"{_p}ThumbProximal"] = f"thumb_1_{_s}"
        _VRM[f"{_p}ThumbIntermediate"] = f"thumb_2_{_s}"
        _VRM[f"{_p}ThumbDistal"] = f"thumb_3_{_s}"


def map_bones(names, vrm_humanoid=None):
    """{our joint: node index}."""
    out = {}
    if vrm_humanoid:
        for hb, node in vrm_humanoid.items():
            if hb in _VRM:
                out[_VRM[hb]] = node
    norm = {}
    for i, n in enumerate(names):
        norm.setdefault(_norm(n), i)
    for joint, cands in name_candidates().items():
        if joint in out:
            continue
        for c in cands:
            if c in norm:
                out[joint] = norm[c]
                break
    return out


# ----------------------------------------------------------------------------------------------
AIM = {"hand_L": "middle_1_L", "hand_R": "middle_1_R", "foot_L": "toe_L", "foot_R": "toe_R"}


def _basis(r, fwd):
    r = unit(r)
    f = np.asarray(fwd, float)
    if abs(float(np.dot(r, f))) > 0.9:
        f = np.array([0.0, 1.0, 0.0])
    f = unit(f - r * float(np.dot(f, r)))
    return np.stack([r, f, np.cross(r, f)], axis=1)


class CharacterRig:
    """Retarget math without any file format: node tree + rest positions + bone map."""

    def __init__(self, parent, rest_pos, bone_map, skel=None, forward=(0.0, 0.0, 1.0), hips_height=None):
        self.skel = skel or build_skeleton()
        self.parent = list(parent)
        self.rest_pos = np.asarray(rest_pos, float)
        self.n = len(self.parent)
        self.map = {j: n for j, n in bone_map.items() if j in self.skel.parent or j == "Hips"}
        self.node_joint = {n: j for j, n in self.map.items()}
        self.children = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p is not None and p >= 0:
                self.children[p].append(i)
        self.order = self._topo()
        self._build_alignment(forward)
        legs = [self.rest_pos[self.map[k]] for k in ("Hips",) if k in self.map]
        self.hips_h = float(self.rest_pos[self.map["Hips"]][1]) if "Hips" in self.map else 1.0
        sk = self.skel
        self.ours_h = float(np.linalg.norm(sk.offsets["shin_L"]) + np.linalg.norm(sk.offsets["foot_L"]) + 0.08)
        self.scale = self.hips_h / self.ours_h if self.hips_h > 0.2 else 1.0

    def _topo(self):
        seen, out = set(), []

        def visit(i):
            if i in seen:
                return
            p = self.parent[i]
            if p is not None and p >= 0:
                visit(p)
            seen.add(i)
            out.append(i)
        for i in range(self.n):
            visit(i)
        return out

    def _our_aim(self, joint):
        sk = self.skel
        kid = AIM.get(joint)
        if kid is None:
            kids = [c for c in sk.order if sk.parent.get(c) == joint]
            kid = kids[0] if kids else None
        if kid is not None:
            return np.array(sk.offsets[kid], float), kid
        end = sk.ends.get(joint)
        return (np.array(end, float), None) if end is not None else (None, None)

    def _build_alignment(self, forward):
        self.delta = {}
        for node, joint in self.node_joint.items():
            r_o, kid = self._our_aim(joint)
            if r_o is None:
                self.delta[node] = IDENTITY_Q.copy()
                continue
            aim_node = self.map.get(kid) if kid else None
            if aim_node is None and self.children[node]:
                aim_node = self.children[node][0]
            if aim_node is None:
                p = self.parent[node]
                self.delta[node] = self.delta[p].copy() if (p is not None and p in self.delta) else IDENTITY_Q.copy()
                continue
            r_c = self.rest_pos[aim_node] - self.rest_pos[node]
            if np.linalg.norm(r_c) < 1e-9:
                p = self.parent[node]
                self.delta[node] = self.delta[p].copy() if (p is not None and p in self.delta) else IDENTITY_Q.copy()
                continue
            B_c, B_o = _basis(r_c, forward), _basis(r_o, (0.0, 0.0, 1.0))
            self.delta[node] = mat_to_quat(B_o @ B_c.T)

    def pose(self, quats, root):
        """quats: our local quaternions (JOINT_ORDER_V2), root: our hips position. Returns (Rd (N,3,3), p' (N,3))."""
        W = {}
        for j in self.skel.order:
            p = PARENT_V2.get(j)
            W[j] = quats[j] if p is None else q_mul(W[p], quats[j])
        Rd = [None] * self.n
        pos = np.zeros((self.n, 3))
        hips_node = self.map.get("Hips")
        for i in self.order:
            p = self.parent[i]
            j = self.node_joint.get(i)
            rd_q = q_mul(W[j], self.delta[i]) if j is not None else (Rd[p] if p is not None and p >= 0 else IDENTITY_Q)
            Rd[i] = rd_q
            if i == hips_node:
                pos[i] = self.rest_pos[i] + self.scale * (np.asarray(root, float) - np.array([0.0, self.ours_h, 0.0]))
            elif p is not None and p >= 0:
                pos[i] = pos[p] + q_rotate(Rd[p], self.rest_pos[i] - self.rest_pos[p])
            else:
                pos[i] = self.rest_pos[i]
        return np.array([q_to_mat(q) for q in Rd]), pos


# ----------------------------------------------------------------------------------------------
class SkinnedCharacter:
    def __init__(self, path):
        p = Path(path)
        if p.suffix.lower() == ".obj":
            from .core.auto_rigger import AutoRiggedCharacter
            self._impl = AutoRiggedCharacter(path)
            self.__dict__.update(self._impl.__dict__)
            return

        from pygltflib import GLTF2
        g = GLTF2().load_binary(str(p)) if p.suffix.lower() in (".glb", ".vrm") else GLTF2().load(str(p))
        self.name, self.g = p.name, g
        blob = g.binary_blob()
        N = len(g.nodes)
        parent = [-1] * N
        for i, n in enumerate(g.nodes):
            for c in n.children or []:
                parent[c] = i
        world = [None] * N

        def W(i):
            if world[i] is None:
                M = _node_matrix(g.nodes[i])
                world[i] = M if parent[i] < 0 else W(parent[i]) @ M
            return world[i]
        for i in range(N):
            W(i)
        rest_pos = np.array([w[:3, 3] for w in world])
        vrm_h, forward = self._vrm(g)
        names = [n.name or f"node{i}" for i, n in enumerate(g.nodes)]
        self.bone_map = map_bones(names, vrm_h)
        if "Hips" not in self.bone_map or not getattr(g, "skins", None):
            # Fall back to AutoRigger if model has no skin or humanoid bones
            from .core.auto_rigger import AutoRiggedCharacter
            self._impl = AutoRiggedCharacter(path)
            self.__dict__.update(self._impl.__dict__)
            return
        self.rig = CharacterRig(parent, rest_pos, self.bone_map, forward=forward)
        self._load_meshes(g, blob, world, parent)

    @staticmethod
    def _vrm(g):
        ext = g.extensions or {}
        if "VRMC_vrm" in ext:                               # VRM 1.x faces +Z
            hb = ext["VRMC_vrm"].get("humanoid", {}).get("humanBones", {})
            return {k: v["node"] for k, v in hb.items()}, (0.0, 0.0, 1.0)
        if "VRM" in ext:                                    # VRM 0.x faces -Z
            hb = ext["VRM"].get("humanoid", {}).get("humanBones", [])
            return {b["bone"]: b["node"] for b in hb}, (0.0, 0.0, -1.0)
        return None, (0.0, 0.0, 1.0)

    def _image(self, g, blob, tex_index):
        import cv2
        if tex_index is None:
            return None
        src = g.textures[tex_index].source
        im = g.images[src]
        if im.bufferView is None:
            return None
        bv = g.bufferViews[im.bufferView]
        data = np.frombuffer(blob, np.uint8, bv.byteLength, bv.byteOffset or 0)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    def _load_meshes(self, g, blob, world, parent):
        V, Nrm, UV, J, Wt, tris, prims, morph = [], [], [], [], [], [], [], []
        base = 0
        skin_cache = {}
        n_joint_slots = 0
        for ni, node in enumerate(g.nodes):
            if node.mesh is None:
                continue
            skin = g.skins[node.skin] if node.skin is not None else None
            if skin is not None and node.skin not in skin_cache:
                ibm = _read_accessor(g, blob, skin.inverseBindMatrices).reshape(-1, 4, 4).astype(np.float64) \
                    if skin.inverseBindMatrices is not None else np.tile(np.eye(4), (len(skin.joints), 1, 1))
                skin_cache[node.skin] = (list(skin.joints), np.transpose(ibm, (0, 2, 1)))      # glTF is column-major
            for prim in g.meshes[node.mesh].primitives:
                a = prim.attributes
                pos = _read_accessor(g, blob, a.POSITION).astype(np.float64)
                nrm = _read_accessor(g, blob, a.NORMAL).astype(np.float64) if a.NORMAL is not None else np.zeros_like(pos)
                uv = _read_accessor(g, blob, a.TEXCOORD_0).astype(np.float32) if a.TEXCOORD_0 is not None else np.zeros((len(pos), 2), np.float32)
                idx = (_read_accessor(g, blob, prim.indices).reshape(-1).astype(np.int64) if prim.indices is not None
                       else np.arange(len(pos)))
                if skin is not None and a.JOINTS_0 is not None:
                    joints_nodes, ibm = skin_cache[node.skin]
                    jj = _read_accessor(g, blob, a.JOINTS_0).astype(np.int64)
                    ww = _read_accessor(g, blob, a.WEIGHTS_0).astype(np.float64)
                    ww /= np.maximum(ww.sum(axis=1, keepdims=True), 1e-9)
                    mats = np.array([world[n] @ ibm[k] for k, n in enumerate(joints_nodes)])        # (J,4,4)
                    ph = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
                    wpos = np.zeros_like(pos)
                    lin = np.zeros((len(pos), 3, 3))
                    wnrm = np.zeros_like(pos)
                    for k in range(4):
                        M = mats[jj[:, k]]
                        wpos += ww[:, k:k + 1] * np.einsum("nij,nj->ni", M, ph)[:, :3]
                        lin += ww[:, k, None, None] * M[:, :3, :3]
                        wnrm += ww[:, k:k + 1] * np.einsum("nij,nj->ni", M[:, :3, :3], nrm)
                    node_ids = np.array(joints_nodes)[jj]                                           # joint slots -> node ids
                else:                                                                               # rigid / static mesh
                    M = world[ni]
                    wpos = (np.concatenate([pos, np.ones((len(pos), 1))], axis=1) @ M.T)[:, :3]
                    lin = np.tile(M[:3, :3], (len(pos), 1, 1))
                    wnrm = nrm @ M[:3, :3].T
                    node_ids = np.full((len(pos), 4), -1)
                    ww = np.zeros((len(pos), 4))
                    ww[:, 0] = 1.0
                    anc = ni
                    while anc >= 0 and anc not in self.rig.node_joint:
                        anc = parent[anc]
                    node_ids[:, 0] = anc
                wnrm /= np.linalg.norm(wnrm, axis=1, keepdims=True) + 1e-12
                V.append(wpos)
                Nrm.append(wnrm)
                UV.append(uv)
                J.append(node_ids)
                Wt.append(ww)
                tris.append(idx.reshape(-1, 3) + base)
                # morph targets (ARKit names) -> world-space deltas
                deltas = {}
                names = None
                for ex in (getattr(prim, "extras", None), getattr(g.meshes[node.mesh], "extras", None)):
                    if isinstance(ex, dict) and ex.get("targetNames"):
                        names = ex["targetNames"]
                        break
                for ti, tgt in enumerate(prim.targets or []):
                    k = arkit_index(names[ti]) if names and ti < len(names) else None
                    if k is not None and "POSITION" in tgt:
                        d = _read_accessor(g, blob, tgt["POSITION"]).astype(np.float64)
                        deltas[k] = np.einsum("nij,nj->ni", lin, d)
                morph.append((base, len(pos), deltas))
                mat = g.materials[prim.material] if prim.material is not None else None
                pbr = mat.pbrMetallicRoughness if mat is not None else None
                tex = pbr.baseColorTexture.index if pbr is not None and pbr.baseColorTexture is not None else None
                prims.append({"start": int(np.concatenate(tris[-1:]).size * 0 + 0), "range": (base, base + len(pos)),
                              "tri_range": None, "color": list(pbr.baseColorFactor) if pbr is not None and pbr.baseColorFactor else [1, 1, 1, 1],
                              "image": self._image(g, blob, tex), "alpha": (mat.alphaMode if mat else "OPAQUE"),
                              "cutoff": (mat.alphaCutoff if mat and mat.alphaCutoff is not None else 0.5),
                              "double": bool(mat.doubleSided) if mat else False, "tris": idx.reshape(-1, 3) + base})
                base += len(pos)
        self.rest_v = np.concatenate(V).astype(np.float32)
        self.rest_n = np.concatenate(Nrm).astype(np.float32)
        self.uv = np.concatenate(UV)
        self.joint_nodes = np.concatenate(J)
        self.weights = np.concatenate(Wt).astype(np.float32)
        self.prims = prims
        self.tris = np.concatenate(tris).astype(np.uint32)
        D = np.zeros((NBS, base, 3), np.float32)
        self.matched = set()
        for off, n, deltas in morph:
            for k, d in deltas.items():
                D[k, off:off + n] = d
                self.matched.add(k)
        self.morph = D.reshape(NBS, -1) if self.matched else None
        self.V = base
        self.node_pos = self.rig.rest_pos.astype(np.float32)

    # -- per frame --------------------------------------------------------------------------------
    def pose(self, quats, root, bs=None):
        if hasattr(self, "_impl"):
            return self._impl.pose(quats, root, bs)
        Rd, pn = self.rig.pose(quats, root)
        v0 = self.rest_v
        if self.morph is not None and bs is not None:
            v0 = v0 + (np.asarray(bs, np.float32) @ self.morph).reshape(-1, 3)
        out = np.zeros_like(v0)
        nout = np.zeros_like(v0)
        for k in range(4):
            nid = self.joint_nodes[:, k]
            w = self.weights[:, k:k + 1]
            valid = (nid >= 0)
            nid = np.where(valid, nid, 0)
            R = Rd[nid].astype(np.float32)
            p = self.node_pos[nid]
            d = v0 - p
            moved = np.einsum("nij,nj->ni", R, d) + pn[nid].astype(np.float32)
            out += w * np.where(valid[:, None], moved, v0)
            nout += w * np.where(valid[:, None], np.einsum("nij,nj->ni", R, self.rest_n), self.rest_n)
        nout /= np.linalg.norm(nout, axis=1, keepdims=True) + 1e-9
        # Ground contact constraint: prevent feet and knees from sinking below ground (y < 0)
        min_y = float(np.min(out[:, 1]))
        if min_y < 0.0:
            out[:, 1] -= min_y
        return out, nout

    def vertex_colors(self):
        """Approximate per-vertex colours (texture sampled at the UVs) for software previews."""
        if hasattr(self, "_impl"):
            return self._impl.vertex_colors()
        col = np.ones((self.V, 3), np.float32) * 0.7
        for pr in self.prims:
            a, b = pr["range"]
            c = np.array(pr["color"][:3], np.float32)
            img = pr["image"]
            if img is not None:
                h, w = img.shape[:2]
                u = np.clip((self.uv[a:b, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
                v = np.clip((self.uv[a:b, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
                col[a:b] = img[v, u, :3].astype(np.float32) / 255.0 * c
            else:
                col[a:b] = c
        return col
