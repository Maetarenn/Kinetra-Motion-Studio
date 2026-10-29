"""File and network output: BVH, USD (ASCII), CSV blendshape curves, UDP/JSON and OSC streaming."""
import csv
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

from .constants import JOINT_ORDER, JOINT_PARENT, BLENDSHAPE_NAMES, skeleton_offsets
from .fullbody import Skeleton
from .mathx import q_to_euler_zxy, q_norm


# ----------------------------------------------------------------------------------------------
def resample_motion(times, root, quats, fps):
    """Uniform re-timing. times (n,), root (n,3), quats (n, J, 4) -> new arrays at the given fps (nlerp)."""
    times = np.asarray(times, float)
    t_new = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(times, t_new, side="right") - 1, 0, len(times) - 2)
    w = ((t_new - times[idx]) / np.maximum(times[idx + 1] - times[idx], 1e-9))[:, None]
    r = root[idx] * (1 - w) + root[idx + 1] * w
    qa, qb = quats[idx], quats[idx + 1]
    sign = np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    q = qa * (1 - w[:, :, None]) + qb * sign * w[:, :, None]
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return t_new, r, q


def quats_to_array(quat_dicts, order=JOINT_ORDER):
    return np.array([[q[n] for n in order] for q in quat_dicts], dtype=np.float64)


def array_to_quats(arr, order=JOINT_ORDER):
    return [{n: arr[i, j] for j, n in enumerate(order)} for i in range(len(arr))]


def _v1_skeleton(lengths):
    off, end = skeleton_offsets(lengths)
    return Skeleton(tuple(JOINT_ORDER), dict(JOINT_PARENT), off, end)


def _prepare(times, root, quats, fps):
    times, root, qa = np.asarray(times, float), np.asarray(root, float), np.asarray(quats, float)
    if fps is None:
        fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
        fps = float(min(max(fps, 1.0), 240.0))
    times, root, qa = resample_motion(times, root, qa, fps)
    return times, root, qa, fps


# ----------------------------------------------------------------------------------------------
def write_bvh(path, times, root, quats, lengths, fps=None, skel=None):
    """root: (n,3) metres, quats: (n, J, 4) in skel.order (default: the 13-joint skeleton). Positions in cm."""
    if len(times) < 2:
        return None
    times, root, qa, fps = _prepare(times, root, quats, fps)
    skel = skel or _v1_skeleton(lengths)
    off, end, order = skel.offsets, skel.ends, skel.order
    kids = {n: [c for c in order if skel.parent.get(c) == n] for n in order}
    out = ["HIERARCHY"]

    def emit(n, depth):
        pad = "\t" * depth
        out.append(f"{pad}{'ROOT' if n == 'Hips' else 'JOINT'} {n}")
        out.append(pad + "{")
        o = [v * 100 for v in off[n]]
        out.append(f"{pad}\tOFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}")
        out.append(pad + ("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if n == "Hips"
                          else "\tCHANNELS 3 Zrotation Xrotation Yrotation"))
        for c in kids[n]:
            emit(c, depth + 1)
        if n in end:
            e = [v * 100 for v in end[n]]
            out.extend([f"{pad}\tEnd Site", pad + "\t{", f"{pad}\t\tOFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}", pad + "\t}"])
        out.append(pad + "}")

    emit("Hips", 0)
    eul = np.array([[q_to_euler_zxy(qa[i, j]) for j in range(qa.shape[1])] for i in range(qa.shape[0])])
    eul = np.degrees(np.unwrap(np.radians(eul), axis=0))             # no 360 degree jumps between frames
    out += ["MOTION", f"Frames: {len(times)}", f"Frame Time: {1.0 / fps:.6f}"]
    for i in range(len(times)):
        row = list(root[i] * 100.0) + list(eul[i].reshape(-1))
        out.append(" ".join(f"{v:.4f}" for v in row))
    Path(path).write_text("\n".join(out) + "\n")
    return str(path)


# ----------------------------------------------------------------------------------------------
def _mat_translate(v):
    return f"( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, 1) )"


def write_usda(path, times, root, quats, lengths, fps=None, name="Kinetra", skel=None):
    """USD skeletal animation (UsdSkel), metres, Y-up. Opens in Blender, Houdini, Maya, Unreal, usdview."""
    if len(times) < 2:
        return None
    times, root, qa, fps = _prepare(times, root, quats, fps)
    skel = skel or _v1_skeleton(lengths)
    off, order = skel.offsets, skel.order
    paths = {}
    for n in order:
        paths[n] = n if n == "Hips" else paths[skel.parent[n]] + "/" + n
    world = {}
    for n in order:
        w = np.array(off[n], float)
        world[n] = w if n == "Hips" else world[skel.parent[n]] + w
    joints = ", ".join(f'"{paths[n]}"' for n in order)
    L = [
        "#usda 1.0", "(", f'    defaultPrim = "{name}"', "    metersPerUnit = 1", '    upAxis = "Y"',
        "    startTimeCode = 1", f"    endTimeCode = {len(times)}", f"    timeCodesPerSecond = {fps:.4f}", ")", "",
        f'def SkelRoot "{name}"', "{", '    def Skeleton "Skel" (', '        prepend apiSchemas = ["SkelBindingAPI"]', "    )", "    {",
        f"        uniform token[] joints = [{joints}]",
        "        uniform matrix4d[] bindTransforms = [" + ", ".join(_mat_translate(world[n]) for n in order) + "]",
        "        uniform matrix4d[] restTransforms = [" + ", ".join(_mat_translate(off[n]) for n in order) + "]",
        f"        rel skel:animationSource = </{name}/Skel/Anim>",
        '        def SkelAnimation "Anim"', "        {",
        f"            uniform token[] joints = [{joints}]",
        "            half3[] scales = [" + ", ".join(["(1, 1, 1)"] * len(order)) + "]",
        "            float3[] translations.timeSamples = {",
    ]
    for i in range(len(times)):
        tr = [tuple(root[i])] + [tuple(off[n]) for n in order[1:]]
        L.append(f"                {i + 1}: [" + ", ".join(f"({a:.5f}, {b:.5f}, {c:.5f})" for a, b, c in tr) + "],")
    L += ["            }", "            quatf[] rotations.timeSamples = {"]
    for i in range(len(times)):
        L.append(f"                {i + 1}: [" + ", ".join(
            f"({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})" for q in qa[i]) + "],")
    L += ["            }", "        }", "    }", "}"]
    Path(path).write_text("\n".join(L) + "\n")
    return str(path)


def write_curves_csv(path, times, curves):
    """curves: (n, 52) blendshape weights."""
    if len(times) < 2:
        return None
    t0 = times[0]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Timestamp"] + BLENDSHAPE_NAMES)
        for t, row in zip(times, curves):
            wr.writerow([round(float(t - t0), 4)] + [round(float(v), 4) for v in row])
    return str(path)


# ----------------------------------------------------------------------------------------------
def _osc_str(s):
    b = s.encode() + b"\0"
    return b + b"\0" * (-len(b) % 4)


def osc_message(addr, floats):
    return _osc_str(addr) + _osc_str("," + "f" * len(floats)) + struct.pack(">" + "f" * len(floats), *floats)


def osc_bundle(messages):
    out = b"#bundle\0" + struct.pack(">II", 0, 1)
    for m in messages:
        out += struct.pack(">i", len(m)) + m
    return out


# Mapping from Kinetra 52 bones to Unreal Engine 5 MetaHuman / Mannequin standard bone names
KINETRA_TO_UE5_BONES = {
    'Hips': 'pelvis', 'spine': 'spine_01', 'chest': 'spine_02', 'neck': 'neck_01', 'head': 'head', 'jaw': 'jaw',
    'upper_arm_L': 'upperarm_l', 'forearm_L': 'lowerarm_l', 'hand_L': 'hand_l',
    'upper_arm_R': 'upperarm_r', 'forearm_R': 'lowerarm_r', 'hand_R': 'hand_r',
    'thigh_L': 'thigh_l', 'shin_L': 'calf_l', 'foot_L': 'foot_l', 'toe_L': 'ball_l',
    'thigh_R': 'thigh_r', 'shin_R': 'calf_r', 'foot_R': 'foot_r', 'toe_R': 'ball_r',
    'thumb_1_L': 'thumb_01_l', 'thumb_2_L': 'thumb_02_l', 'thumb_3_L': 'thumb_03_l',
    'index_1_L': 'index_01_l', 'index_2_L': 'index_02_l', 'index_3_L': 'index_03_l',
    'middle_1_L': 'middle_01_l', 'middle_2_L': 'middle_02_l', 'middle_3_L': 'middle_03_l',
    'ring_1_L': 'ring_01_l', 'ring_2_L': 'ring_02_l', 'ring_3_L': 'ring_03_l',
    'pinky_1_L': 'pinky_01_l', 'pinky_2_L': 'pinky_02_l', 'pinky_3_L': 'pinky_03_l',
    'thumb_1_R': 'thumb_01_r', 'thumb_2_R': 'thumb_02_r', 'thumb_3_R': 'thumb_03_r',
    'index_1_R': 'index_01_r', 'index_2_R': 'index_02_r', 'index_3_R': 'index_03_r',
    'middle_1_R': 'middle_01_r', 'middle_2_R': 'middle_02_r', 'middle_3_R': 'middle_03_r',
    'ring_1_R': 'ring_01_r', 'ring_2_R': 'ring_02_r', 'ring_3_R': 'ring_03_r',
    'pinky_1_R': 'pinky_01_r', 'pinky_2_R': 'pinky_02_r', 'pinky_3_R': 'pinky_03_r',
}

class Streamer:
    """Real-time output of bone quaternions and blendshapes as UDP/JSON or OSC."""

    def __init__(self):
        self.enabled, self.fmt, self.addr = False, "json", ("127.0.0.1", 9000)
        self._sock = None

    def configure(self, host="127.0.0.1", port=9000, fmt="json"):
        self.addr, self.fmt = (host, int(port)), fmt
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def send(self, idx, root, quats, faces, prop=None):
        """Sends real-time frame to Unreal Engine 5 LiveLink, VMC, OSC or JSON."""
        if not self.enabled or self._sock is None or quats is None:
            return
        try:
            if self.fmt == "ue5_livelink":
                # UE5 Coordinate transform: Kinetra (Y-up, Z-fwd in metres) -> Unreal Engine (Z-up, X-fwd, Y-right in cm)
                ue_root = [round(float(root[2] * 100.0), 3), round(float(root[0] * 100.0), 3), round(float(root[1] * 100.0), 3)]
                ue_bones = {}
                for k, q in quats.items():
                    ue_name = KINETRA_TO_UE5_BONES.get(k, k)
                    ue_bones[ue_name] = [round(float(q[3]), 6), round(float(q[1]), 6), round(float(q[2]), 6), round(float(q[0]), 6)]
                if prop is not None and getattr(prop, 'quat', None) is not None:
                    pq = prop.quat
                    ue_bones['weapon_r' if getattr(prop, 'side', 'R') == 'R' else 'weapon_l'] = [
                        round(float(pq[3]), 6), round(float(pq[1]), 6), round(float(pq[2]), 6), round(float(pq[0]), 6)
                    ]
                bs_dict = {}
                if faces and len(faces) > 0:
                    for i, name in enumerate(BLENDSHAPE_NAMES):
                        v = float(faces[0][i])
                        bs_dict[name] = round(v, 4)
                        bs_dict[name[0].upper() + name[1:]] = round(v, 4)
                pkt = {
                    'version': 1, 'subject': 'Kinetra_Actor', 'frame': idx, 'time': time.time(),
                    'root': ue_root, 'bones': ue_bones, 'blendshapes': bs_dict
                }
                self._sock.sendto(json.dumps(pkt).encode('utf-8'), self.addr)
            elif self.fmt == "vmc":
                msgs = [osc_message('/VMC/Ext/Root/Pos', [float(root[0]), float(root[1]), float(root[2]), 0.0, 0.0, 0.0, 1.0])]
                for k, q in quats.items():
                    msgs.append(osc_message(f'/VMC/Ext/Bone/{k}', [0.0, 0.0, 0.0, float(q[1]), float(q[2]), float(q[3]), float(q[0])]))
                if faces and len(faces) > 0:
                    for i, name in enumerate(BLENDSHAPE_NAMES[:24]):
                        msgs.append(osc_message(f'/VMC/Ext/Blend/{name}', [float(faces[0][i])]))
                msgs.append(osc_message('/VMC/Ext/OK', [1.0]))
                self._sock.sendto(osc_bundle(msgs), self.addr)
            elif self.fmt == "osc":
                msgs = [osc_message("/kinetra/root", [float(v) for v in root])]
                msgs += [osc_message(f"/kinetra/bone/{k}", [float(x) for x in q]) for k, q in quats.items()]
                msgs += [osc_message(f"/kinetra/face/{i}", [float(x) for x in bs]) for i, bs in enumerate(faces)]
                self._sock.sendto(osc_bundle(msgs), self.addr)
            else:
                pkt = {"frame": idx, "t": time.time(), "root": [round(float(v), 5) for v in root],
                       "bones": {k: [round(float(x), 6) for x in q] for k, q in quats.items()},
                       "faces": [{k: round(float(v), 4) for k, v in zip(BLENDSHAPE_NAMES, bs)} for bs in faces]}
                if faces:
                    pkt["blendshapes"] = pkt["faces"][0]
                self._sock.sendto(json.dumps(pkt).encode("utf-8"), self.addr)
        except OSError:
            pass


def write_points_usd(path, times, points, tris, name="FaceFeatures", fps=None):
    """Animated point cache (UsdGeom Mesh with time-sampled points). points: (n, V, 3) metres, tris: (F,3) indices
    into the V points. This is the face-performance asset a VFX pipeline retargets onto a CG character."""
    times = np.asarray(times, float)
    if len(times) < 2:
        return None
    fps = fps or float(min(max((len(times) - 1) / max(times[-1] - times[0], 1e-6), 1.0), 240.0))
    tris = np.asarray(tris, dtype=int)
    L = ["#usda 1.0", "(", f'    defaultPrim = "{name}"', "    metersPerUnit = 1", '    upAxis = "Y"', "    startTimeCode = 1",
         f"    endTimeCode = {len(times)}", f"    timeCodesPerSecond = {fps:.4f}", ")", "", f'def Mesh "{name}"', "{",
         "    int[] faceVertexCounts = [" + ", ".join(["3"] * len(tris)) + "]",
         "    int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in tris.reshape(-1)) + "]",
         '    uniform token subdivisionScheme = "none"', "    point3f[] points.timeSamples = {"]
    for i in range(len(times)):
        L.append(f"        {i + 1}: [" + ", ".join(f"({a:.5f}, {b:.5f}, {c:.5f})" for a, b, c in points[i]) + "],")
    L += ["    }", "}"]
    Path(path).write_text("\n".join(L) + "\n")
    return str(path)
