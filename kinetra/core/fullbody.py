"""Skeleton v2 (52 joints): Hips, 2-segment spine, neck, head, jaw, eyes, arms with hands and 15 finger joints
per hand, legs with feet and toes. Builds rest offsets, assembles quaternions from the body solver, the hand
solvers and the face blendshapes, and runs forward kinematics."""
import math
from dataclasses import dataclass, field

import numpy as np

from .constants import BS_INDEX, DEFAULT_LENGTHS
from .hand import FINGER_JOINTS, HandSolver
from .mathx import q_mul, q_conj, q_axis_angle, q_rotate, q_to_mat, IDENTITY_Q

NECK_LEN = 0.09
HEAD_TOP = 0.14
SIDES = ("L", "R")


def _arm(s):
    return ["upper_arm_" + s, "forearm_" + s, "hand_" + s] + [f"{j}_{s}" for j in FINGER_JOINTS]


def _leg(s):
    return ["thigh_" + s, "shin_" + s, "foot_" + s, "toe_" + s]


JOINT_ORDER_V2 = tuple(["Hips", "spine", "chest", "neck", "head", "jaw", "eye_L", "eye_R"]
                       + _arm("L") + _arm("R") + _leg("L") + _leg("R"))

PARENT_V2 = {"spine": "Hips", "chest": "spine", "neck": "chest", "head": "neck", "jaw": "head",
             "eye_L": "head", "eye_R": "head"}
for _s in SIDES:
    PARENT_V2.update({"upper_arm_" + _s: "chest", "forearm_" + _s: "upper_arm_" + _s, "hand_" + _s: "forearm_" + _s,
                      "thigh_" + _s: "Hips", "shin_" + _s: "thigh_" + _s, "foot_" + _s: "shin_" + _s,
                      "toe_" + _s: "foot_" + _s})
    for _f in ("thumb", "index", "middle", "ring", "pinky"):
        PARENT_V2[f"{_f}_1_{_s}"] = "hand_" + _s
        PARENT_V2[f"{_f}_2_{_s}"] = f"{_f}_1_{_s}"
        PARENT_V2[f"{_f}_3_{_s}"] = f"{_f}_2_{_s}"


@dataclass
class Skeleton:
    order: tuple
    parent: dict
    offsets: dict
    ends: dict = field(default_factory=dict)

    def to_meta(self):
        return {"order": list(self.order), "offsets": {k: list(map(float, v)) for k, v in self.offsets.items()},
                "ends": {k: list(map(float, v)) for k, v in self.ends.items()}}

    @classmethod
    def from_meta(cls, m):
        order = tuple(m["order"])
        return cls(order, {j: PARENT_V2[j] for j in order if j in PARENT_V2},
                   {k: tuple(v) for k, v in m["offsets"].items()}, {k: tuple(v) for k, v in m["ends"].items()})


def build_skeleton(L=None, hands=None):
    """Rest-pose skeleton in metres (y up, +x = character's left). hands: {"L": HandSolver, "R": HandSolver}."""
    L = dict(DEFAULT_LENGTHS if L is None else L)
    hands = hands or {}
    sp = L["spine"]
    off = {"Hips": (0, 0, 0), "spine": (0, 0, 0), "chest": (0, 0.5 * sp, 0), "neck": (0, 0.5 * sp, 0),
           "head": (0, NECK_LEN, 0), "jaw": (0, 0.07, 0.0), "eye_L": (0.032, 0.115, 0.085), "eye_R": (-0.032, 0.115, 0.085)}
    ends = {"head": (0, HEAD_TOP, 0), "jaw": (0, -0.065, 0.085), "eye_L": (0, 0, 0.02), "eye_R": (0, 0, 0.02),
            "toe_L": (0, 0, 0.35 * L["foot"]), "toe_R": (0, 0, 0.35 * L["foot"])}
    for s, sx in (("L", 1.0), ("R", -1.0)):
        off.update({"upper_arm_" + s: (sx * L["sw"], 0.5 * sp, 0), "forearm_" + s: (0, -L["upper_arm"], 0),
                    "hand_" + s: (0, -L["forearm"], 0), "thigh_" + s: (sx * L["hw"], 0, 0),
                    "shin_" + s: (0, -L["thigh"], 0), "foot_" + s: (0, -L["shin"], 0),
                    "toe_" + s: (0, 0, 0.65 * L["foot"])})
        hs = hands.get(s) or HandSolver()
        for k, v in hs.rest_offsets().items():
            if k.endswith("_end"):
                ends[f"{k[:-4]}_3_{s}"] = v
            else:
                f, i = k.split("_")
                off[f"{f}_{i}_{s}"] = v
    return Skeleton(JOINT_ORDER_V2, PARENT_V2, off, ends)


# ----------------------------------------------------------------------------------------------
def q_pow(q, p):
    """q ** p for unit quaternions (fractional rotation about the same axis)."""
    w = float(np.clip(q[0], -1.0, 1.0))
    ang = 2.0 * math.acos(abs(w))
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return IDENTITY_Q.copy()
    axis = q[1:] / s * (1.0 if w >= 0 else -1.0)
    return q_axis_angle(axis, ang * p)


def face_joint_quats(bs):
    """Jaw and eye rotations from ARKit blendshapes (degrees are conservative human ranges)."""
    if bs is None:
        return {"jaw": IDENTITY_Q.copy(), "eye_L": IDENTITY_Q.copy(), "eye_R": IDENTITY_Q.copy()}
    b = lambda n: float(bs[BS_INDEX[n]])
    jaw = q_mul(q_axis_angle([0, 1, 0], (b("jawLeft") - b("jawRight")) * 0.15),
                q_axis_angle([1, 0, 0], b("jawOpen") * 0.42))
    out = {"jaw": jaw}
    for s, sx in (("L", 1.0), ("R", -1.0)):
        yaw = sx * (b("eyeLookOut" + {"L": "Left", "R": "Right"}[s]) - b("eyeLookIn" + {"L": "Left", "R": "Right"}[s])) * 0.55
        pitch = (b("eyeLookDown" + {"L": "Left", "R": "Right"}[s]) - b("eyeLookUp" + {"L": "Left", "R": "Right"}[s])) * 0.45
        out["eye_" + s] = q_mul(q_axis_angle([0, 1, 0], yaw), q_axis_angle([1, 0, 0], pitch))
    return out


def assemble(body, world, hands=None, bs=None, head_q=None):
    """body: v1 local quats (13 joints) from BodySolver; world: its world quats; hands: {"L"/"R": solver result or
    None}; bs: (52,) blendshapes or None; head_q: (4,) world quaternion from face tracker or None.
    Returns a dict with all JOINT_ORDER_V2 quaternions."""
    q = {"Hips": body["Hips"]}
    half = q_pow(body["spine"], 0.5)
    q["spine"], q["chest"] = half, half

    if head_q is not None and "chest" in world:
        head_rel = q_mul(q_conj(world["chest"]), head_q)
        neck = q_pow(head_rel, 0.35)
        q["neck"] = neck
        q["head"] = q_mul(q_conj(neck), head_rel)
    else:
        neck = q_pow(body["head"], 0.4)
        q["neck"] = neck
        q["head"] = q_mul(q_conj(neck), body["head"])

    q.update(face_joint_quats(bs))
    for s in SIDES:
        for n in ("upper_arm_", "forearm_", "thigh_", "shin_", "foot_"):
            q[n + s] = body[n + s]
        q["toe_" + s] = IDENTITY_Q.copy()
        h = (hands or {}).get(s)
        if h is None:
            q["hand_" + s] = IDENTITY_Q.copy()
            for j in FINGER_JOINTS:
                q[f"{j}_{s}"] = IDENTITY_Q.copy()
        else:
            q["hand_" + s] = q_mul(q_conj(world["forearm_" + s]), h["palm_q"])
            for j in FINGER_JOINTS:
                q[f"{j}_{s}"] = h["quats"][j]
    return q


def forward_kinematics(skel, quats, root):
    """Joint positions (J,3) and world rotation matrices (J,3,3) from local quaternions."""
    n = len(skel.order)
    pos, rot, idx = np.zeros((n, 3)), np.zeros((n, 3, 3)), {j: i for i, j in enumerate(skel.order)}
    for i, j in enumerate(skel.order):
        R = q_to_mat(quats[j])
        p = skel.parent.get(j)
        if p is None:
            pos[i], rot[i] = np.asarray(root, float), R
        else:
            pi = idx[p]
            rot[i] = rot[pi] @ R
            pos[i] = pos[pi] + rot[pi] @ np.asarray(skel.offsets[j])
    return pos, rot
