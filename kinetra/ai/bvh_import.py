"""BVH reader for AI training: parses any hierarchy, runs forward kinematics and extracts the 12 body joints,
so public mocap libraries (CMU, Mixamo exports, ...) and Kinetra's own BVH files can be used as training data."""
import math
import re
from pathlib import Path

import numpy as np

_TARGETS = {   # normalised joint names (lower case, letters/digits only) per target joint
    "sh_L": ("leftarm", "leftupperarm", "upperarml", "lupperarm", "lshoulderjoint"),
    "sh_R": ("rightarm", "rightupperarm", "upperarmr", "rupperarm"),
    "el_L": ("leftforearm", "forearml", "lowerarml", "lforearm"),
    "el_R": ("rightforearm", "forearmr", "lowerarmr", "rforearm"),
    "wr_L": ("lefthand", "handl", "lhand"), "wr_R": ("righthand", "handr", "rhand"),
    "hp_L": ("leftupleg", "leftupperleg", "thighl", "lthigh", "lfemur", "leftthigh"),
    "hp_R": ("rightupleg", "rightupperleg", "thighr", "rthigh", "rfemur", "rightthigh"),
    "kn_L": ("leftleg", "leftlowerleg", "shinl", "calfl", "lshin", "ltibia", "leftshin"),
    "kn_R": ("rightleg", "rightlowerleg", "shinr", "calfr", "rshin", "rtibia", "rightshin"),
    "an_L": ("leftfoot", "footl", "lfoot"), "an_R": ("rightfoot", "footr", "rfoot"),
}
_ORDER = ("sh_L", "sh_R", "el_L", "el_R", "wr_L", "wr_R", "hp_L", "hp_R", "kn_L", "kn_R", "an_L", "an_R")


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.split(":")[-1].lower())


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def parse_bvh(text):
    """Returns (joints, motion, frame_time). joints: list of dict(name, parent, offset, channels)."""
    head, motion = text.split("MOTION")
    toks = head.replace("{", " { ").replace("}", " } ").split()
    joints, stack, i = [], [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            joints.append({"name": toks[i + 1], "parent": stack[-1] if stack else None, "offset": np.zeros(3), "channels": []})
            i += 2
        elif t == "End":
            i += 2                                           # End Site
            stack.append(-1)
        elif t == "{":
            if not stack or stack[-1] != -1:
                stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            tgt = joints[stack[-1] if stack and stack[-1] != -1 else len(joints) - 1]
            if not (stack and stack[-1] == -1):
                tgt["offset"] = np.array([float(x) for x in toks[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1])
            joints[len(joints) - 1 if not stack or stack[-1] == -1 else stack[-1]]["channels"] = toks[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    lines = [l for l in motion.strip().splitlines() if l.strip()]
    ft = float(lines[1].split(":")[1])
    data = np.array([[float(v) for v in l.split()] for l in lines[2:]])
    return joints, data, ft


def joint_positions(joints, data):
    """Forward kinematics -> (T, J, 3) world positions."""
    T, J = len(data), len(joints)
    pos, rot = np.zeros((T, J, 3)), np.zeros((T, J, 3, 3))
    for t in range(T):
        c = 0
        for j, jt in enumerate(joints):
            off, R, tr = jt["offset"].copy(), np.eye(3), np.zeros(3)
            for ch in jt["channels"]:
                v = data[t, c]
                c += 1
                if ch.endswith("position"):
                    tr["XYZ".index(ch[0])] = v
                else:
                    R = R @ _rot(ch[0], v)
            if jt["parent"] is None:
                pos[t, j], rot[t, j] = off + tr, R
            else:
                p = jt["parent"]
                pos[t, j] = pos[t, p] + rot[t, p] @ off
                rot[t, j] = rot[t, p] @ R
    return pos


def body_from_bvh(text):
    """(T,12,3) y-up metres-ish (scale-free: the prior normalises by torso length) or None if joints are missing."""
    joints, data, _ = parse_bvh(text)
    names = {_norm(j["name"]): i for i, j in enumerate(joints)}
    idx = []
    for k in _ORDER:
        hit = next((names[n] for n in _TARGETS[k] if n in names), None)
        if hit is None:
            return None
        idx.append(hit)
    P = joint_positions(joints, data)[:, idx]
    up = ((P[:, 0] + P[:, 1]) / 2 - (P[:, 6] + P[:, 7]) / 2).mean(axis=0)          # hips -> shoulders
    ax = int(np.argmax(np.abs(up)))
    if ax != 1:                                                                    # z-up (or x-up) files -> y-up
        perm = [0, 1, 2]
        perm[1], perm[ax] = ax, 1
        P = P[..., perm]
        if ax == 2:
            P[..., 2] *= -1
        up = ((P[:, 0] + P[:, 1]) / 2 - (P[:, 6] + P[:, 7]) / 2).mean(axis=0)
    if up[1] < 0:
        P = P * np.array([1, -1, 1])
    return P


def load_bvh_folder(folder, max_files=400):
    out = []
    for f in sorted(Path(folder).rglob("*.bvh"))[:max_files]:
        try:
            seq = body_from_bvh(f.read_text(errors="ignore"))
        except Exception:
            continue
        if seq is not None and len(seq) >= 30:
            out.append(seq)
    return out
