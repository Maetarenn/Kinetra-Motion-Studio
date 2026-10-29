"""Procedural, anatomically limited training data: finger gestures and whole-body motion."""
import math

import numpy as np

from ..core.anatomy import apply_limits, bone_axes
from ..core.fullbody import build_skeleton, forward_kinematics, JOINT_ORDER_V2
from ..core.mathx import q_axis_angle, q_mul, IDENTITY_Q
from .hand_prior import NDIM

D = math.radians

# gesture = (curl per finger [thumb, index, middle, ring, pinky], spread)
GESTURES = {
    "open": ((0.05, 0.0, 0.0, 0.0, 0.0), 0.5), "relaxed": ((0.3, 0.25, 0.3, 0.35, 0.4), 0.2),
    "fist": ((0.9, 1.0, 1.0, 1.0, 1.0), 0.0), "point": ((0.9, 0.0, 1.0, 1.0, 1.0), 0.0),
    "peace": ((0.9, 0.0, 0.0, 1.0, 1.0), 0.8), "thumbs_up": ((0.0, 1.0, 1.0, 1.0, 1.0), 0.0),
    "pinch": ((0.6, 0.55, 0.25, 0.2, 0.2), 0.1), "ok": ((0.6, 0.6, 0.0, 0.0, 0.0), 0.5),
    "grab": ((0.55, 0.6, 0.6, 0.6, 0.6), 0.3), "rock": ((0.9, 0.0, 1.0, 1.0, 0.0), 0.7),
    "three": ((0.9, 0.0, 0.0, 0.0, 1.0), 0.6), "claw": ((0.3, 0.45, 0.45, 0.45, 0.45), 0.6),
}
_ABD = np.array([0.0, 14.0, 4.0, -5.0, -14.0])       # spread direction per finger (degrees at spread = 1)


def gesture_vec(name, rng, noise=0.05):
    curl, spread = GESTURES[name]
    return pose_from_curl(np.array(curl), spread, rng, noise)


def pose_from_curl(curl, spread, rng, noise=0.05):
    x = np.zeros(NDIM)
    for i in range(5):
        c = float(np.clip(curl[i] + rng.normal(0, noise), 0, 1))
        if i == 0:                                        # thumb: opposition rather than curl
            x[0:4] = [D(55) * c, D(20) * spread, D(60) * c, D(70) * c]
        else:
            m = D(85) * c
            pip = D(105) * c * (0.9 + 0.1 * rng.random())
            x[4 * i:4 * i + 4] = [m, D(_ABD[i]) * spread * (1 - 0.6 * c) + rng.normal(0, D(2)), pip, 0.66 * pip + rng.normal(0, D(3))]
    return x


def random_hand_pose(rng):
    if rng.random() < 0.6:
        return gesture_vec(rng.choice(list(GESTURES)), rng, 0.08)
    a, b = (GESTURES[n] for n in rng.choice(list(GESTURES), 2))
    t = rng.random()
    return pose_from_curl(t * np.array(a[0]) + (1 - t) * np.array(b[0]), t * a[1] + (1 - t) * b[1], rng, 0.1)


def hand_sequences(rng, n_seq=60, length=240):
    """Gesture sequences with dwell times and smooth transitions (training data for the transition model)."""
    seqs = []
    for _ in range(n_seq):
        frames, cur = [], random_hand_pose(rng)
        while len(frames) < length:
            nxt = random_hand_pose(rng)
            dwell, tr = int(rng.integers(8, 50)), int(rng.integers(5, 20))
            frames += [cur + rng.normal(0, D(1.5), NDIM) for _ in range(dwell)]
            for k in range(tr):
                s = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / tr)
                frames.append((1 - s) * cur + s * nxt)
            cur = nxt
        seqs.append(np.array(frames[:length]))
    return seqs


# ----------------------------------------------------------------------------------------------
_SK = None
BODY_JOINTS = ("upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R", "hand_L", "hand_R",
               "thigh_L", "thigh_R", "shin_L", "shin_R", "foot_L", "foot_R")      # shoulders .. ankles (12 joints)


def _skel():
    global _SK
    if _SK is None:
        _SK = build_skeleton()
        _SK._axes = bone_axes(_SK)
    return _SK


def body_sequence(rng, T=120, fps=30.0):
    """(T,12,3) joint positions of a random plausible motion: legs program + arms program + torso sway."""
    sk = _skel()
    scale = rng.uniform(0.9, 1.1)
    legs = rng.choice(["walk", "run", "stand", "squat", "step"])
    arms = rng.choice(["swing", "wave", "reach", "hold", "both_up"])
    f = rng.uniform(0.8, 1.2) * (2.2 if legs == "run" else 1.0)
    ph0 = rng.uniform(0, 2 * math.pi)
    tgt = rng.uniform([-0.6, 0.2, 0.2], [0.6, 1.0, 0.9])
    out = np.zeros((T, 12, 3))
    heading = rng.uniform(-math.pi, math.pi)
    for t in range(T):
        ph = 2 * math.pi * f * t / fps + ph0
        q = {j: IDENTITY_Q.copy() for j in sk.order}
        amp = {"walk": 0.5, "run": 0.85, "stand": 0.0, "squat": 0.0, "step": 0.3}[legs]
        s1, s2 = math.sin(ph), math.sin(ph + math.pi)
        q["thigh_L"] = q_axis_angle([1, 0, 0], -amp * s1)
        q["thigh_R"] = q_axis_angle([1, 0, 0], -amp * s2)
        q["shin_L"] = q_axis_angle([1, 0, 0], 1.1 * amp * max(0, math.sin(ph + 1.0)))
        q["shin_R"] = q_axis_angle([1, 0, 0], 1.1 * amp * max(0, math.sin(ph + 1.0 + math.pi)))
        sq = 0.5 - 0.5 * math.cos(ph / 2) if legs == "squat" else 0.0
        if legs == "squat":
            q["thigh_L"] = q["thigh_R"] = q_axis_angle([1, 0, 0], -1.5 * sq)
            q["shin_L"] = q["shin_R"] = q_axis_angle([1, 0, 0], 1.6 * sq)
            q["spine"] = q_axis_angle([1, 0, 0], 0.4 * sq)
        k = 0.7 if arms == "swing" else 0.0
        q["upper_arm_L"] = q_axis_angle([-1, 0, 0], k * amp * s2 + (0.3 if arms == "hold" else 0.0))
        q["upper_arm_R"] = q_axis_angle([-1, 0, 0], k * amp * s1 + (0.3 if arms == "hold" else 0.0))
        q["forearm_L"] = q_axis_angle([-1, 0, 0], 0.4 + 0.5 * max(0, s2) * k)
        q["forearm_R"] = q_axis_angle([-1, 0, 0], 0.4 + 0.5 * max(0, s1) * k)
        if arms == "wave":
            q["upper_arm_R"] = q_mul(q_axis_angle([0, 0, 1], -2.2), q_axis_angle([0, 1, 0], 0.3))
            q["forearm_R"] = q_axis_angle([-1, 0, 0], 0.9 + 0.5 * math.sin(3 * ph))
        elif arms == "reach":
            reach = 0.5 - 0.5 * math.cos(ph / 2)
            q["upper_arm_R"] = q_axis_angle([-1, 0, 0], reach * (1.2 + tgt[1]))
            q["forearm_R"] = q_axis_angle([-1, 0, 0], 0.2 + (1 - reach) * 0.8)
        elif arms == "both_up":
            q["upper_arm_L"] = q_axis_angle([0, 0, 1], 2.4 - 0.2 * math.sin(ph))
            q["upper_arm_R"] = q_axis_angle([0, 0, -1], 2.4 - 0.2 * math.sin(ph))
        q["chest"] = q_axis_angle([0, 1, 0], 0.18 * math.sin(ph) * (0.5 + amp))
        q["Hips"] = q_axis_angle([0, 1, 0], heading + 0.05 * math.sin(ph / 2))
        q, _ = apply_limits(q, sk, sk._axes)
        stride = 1.1 * f if legs in ("walk", "run") else 0.0
        root = np.array([0.0, 0.95 - 0.35 * sq, 0.0]) + np.array([math.sin(heading), 0, math.cos(heading)]) * stride * t / fps
        pos, _ = forward_kinematics(sk, q, root)
        out[t] = np.array([pos[JOINT_ORDER_V2.index(j)] for j in BODY_JOINTS]) * scale
    return out
