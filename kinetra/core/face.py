"""Face math without MediaPipe: head-pose invariant face frame, head rotation, muscle model, heat colours."""
import math

import numpy as np

from .constants import BS_INDEX, NBS
from .mathx import mat_to_quat


def to_face_frame(P):
    """Head-pose invariant face frame: origin between the eyes, scale = outer-eye distance.
    Returns (L, R, origin, scale); L are landmarks in that frame."""
    x_ax = P[263] - P[33]
    s = float(np.linalg.norm(x_ax)) + 1e-9
    x_ax = x_ax / s
    down = P[152] - P[10]
    down = down - x_ax * np.dot(down, x_ax)
    down = down / (np.linalg.norm(down) + 1e-9)
    R = np.stack([x_ax, down, np.cross(x_ax, down)])
    o = (P[133] + P[362]) / 2
    return ((P - o) @ R.T) / s, R, o, s


def head_pose_scene(R):
    """Head rotation in scene axes (x right, y up, z toward the viewer)."""
    F = np.diag([1.0, -1.0, -1.0])
    return F @ R.T @ F


def head_quat(R):
    return mat_to_quat(head_pose_scene(R))


def euler_from_R(R):
    M = R.T
    yaw = math.degrees(math.atan2(M[0, 2], M[2, 2]))
    pitch = math.degrees(math.atan2(-M[1, 2], math.hypot(M[0, 2], M[2, 2])))
    roll = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return yaw, pitch, roll


# name, {blendshape: weight}, anchor landmark ids used for the heat map
MUSCLES = (
    ("Frontalis L", {"browOuterUpLeft": .6, "browInnerUp": .4}, (334, 296, 10)),
    ("Frontalis R", {"browOuterUpRight": .6, "browInnerUp": .4}, (105, 66, 10)),
    ("Corrugator L", {"browDownLeft": .8}, (336, 285)),
    ("Corrugator R", {"browDownRight": .8}, (107, 55)),
    ("Procerus", {"browDownLeft": .4, "browDownRight": .4, "noseSneerLeft": .2, "noseSneerRight": .2}, (168, 6)),
    ("Orbic. Oculi L", {"eyeBlinkLeft": .6, "eyeSquintLeft": .4, "cheekSquintLeft": .3}, (386, 374, 263, 362)),
    ("Orbic. Oculi R", {"eyeBlinkRight": .6, "eyeSquintRight": .4, "cheekSquintRight": .3}, (159, 145, 33, 133)),
    ("Lev. Palpebrae L", {"eyeWideLeft": 1.0}, (386, 385)),
    ("Lev. Palpebrae R", {"eyeWideRight": 1.0}, (159, 158)),
    ("Zygomaticus Maj. L", {"mouthSmileLeft": 1.0, "cheekSquintLeft": .2}, (291, 425)),
    ("Zygomaticus Maj. R", {"mouthSmileRight": 1.0, "cheekSquintRight": .2}, (61, 205)),
    ("Risorius L", {"mouthStretchLeft": .8, "mouthDimpleLeft": .4}, (291, 454)),
    ("Risorius R", {"mouthStretchRight": .8, "mouthDimpleRight": .4}, (61, 234)),
    ("Lev. Labii Sup. L", {"mouthUpperUpLeft": 1.0, "noseSneerLeft": .4}, (267, 358)),
    ("Lev. Labii Sup. R", {"mouthUpperUpRight": 1.0, "noseSneerRight": .4}, (37, 129)),
    ("Depressor Anguli L", {"mouthFrownLeft": 1.0}, (291, 314)),
    ("Depressor Anguli R", {"mouthFrownRight": 1.0}, (61, 84)),
    ("Depressor Lab. Inf.", {"mouthLowerDownLeft": .5, "mouthLowerDownRight": .5}, (17, 314, 84)),
    ("Mentalis", {"mouthShrugLower": .6, "mouthPucker": .3, "mouthRollLower": .4}, (152, 17)),
    ("Orbicularis Oris", {"mouthPucker": .7, "mouthFunnel": .6, "mouthClose": .4,
                          "mouthPressLeft": .3, "mouthPressRight": .3}, (13, 14, 0, 17, 61, 291)),
    ("Buccinator", {"cheekPuff": 1.0}, (50, 280, 205, 425)),
    ("Jaw Opener", {"jawOpen": 1.0}, (152, 172, 397)),
    ("Lat. Pterygoid", {"jawLeft": .8, "jawRight": .8, "jawForward": .8}, (172, 397)),
    ("Nasalis L", {"noseSneerLeft": 1.0}, (358,)),
    ("Nasalis R", {"noseSneerRight": 1.0}, (129,)),
)
MUSCLE_NAMES = [m[0] for m in MUSCLES]


class MuscleModel:
    W = np.zeros((len(MUSCLES), NBS))
    for _m, (_n, _bs, _a) in enumerate(MUSCLES):
        for _k, _w in _bs.items():
            W[_m, BS_INDEX[_k]] = _w

    @classmethod
    def activations(cls, bs):
        return np.clip(cls.W @ bs, 0.0, 1.0)

    @staticmethod
    def vertex_weights(L, sigma=0.25):
        """(468, M) gaussian influence of each muscle on every mesh vertex (face-frame units)."""
        V = L[:468]
        out = np.zeros((468, len(MUSCLES)))
        for m, (_, _, anchors) in enumerate(MUSCLES):
            d2 = np.min(((V[:, None, :] - V[list(anchors)][None, :, :]) ** 2).sum(-1), axis=1)
            out[:, m] = np.exp(-d2 / (2 * sigma * sigma))
        return out


def heat_rgba(h):
    h = np.clip(h, 0.0, 1.0)[:, None]
    c0, c1, c2 = np.array([0.0, .94, 1.0]), np.array([1.0, .70, 0.0]), np.array([1.0, .15, .10])
    rgb = np.where(h < .5, c0 + (c1 - c0) * (h / .5), c1 + (c2 - c1) * ((h - .5) / .5))
    return np.concatenate([rgb, np.clip(h * 1.6, 0, 1)], axis=1).astype(np.float32)
