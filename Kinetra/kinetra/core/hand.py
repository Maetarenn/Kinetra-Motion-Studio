"""Finger solver for the 21 MediaPipe hand landmarks.

Hand frame (identical for left and right hands, so mirrored hands need no special cases):
  Y = -(wrist -> middle knuckle)  (rest: fingers hang along -Y)
  Z = pinky knuckle -> index knuckle, made orthogonal to Y   (thumb side)
  X = Y x Z
Rest pose: fingers straight along -Y, thumb along THUMB_REST. Every finger joint stores a local rotation relative
to its parent: knuckle joints are 2-DoF swings, middle and tip joints are single-axis hinges.
"""
import math
from collections import deque

import numpy as np

from .mathx import (unit, q_mul, q_conj, q_axis_angle, q_from_vectors, q_rotate, frame_quat, q_to_mat, IDENTITY_Q)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
          "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FINGER_JOINTS = tuple(f"{f}_{i}" for f in FINGERS for i in (1, 2, 3))
FINGER_REST = {f: np.array([0.0, -1.0, 0.0]) for f in FINGERS}
THUMB_REST = unit([0.0, -0.8, 0.6])
FINGER_REST["thumb"] = THUMB_REST

# default palm geometry (metres, hand frame, wrist at the origin) and bone lengths (proximal, middle, distal)
DEFAULT_PALM = {"thumb": (0.0, -0.020, 0.022), "index": (0.0, -0.088, 0.030), "middle": (0.0, -0.092, 0.0),
                "ring": (0.0, -0.086, -0.012), "pinky": (0.0, -0.077, -0.028)}
DEFAULT_BONES = {"thumb": (0.045, 0.033, 0.027), "index": (0.043, 0.025, 0.020), "middle": (0.047, 0.028, 0.022),
                 "ring": (0.043, 0.026, 0.020), "pinky": (0.034, 0.019, 0.017)}


def hand_frame(P):
    """Rotation matrix (columns X, Y, Z) of the hand frame from 21 landmarks."""
    Y = -unit(P[9] - P[0], [0.0, -1.0, 0.0])
    t = P[5] - P[17]
    Z = unit(t - Y * float(np.dot(t, Y)), [0.0, 0.0, 1.0])
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z], axis=1)


def _hinge(parent_w, pd, cd, gain=1.0):
    raw_ang = math.acos(float(np.clip(np.dot(pd, cd), -1.0, 1.0)))
    ang = min(raw_ang * gain, math.radians(110.0))
    axis_w = unit(np.cross(pd, cd), q_rotate(parent_w, np.array([0.0, 0.0, 1.0])))
    return q_axis_angle(q_rotate(q_conj(parent_w), axis_w), ang)


class HandSolver:
    """One instance per hand. solve() returns local finger quaternions and the palm orientation."""
    TEMPLATE_FRAMES = 40

    def __init__(self):
        self._palm_obs, self._bone_obs = deque(maxlen=200), deque(maxlen=200)

    def reset_template(self):
        self._palm_obs.clear()
        self._bone_obs.clear()

    # -- template ---------------------------------------------------------------------------------
    def palm_offsets(self):
        if len(self._palm_obs) < self.TEMPLATE_FRAMES:
            return {f: np.array(v) for f, v in DEFAULT_PALM.items()}
        med = np.median(np.array(self._palm_obs), axis=0)             # (5, 3)
        return {f: med[i] for i, f in enumerate(FINGERS)}

    def bone_lengths(self):
        if len(self._bone_obs) < self.TEMPLATE_FRAMES:
            return {f: tuple(v) for f, v in DEFAULT_BONES.items()}
        med = np.median(np.array(self._bone_obs), axis=0)             # (5, 3)
        return {f: tuple(med[i]) for i, f in enumerate(FINGERS)}

    # -- solve ------------------------------------------------------------------------------------
    def solve(self, P, learn=True):
        """P: (21,3) landmarks in scene axes (y up). Returns dict(palm_q, quats, tips, curl)."""
        P = np.asarray(P, dtype=np.float64)
        R = hand_frame(P)
        Pl = (P - P[0]) @ R                                           # landmarks in the hand frame
        if learn:
            self._palm_obs.append([Pl[CHAINS[f][0]] for f in FINGERS])
            self._bone_obs.append([[np.linalg.norm(Pl[c[i + 1]] - Pl[c[i]]) for i in range(3)]
                                   for c in (CHAINS[f] for f in FINGERS)])
        quats, curl = {}, {}
        for f in FINGERS:
            a, b, c, d = CHAINS[f]
            d1, d2, d3 = (unit(Pl[b] - Pl[a], FINGER_REST[f]), unit(Pl[c] - Pl[b], FINGER_REST[f]),
                          unit(Pl[d] - Pl[c], FINGER_REST[f]))
            q1 = q_from_vectors(FINGER_REST[f], d1)
            q2 = _hinge(q1, d1, d2)
            w2 = q_mul(q1, q2)
            q3 = _hinge(w2, d2, d3)
            quats[f + "_1"], quats[f + "_2"], quats[f + "_3"] = q1, q2, q3
            curl[f] = float(np.clip((math.acos(np.clip(np.dot(d1, d2), -1, 1)) +
                                     math.acos(np.clip(np.dot(d2, d3), -1, 1))) / math.radians(180), 0, 1))
        return {"palm_q": frame_quat(R[:, 0], R[:, 1]), "palm_R": R, "quats": quats, "curl": curl}

    def rest_offsets(self):
        """Joint offsets in the parent's local frame (hand frame at rest) for the export skeleton."""
        palm, bones, off = self.palm_offsets(), self.bone_lengths(), {}
        for f in FINGERS:
            off[f + "_1"] = tuple(palm[f])
            off[f + "_2"] = tuple(FINGER_REST[f] * bones[f][0])
            off[f + "_3"] = tuple(FINGER_REST[f] * bones[f][1])
            off[f + "_end"] = tuple(FINGER_REST[f] * bones[f][2])
        return off


def hand_fk(quats, offsets):
    """Landmark positions (21,3) in the hand frame from local finger quaternions (wrist at the origin)."""
    out = np.zeros((21, 3))
    for f in FINGERS:
        a, b, c, d = CHAINS[f]
        p1 = np.array(offsets[f + "_1"])
        w1 = quats[f + "_1"]
        w2 = q_mul(w1, quats[f + "_2"])
        w3 = q_mul(w2, quats[f + "_3"])
        out[a] = p1
        out[b] = p1 + q_rotate(w1, np.array(offsets[f + "_2"]))
        out[c] = out[b] + q_rotate(w2, np.array(offsets[f + "_3"]))
        out[d] = out[c] + q_rotate(w3, np.array(offsets[f + "_end"]))
    return out
