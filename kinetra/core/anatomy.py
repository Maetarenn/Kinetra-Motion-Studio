"""Anatomical joint limits for the 52-joint skeleton: how far a human joint can really move.

Ball joints are limited by swing (cone) and twist about the bone axis, hinge joints by their flexion angle.
The same table is used to clamp tracked motion and to sample plausible random poses for AI training."""
import math

import numpy as np

from .mathx import unit, q_mul, q_conj, q_axis_angle, q_norm, IDENTITY_Q

# joint-name prefix -> (kind, max_swing_or_flex_deg, max_twist_deg)
_LIMITS = {
    "spine": ("ball", 35, 25), "chest": ("ball", 35, 25), "neck": ("ball", 45, 45), "head": ("ball", 40, 50),
    "jaw": ("ball", 30, 8), "eye_": ("ball", 40, 5),
    "upper_arm_": ("ball", 175, 100), "forearm_": ("hinge", 162, 0), "hand_": ("ball", 85, 90),
    "thigh_": ("ball", 125, 50), "shin_": ("hinge", 150, 0), "foot_": ("ball", 55, 30), "toe_": ("ball", 60, 10),
    "thumb_1": ("ball", 70, 15), "thumb_2": ("hinge", 60, 0), "thumb_3": ("hinge", 80, 0),
    "index_1": ("ball", 95, 8), "middle_1": ("ball", 95, 8), "ring_1": ("ball", 95, 8), "pinky_1": ("ball", 95, 12),
    "index_2": ("hinge", 110, 0), "middle_2": ("hinge", 110, 0), "ring_2": ("hinge", 110, 0), "pinky_2": ("hinge", 110, 0),
    "index_3": ("hinge", 90, 0), "middle_3": ("hinge", 90, 0), "ring_3": ("hinge", 90, 0), "pinky_3": ("hinge", 90, 0),
}


def limits_for(joint):
    """(kind, max_swing/flex_deg, max_twist_deg) for a joint name, or None for the root."""
    if joint == "Hips":
        return None
    for k in sorted(_LIMITS, key=len, reverse=True):
        if joint.startswith(k):
            return _LIMITS[k]
    return None


def swing_twist(q, axis):
    """q = swing * twist, twist is the rotation about `axis`. Returns (swing, twist_angle_signed)."""
    r = q[1:]
    d = float(np.dot(r, axis))
    twist = q_norm(np.array([q[0], *(axis * d)]))
    swing = q_mul(q, q_conj(twist))
    ang = 2.0 * math.atan2(d, q[0])
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    return swing, ang


def _angle(q):
    return 2.0 * math.acos(min(abs(float(q[0])), 1.0))


def limit_joint(q, kind, max_a, max_twist, axis):
    """Clamp one local rotation. Returns (q, was_clamped)."""
    q = q_norm(q)
    if q[0] < 0:
        q = -q
    if kind == "hinge":
        a = _angle(q)
        if a > math.radians(max_a):
            return q_axis_angle(q[1:], math.radians(max_a)), True
        return q, False
    swing, tw = swing_twist(q, axis)
    clamped = False
    if swing[0] < 0:
        swing = -swing
    sa = _angle(swing)
    if sa > math.radians(max_a):
        swing = q_axis_angle(swing[1:], math.radians(max_a))
        clamped = True
    lim = math.radians(max_twist)
    if abs(tw) > lim:
        tw = math.copysign(lim, tw)
        clamped = True
    return q_mul(swing, q_axis_angle(axis, tw)), clamped


def bone_axes(skel):
    """Local bone axis of every joint = direction to its first child (falls back to +y)."""
    ax = {}
    for j in skel.order:
        kids = [c for c in skel.order if skel.parent.get(c) == j]
        v = np.array(skel.offsets[kids[0]]) if kids else np.array(skel.ends.get(j, (0, 1, 0)))
        ax[j] = unit(v, [0.0, 1.0, 0.0])
    return ax


def apply_limits(quats, skel, axes=None):
    """Returns (clamped quats dict, list of joints that had to be clamped)."""
    axes = axes or bone_axes(skel)
    out, hit = {}, []
    for j, q in quats.items():
        lim = limits_for(j)
        if lim is None:
            out[j] = q
            continue
        out[j], c = limit_joint(q, lim[0], lim[1], lim[2], axes.get(j, np.array([0.0, 1.0, 0.0])))
        if c:
            hit.append(j)
    return out, hit


def random_pose(rng, skel, scale=0.5):
    """Sample a plausible pose: each joint gets a random rotation inside a fraction of its limits."""
    axes = bone_axes(skel)
    q = {}
    for j in skel.order:
        lim = limits_for(j)
        if lim is None:
            q[j] = IDENTITY_Q.copy()
            continue
        kind, a, tw = lim
        if kind == "hinge":
            q[j] = q_axis_angle(axes_perp(axes[j], rng), math.radians(rng.uniform(0, a * scale * 1.6)))
        else:
            ax = unit(rng.normal(size=3))
            sw = q_axis_angle(ax - axes[j] * float(np.dot(ax, axes[j])) + 1e-9, math.radians(rng.uniform(0, a * scale)))
            q[j] = q_mul(sw, q_axis_angle(axes[j], math.radians(rng.uniform(-tw, tw) * scale)))
    return q


def axes_perp(axis, rng):
    v = rng.normal(size=3)
    return unit(v - axis * float(np.dot(v, axis)))
