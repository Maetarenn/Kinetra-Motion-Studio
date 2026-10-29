import numpy as np
from kinetra.core.body_solver import BodySolver
from kinetra.core.constants import *
from kinetra.core.mathx import q_to_mat, q_mul, unit


def synth_pose(arm_flex=0.6, knee_flex=0.4):
    """Hip-centred standing pose, y up, +x = left, facing +z."""
    J = np.zeros((33, 3))
    J[HIP_L], J[HIP_R] = [0.10, 0, 0], [-0.10, 0, 0]
    J[SHO_L], J[SHO_R] = [0.19, 0.50, 0], [-0.19, 0.50, 0]
    J[EAR_L], J[EAR_R] = [0.08, 0.70, 0], [-0.08, 0.70, 0]
    J[NOSE] = [0, 0.68, 0.10]
    for s, sx in (("L", 1), ("R", -1)):
        a, b, c, deps = ARMS[s]
        J[b] = J[a] + [0, -0.30, 0]
        J[c] = J[b] + [0, -0.26 * np.cos(arm_flex), 0.26 * np.sin(arm_flex)]
        for d in deps:
            J[d] = J[c]
        a, b, c, deps = LEGS[s]
        J[b] = J[a] + [0, -0.43, 0]
        J[c] = J[b] + [0, -0.42 * np.cos(knee_flex), -0.42 * np.sin(knee_flex)]
        J[HEEL[s]] = J[c] + [0, -0.07, -0.05]
        J[TOE[s]] = J[c] + [0, -0.07, 0.15]
    return J


def fk(quats, L, root):
    off = {"Hips": (0, 0, 0), "spine": (0, 0, 0), "head": (0, L["spine"], 0),
           "upper_arm_L": (L["sw"], L["spine"], 0), "forearm_L": (0, -L["upper_arm"], 0),
           "upper_arm_R": (-L["sw"], L["spine"], 0), "forearm_R": (0, -L["upper_arm"], 0),
           "thigh_L": (L["hw"], 0, 0), "shin_L": (0, -L["thigh"], 0), "foot_L": (0, -L["shin"], 0),
           "thigh_R": (-L["hw"], 0, 0), "shin_R": (0, -L["thigh"], 0), "foot_R": (0, -L["shin"], 0)}
    world, pos = {}, {}
    for n in JOINT_ORDER:
        p = JOINT_PARENT.get(n)
        R = q_to_mat(quats[n])
        if p is None:
            world[n], pos[n] = R, np.array(root, float)
        else:
            world[n] = world[p] @ R
            pos[n] = pos[p] + world[p] @ np.array(off[n])
    return world, pos


def run(J, frames=90, **kw):
    s = BodySolver()
    out = None
    for i in range(frames):
        out = s.process(J, i / 30.0, root=(0.0, 0.0), **kw)
    return s, out


def test_calibration_and_fk_roundtrip():
    J = synth_pose()
    s, out = run(J)
    assert not s.calibrating and s.seg is not None
    L = out["lengths"]
    world, pos = fk(out["quats"], L, out["joints"][HIP_MID])
    Jo = out["joints"]
    # wrists and ankles reproduced by forward kinematics of the solved quaternions
    for s_ in "LR":
        w_fk = pos["forearm_" + s_] + world["forearm_" + s_] @ np.array([0, -L["forearm"], 0])
        assert np.linalg.norm(w_fk - Jo[ARMS[s_][2]]) < 0.02, ("wrist", s_)
        a_fk = pos["foot_" + s_]
        assert np.linalg.norm(a_fk - Jo[LEGS[s_][2]]) < 0.02, ("ankle", s_)
        t_fk = pos["foot_" + s_] + world["foot_" + s_] @ np.array([0, 0, L["foot"]])
        assert np.linalg.norm(t_fk - Jo[TOE[s_]]) < 0.04, ("toe", s_)
    head_fk = pos["head"]
    assert np.linalg.norm(head_fk - Jo[SHO_MID]) < 0.01


def test_body_yaw_is_captured():
    J = synth_pose()
    th = np.radians(40)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    s, out = run(J @ Ry.T)
    world, _ = fk(out["quats"], out["lengths"], out["joints"][HIP_MID])
    fwd = world["Hips"] @ np.array([0, 0, 1.0])
    assert abs(np.degrees(np.arctan2(fwd[0], fwd[2])) - 40) < 2.0


def test_bone_lengths_are_constant():
    J = synth_pose()
    s = BodySolver()
    lens = []
    rng = np.random.default_rng(1)
    for i in range(120):
        noisy = J + rng.normal(0, 0.01, J.shape)
        out = s.process(noisy, i / 30.0, root=(0, 0))
        if i > 80:
            Jo = out["joints"]
            lens.append(np.linalg.norm(Jo[13] - Jo[11]))
    assert np.std(lens) < 1e-6


def test_foot_lock_plants_feet_and_keeps_leg_lengths():
    J = synth_pose(knee_flex=0.1)
    J[:, 1] += 0.97                      # standing on the floor (ankle ~0.08 above)
    s = BodySolver()
    # standing still for 60 frames: both feet must lock
    for i in range(60):
        out = s.process(J, i / 30.0, root=(0, 0))
    assert out["locks"].all()
    Jo = out["joints"]
    for side in "LR":
        hip, knee, ank, _ = LEGS[side]
        assert abs(np.linalg.norm(Jo[knee] - Jo[hip]) - s.seg["thigh_" + side]) < 1e-3
        assert abs(np.linalg.norm(Jo[ank] - Jo[knee]) - s.seg["shin_" + side]) < 1e-3
        assert min(Jo[HEEL[side], 1], Jo[TOE[side], 1]) < 0.03


def test_prefiltered_input_is_not_filtered_again():
    J = synth_pose() + np.array([0, 0.97, 0])
    s = BodySolver()
    s.process(J, 0.0, absolute=True, prefiltered=True)
    moved = J + 0.5
    out = s.process(moved, 0.033, absolute=True, prefiltered=True, ground=0.0)
    assert np.allclose(out["joints"][:33], moved, atol=1e-9)      # a live filter would have lagged behind
