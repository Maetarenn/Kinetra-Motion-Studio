import numpy as np
from kinetra.core.fullbody import build_skeleton, forward_kinematics
from kinetra.core.mannequin import MannequinModel, make_model
from kinetra.core.bodymesh import BodyModel
from kinetra.core.mathx import q_axis_angle, IDENTITY_Q


def test_mannequin_has_the_model_interface_and_covers_the_whole_body():
    sk = build_skeleton()
    for style in ("mannequin_f", "mannequin_m"):
        m = make_model(sk, style)
        assert isinstance(m, MannequinModel) and m.tris.max() < len(m.v_local) and len(m.colors) == len(m.v_local)
        q = {j: IDENTITY_Q.copy() for j in sk.order}
        pos, rot = forward_kinematics(sk, q, [0, 0.93, 0])
        V, N = m.pose(pos, rot)
        assert np.isfinite(V).all() and abs(np.linalg.norm(N, axis=1).mean() - 1) < 0.02
        assert 1.5 < np.ptp(V[:, 1]) < 2.0                                      # human height in metres
        joints_used = {sk.order[j] for j in m.slices}
        for j in ("Hips", "chest", "head", "upper_arm_L", "hand_R", "index_3_L", "thigh_R", "foot_L", "toe_R"):
            assert j in joints_used, j


def test_mannequin_follows_a_pose_and_style_factory_still_offers_the_detailed_model():
    sk = build_skeleton()
    m = make_model(sk)
    q = {j: IDENTITY_Q.copy() for j in sk.order}
    q["upper_arm_R"] = q_axis_angle([0, 0, -1], np.radians(90))
    pos, rot = forward_kinematics(sk, q, [0, 0.93, 0])
    V0, _ = m.pose(*forward_kinematics(sk, {j: IDENTITY_Q.copy() for j in sk.order}, [0, 0.93, 0]))
    V1, _ = m.pose(pos, rot)
    assert np.abs(V1 - V0).max() > 0.3
    assert isinstance(make_model(sk, "detailed"), BodyModel) and not isinstance(make_model(sk, "detailed"), MannequinModel)
