import math
import numpy as np
from kinetra.core.anatomy import *
from kinetra.core.fullbody import build_skeleton
from kinetra.core.mathx import q_axis_angle, q_mul

SK = build_skeleton()
AX = bone_axes(SK)


def test_hinge_limit_clamps_only_beyond_range():
    q, hit = apply_limits({"shin_L": q_axis_angle([1, 0, 0], math.radians(170))}, SK)
    assert hit == ["shin_L"] and abs(math.degrees(2 * math.acos(q["shin_L"][0])) - 150) < 0.5
    q, hit = apply_limits({"shin_L": q_axis_angle([1, 0, 0], math.radians(90))}, SK)
    assert hit == []


def test_ball_joint_swing_and_twist_limits():
    big_swing = q_axis_angle([1, 0, 0], math.radians(200))
    q, hit = apply_limits({"thigh_L": big_swing}, SK)
    assert hit == ["thigh_L"] and abs(math.degrees(2 * math.acos(abs(q["thigh_L"][0]))) - 125) < 1.0
    twist = q_axis_angle(AX["thigh_L"], math.radians(120))
    q, hit = apply_limits({"thigh_L": twist}, SK)
    _, tw = swing_twist(q["thigh_L"], AX["thigh_L"])
    assert abs(math.degrees(abs(tw)) - 50) < 0.5


def test_valid_pose_is_left_alone_and_random_poses_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(30):
        pose = random_pose(rng, SK, 0.5)
        clamped, hit = apply_limits(pose, SK)
        assert hit == [], hit
    q = {"upper_arm_L": q_axis_angle([0, 0, 1], math.radians(60))}
    out, hit = apply_limits(q, SK)
    assert hit == [] and np.allclose(abs(out["upper_arm_L"]), abs(q["upper_arm_L"]), atol=1e-6)
