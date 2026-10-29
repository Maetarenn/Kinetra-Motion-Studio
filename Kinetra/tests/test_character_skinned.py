import os
from pathlib import Path
import numpy as np
import pytest
from kinetra.character_skinned import CharacterRig, map_bones, SkinnedCharacter
from kinetra.core.fullbody import JOINT_ORDER_V2
from kinetra.core.mathx import q_axis_angle, IDENTITY_Q


def tpose_rig():
    pos = np.array([(0, 1, 0), (0, 1.1, 0), (0, 1.25, 0), (0, 1.45, 0), (0, 1.55, 0), (0.18, 1.42, 0), (0.42, 1.42, 0),
                    (0.66, 1.42, 0), (0.76, 1.42, 0), (0.1, 0.95, 0), (0.1, 0.5, 0), (0.1, 0.08, 0), (0.1, 0.0, 0.12)])
    parent = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 0, 9, 10, 11]
    names = ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Neck", "mixamorig:Head", "mixamorig:LeftArm",
             "mixamorig:LeftForeArm", "mixamorig:LeftHand", "mixamorig:LeftHandMiddle1", "mixamorig:LeftUpLeg",
             "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase"]
    return CharacterRig(parent, pos, map_bones(names)), pos


def ident():
    return {j: IDENTITY_Q.copy() for j in JOINT_ORDER_V2}


def test_bone_names_of_common_rigs_are_mapped():
    m = map_bones(["Hips", "Spine", "LeftUpperArm", "LeftLowerArm", "LeftHand", "LeftHandIndex2", "RightUpperLeg", "RightToes"])
    assert m["Hips"] == 0 and m["upper_arm_L"] == 2 and m["forearm_L"] == 3 and m["index_2_L"] == 5
    assert m["thigh_R"] == 6 and m["toe_R"] == 7
    m = map_bones(["pelvis", "upperarm_l", "lowerarm_l", "hand_l", "thigh_r", "calf_r", "foot_r"])
    assert m["upper_arm_L"] == 1 and m["forearm_L"] == 2 and m["shin_R"] == 5


def test_tpose_character_is_brought_to_kinetra_rest_pose():
    rig, pos = tpose_rig()
    Rd, p = rig.pose(ident(), [0, rig.ours_h, 0])
    d = p[6] - p[5]
    assert np.allclose(d / np.linalg.norm(d), [0, -1, 0], atol=1e-6)               # T-pose arm now hangs down
    assert abs(np.linalg.norm(d) - 0.24) < 1e-9                                      # the character's own proportions are kept
    assert np.allclose(p[8] - p[7], [0, -0.1, 0], atol=1e-6)                         # hand continues the arm


def test_kinetra_motion_drives_the_character_limbs():
    rig, _ = tpose_rig()
    q = ident()
    q["upper_arm_L"] = q_axis_angle([-1, 0, 0], np.pi / 2)                           # arm forward
    q["thigh_L"] = q_axis_angle([-1, 0, 0], np.pi / 2)                               # leg forward
    Rd, p = rig.pose(q, [0, rig.ours_h, 0])
    arm = (p[6] - p[5]) / np.linalg.norm(p[6] - p[5])
    leg = (p[10] - p[9]) / np.linalg.norm(p[10] - p[9])
    assert np.allclose(arm, [0, 0, 1], atol=1e-6) and np.allclose(leg, [0, 0, 1], atol=1e-6)


def test_root_translation_is_scaled_to_the_character():
    rig, pos = tpose_rig()
    _, p0 = rig.pose(ident(), [0, rig.ours_h, 0])
    _, p1 = rig.pose(ident(), [1.0, rig.ours_h, 0])
    assert abs((p1[0] - p0[0])[0] - rig.scale) < 1e-9


MICHELLE = Path(os.environ.get("KINETRA_TEST_GLB", "/tmp/chars/Michelle.glb"))


@pytest.mark.skipif(not MICHELLE.exists(), reason="sample character not downloaded")
def test_real_mixamo_character_loads_and_animates():
    ch = SkinnedCharacter(MICHELLE)
    assert len(ch.bone_map) >= 45 and ch.V > 5000
    q = ident()
    q["upper_arm_R"] = q_axis_angle([0, 0, -1], np.radians(90))
    V0, _ = ch.pose(ident(), [0, 0.93, 0])
    V1, N1 = ch.pose(q, [0, 0.93, 0])
    assert np.isfinite(V1).all() and np.abs(V1 - V0).max() > 0.3                    # the arm really moved
    assert abs(np.linalg.norm(N1, axis=1).mean() - 1.0) < 1e-3
    assert np.ptp(V0[:, 1]) > 1.2                                                    # a person-sized mesh
