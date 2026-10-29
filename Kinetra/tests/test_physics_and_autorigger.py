import numpy as np
from pathlib import Path
from kinetra.core.physics_engine import PhysicsEngine, fast_lbs, _LIB
from kinetra.core.auto_rigger import AutoRiggedCharacter
from kinetra.character_skinned import SkinnedCharacter
from kinetra.core.fullbody import JOINT_ORDER_V2
from kinetra.core.mathx import IDENTITY_Q, q_axis_angle


def test_cpp_physics_engine_loaded_and_steps():
    assert _LIB is not None, "kinetra_physics.dll must be compiled and loaded"
    pe = PhysicsEngine(num_joints=33, ground_y=0.0)
    J = np.zeros((33, 3), dtype=np.float32)
    J[:, 1] = 1.0  # height 1m
    pe.reset(J)

    # Step forward towards target with offset
    target = J.copy()
    target[:, 1] = 1.2
    stepped = pe.step(target, dt=1.0 / 60.0)
    assert stepped.shape == (33, 3)
    assert np.all(stepped[:, 1] > 1.0) and np.all(stepped[:, 1] <= 1.2)


def test_physics_engine_ground_constraint():
    pe = PhysicsEngine(num_joints=33, ground_y=0.0)
    J = np.zeros((33, 3), dtype=np.float32)
    pe.reset(J)

    # Target below ground
    target = np.full((33, 3), -0.5, dtype=np.float32)
    stepped = pe.step(target, dt=1.0 / 60.0)
    assert np.all(stepped[:, 1] >= 0.0), "Ground penetration constraint must prevent y < 0.0"


def test_fast_lbs_c_acceleration():
    nv = 100
    rest_v = np.random.randn(nv, 3).astype(np.float32)
    rest_n = np.random.randn(nv, 3).astype(np.float32)
    rest_n /= np.linalg.norm(rest_n, axis=1, keepdims=True)

    matrices = np.tile(np.eye(4, dtype=np.float32)[None], (10, 1, 1))
    joint_nodes = np.zeros((nv, 4), dtype=np.int32)
    weights = np.zeros((nv, 4), dtype=np.float32)
    weights[:, 0] = 1.0

    out_v, out_n = fast_lbs(rest_v, rest_n, matrices, joint_nodes, weights)
    assert np.allclose(out_v, rest_v, atol=1e-5)
    assert np.allclose(out_n, rest_n, atol=1e-5)


def test_auto_rigging_and_obj_animation():
    obj_path = Path(__file__).parent.parent / "kinetra" / "assets" / "models" / "bjd_basemesh.obj"
    assert obj_path.exists(), "BJD Basemesh OBJ asset must exist"

    ch = SkinnedCharacter(str(obj_path))
    assert ch.V > 1000
    assert len(ch.tris) > 1000

    q = {j: IDENTITY_Q.copy() for j in JOINT_ORDER_V2}
    q["upper_arm_L"] = q_axis_angle([-1, 0, 0], np.pi / 2)

    v0, n0 = ch.pose({j: IDENTITY_Q.copy() for j in JOINT_ORDER_V2}, [0, 0.92, 0])
    v1, n1 = ch.pose(q, [0, 0.92, 0])

    assert np.isfinite(v1).all()
    assert np.isfinite(n1).all()
    # Arm rotation must move left arm vertices
    assert np.abs(v1 - v0).max() > 0.15
