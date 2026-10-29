"""Physical dynamics & collision engine for Kinetra.

Bridges high-performance C++ physics (kinetra_physics.dll) with seamless NumPy fallback.
Provides:
  - Second-order spring-damper joint dynamics (PD tracking towards vision landmarks)
  - Inertia and velocity damping to eliminate high-frequency jitter
  - Ground non-penetration constraints and floor friction
  - High-performance Linear Blend Skinning (LBS)
"""
import ctypes
import os
from pathlib import Path
import numpy as np

_DLL_PATH = Path(__file__).parent / "kinetra_physics.dll"
_LIB = None

if _DLL_PATH.exists():
    try:
        _LIB = ctypes.CDLL(str(_DLL_PATH))

        class KinetraPhysicsConfig(ctypes.Structure):
            _fields_ = [
                ("gravity_y", ctypes.c_float),
                ("ground_y", ctypes.c_float),
                ("floor_restitution", ctypes.c_float),
                ("floor_friction", ctypes.c_float),
                ("default_stiffness", ctypes.c_float),
                ("default_damping", ctypes.c_float),
            ]

        _LIB.kinetra_physics_create.argtypes = [ctypes.c_int, ctypes.POINTER(KinetraPhysicsConfig)]
        _LIB.kinetra_physics_create.restype = ctypes.c_void_p

        _LIB.kinetra_physics_destroy.argtypes = [ctypes.c_void_p]
        _LIB.kinetra_physics_destroy.restype = None

        _LIB.kinetra_physics_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        _LIB.kinetra_physics_reset.restype = None

        _LIB.kinetra_physics_set_ground.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        _LIB.kinetra_physics_set_ground.restype = None

        _LIB.kinetra_physics_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        _LIB.kinetra_physics_step.restype = None

        _LIB.kinetra_fast_lbs.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        _LIB.kinetra_fast_lbs.restype = None
    except Exception:
        _LIB = None


class PhysicsEngine:
    def __init__(self, num_joints=33, stiffness=320.0, damping=38.0, ground_y=0.0):
        self.num_joints = num_joints
        self.stiffness = stiffness
        self.damping = damping
        self.ground_y = ground_y
        self.pos = np.zeros((num_joints, 3), dtype=np.float32)
        self.vel = np.zeros((num_joints, 3), dtype=np.float32)
        self.is_initialized = False

        self._handle = None
        if _LIB is not None:
            cfg = KinetraPhysicsConfig(
                gravity_y=-9.81,
                ground_y=ground_y,
                floor_restitution=0.0,
                floor_friction=0.85,
                default_stiffness=stiffness,
                default_damping=damping
            )
            self._handle = _LIB.kinetra_physics_create(num_joints, ctypes.byref(cfg))

    def __del__(self):
        if self._handle and _LIB is not None:
            try:
                _LIB.kinetra_physics_destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def reset(self, initial_joints=None):
        if initial_joints is not None:
            self.pos = np.asarray(initial_joints, dtype=np.float32).copy()
            self.vel.fill(0.0)
            self.is_initialized = True
            if self._handle:
                p_ptr = self.pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                _LIB.kinetra_physics_reset(self._handle, p_ptr)
        else:
            self.is_initialized = False

    def set_ground(self, ground_y, restitution=0.0, friction=0.85):
        self.ground_y = float(ground_y)
        if self._handle:
            _LIB.kinetra_physics_set_ground(self._handle, ground_y, restitution, friction)

    def step(self, targets, dt=1.0 / 60.0):
        targets = np.asarray(targets, dtype=np.float32)
        if not self.is_initialized:
            self.reset(targets)
            return self.pos.copy()

        dt = max(0.001, min(0.05, float(dt)))

        if self._handle:
            out_p = np.empty_like(self.pos)
            out_v = np.empty_like(self.vel)
            t_ptr = targets.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            p_ptr = out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            v_ptr = out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _LIB.kinetra_physics_step(self._handle, dt, t_ptr, p_ptr, v_ptr)
            self.pos = out_p
            self.vel = out_v
            return self.pos.copy()

        # NumPy fallback
        err = targets - self.pos
        acc = err * self.stiffness - self.vel * self.damping
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Floor non-penetration constraint
        below = self.pos[:, 1] < self.ground_y
        if np.any(below):
            self.pos[below, 1] = self.ground_y
            neg_v = below & (self.vel[:, 1] < 0)
            self.vel[neg_v, 1] = 0.0
            self.vel[neg_v, 0] *= 0.15
            self.vel[neg_v, 2] *= 0.15

        return self.pos.copy()


def fast_lbs(rest_v, rest_n, joint_matrices, joint_nodes, weights):
    """Linear blend skinning with C++ acceleration when available."""
    rest_v = np.ascontiguousarray(rest_v, dtype=np.float32)
    rest_n = np.ascontiguousarray(rest_n, dtype=np.float32)
    joint_matrices = np.ascontiguousarray(joint_matrices, dtype=np.float32)
    joint_nodes = np.ascontiguousarray(joint_nodes, dtype=np.int32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    nv = len(rest_v)

    if _LIB is not None:
        out_v = np.empty_like(rest_v)
        out_n = np.empty_like(rest_n)
        _LIB.kinetra_fast_lbs(
            rest_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rest_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            joint_matrices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            joint_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nv,
            out_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out_v, out_n

    # NumPy implementation
    out_v = np.zeros_like(rest_v)
    out_n = np.zeros_like(rest_n)
    for k in range(4):
        nid = joint_nodes[:, k]
        w = weights[:, k:k + 1]
        valid = (nid >= 0)
        nid_safe = np.where(valid, nid, 0)
        M = joint_matrices[nid_safe]  # (N, 4, 4)
        R = M[:, :3, :3]
        T = M[:, :3, 3]
        v_def = np.einsum("nij,nj->ni", R, rest_v) + T
        n_def = np.einsum("nij,nj->ni", R, rest_n)
        out_v += w * np.where(valid[:, None], v_def, rest_v)
        out_n += w * np.where(valid[:, None], n_def, rest_n)
    norm = np.linalg.norm(out_n, axis=1, keepdims=True) + 1e-9
    out_n /= norm
    return out_v, out_n
