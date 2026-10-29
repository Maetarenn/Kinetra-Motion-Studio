"""Body solver.

Input : 33 joints (y up, metres, +x = subject's left) from a single camera (hip-centred) or from
        multi-camera triangulation (absolute, floor = y 0).
Stages: One-Euro filter -> auto/manual calibration (bone lengths, ground, neutral head)
        -> bone-length enforcement -> ground -> hinge limits -> root -> foot lock with two-bone leg IK
        -> local quaternions with real twist (pelvis, chest, head, arms, legs, feet).
"""
import math
from collections import deque

import numpy as np

from .constants import (ARMS, LEGS, HEEL, TOE, FOOT_IDS, HIP_MID, SHO_MID, EAR_MID, NOSE, EAR_L, EAR_R,
                        SHO_L, SHO_R, HIP_L, HIP_R, JOINT_ORDER, DEFAULT_LENGTHS)
from .mathx import (OneEuroFilter, unit, q_mul, q_conj, q_axis_angle, q_from_vectors, q_rotate, q_mean,
                    q_pow, q_slerp, frame_quat, mat_to_quat, two_bone_ik, DOWN, FWD, UP, IDENTITY_Q)
from .physics_engine import PhysicsEngine

REST_AXIS_ARM = np.array([-1.0, 0.0, 0.0])   # elbow flexion axis with the arm hanging down
REST_AXIS_LEG = np.array([1.0, 0.0, 0.0])    # knee flexion axis with the leg hanging down

# Natural human relaxed rest pose (rahat, insani durus: hafif kol acikligi, yumusak dirsek/diz mikro-fleksiyonu)
RELAXED_DEFAULTS = {
    "upper_arm_L": q_axis_angle([0, 0, 1], math.radians(-8.0)),
    "upper_arm_R": q_axis_angle([0, 0, 1], math.radians(8.0)),
    "forearm_L": q_axis_angle([1, 0, 0], math.radians(10.0)),
    "forearm_R": q_axis_angle([1, 0, 0], math.radians(10.0)),
    "thigh_L": q_axis_angle([0, 0, 1], math.radians(2.5)),
    "thigh_R": q_axis_angle([0, 0, 1], math.radians(-2.5)),
    "shin_L": q_axis_angle([1, 0, 0], math.radians(3.5)),
    "shin_R": q_axis_angle([1, 0, 0], math.radians(3.5)),
    "foot_L": IDENTITY_Q.copy(),
    "foot_R": IDENTITY_Q.copy(),
}


def _canon(q):
    return -q if q[0] < 0 else q


class BodySolver:
    V_LOCK = 0.06            # m/s : foot is considered planted below this speed
    V_RELEASE = 0.45         # m/s : planted foot is released above this speed
    LOCK_FRAMES = 3
    LOCK_HEIGHT = 0.12       # only lock feet that are actually near the floor
    LOCK_DRIFT = 0.08        # release if the free foot walks this far from the lock point
    MAX_FLEX = math.radians(160.0)  # full natural elbow/knee flexion up to 160 deg for head touching
    CALIB_FRAMES = 60

    def __init__(self, f_min=1.0, beta=5.0, use_physics=True):
        self.filter = OneEuroFilter(f_min, beta, vector=True)
        self.root_filter = OneEuroFilter(f_min, beta, vector=True)
        self.use_physics = use_physics
        self.physics = PhysicsEngine(num_joints=36, ground_y=0.0)
        self.reset()

    # ------------------------------------------------------------------------------------------
    def reset(self):
        self.filter.reset()
        self.root_filter.reset()
        if hasattr(self, "physics") and self.physics:
            self.physics.reset()
        self.root = np.zeros(2)
        self.last_J = None
        self.prev_t = None
        self.prev_feet = {}
        self.low_count = {"L": 0, "R": 0}
        self.locked = {"L": False, "R": False}
        self.anchor = {"L": np.zeros(3), "R": np.zeros(3)}
        self.lock_w = {"L": 0.0, "R": 0.0}
        self.ground_hist = deque(maxlen=150)
        self.ground_y = 0.0
        self.seg = None
        self.head_neutral = None
        self.ground_fixed = None
        self.start_calibration()

    def start_calibration(self):
        self.calib_left = self.CALIB_FRAMES
        self._cal_len, self._cal_ground, self._cal_head = [], [], []
        self.seg = None
        self.head_neutral = None

    def set_filter(self, beta, f_min):
        for f in (self.filter, self.root_filter):
            f.beta, f.f_min = beta, f_min

    @property
    def calibrating(self):
        return self.calib_left > 0

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _measure(J):
        d = lambda a, b: float(np.linalg.norm(J[a] - J[b]))
        m = {"spine": d(HIP_MID, SHO_MID), "sw": d(SHO_L, SHO_R) / 2, "hw": d(HIP_L, HIP_R) / 2}
        for s in "LR":
            a, b, c, _ = ARMS[s]
            m["upper_arm_" + s], m["forearm_" + s] = d(a, b), d(b, c)
            a, b, c, _ = LEGS[s]
            m["thigh_" + s], m["shin_" + s] = d(a, b), d(b, c)
            m["foot_" + s] = d(LEGS[s][2], TOE[s])
        return m

    def lengths(self):
        """Per-segment lengths (metres) in the layout used by the exporters."""
        m = self.seg if self.seg is not None else self._measure(self.last_J) if self.last_J is not None else None
        if m is None:
            return dict(DEFAULT_LENGTHS)
        avg = lambda k: (m[k + "_L"] + m[k + "_R"]) / 2
        return {"spine": m["spine"], "head": DEFAULT_LENGTHS["head"], "sw": m["sw"], "hw": m["hw"],
                "upper_arm": avg("upper_arm"), "forearm": avg("forearm"), "thigh": avg("thigh"),
                "shin": avg("shin"), "foot": avg("foot")}

    def _enforce_lengths(self, J):
        if self.seg is None:
            return
        for chain, names in ((ARMS, ("upper_arm_", "forearm_")), (LEGS, ("thigh_", "shin_"))):
            for s, (a, b, c, deps) in chain.items():
                u, v = unit(J[b] - J[a], DOWN), unit(J[c] - J[b], DOWN)
                nb = J[a] + u * self.seg[names[0] + s]
                nc = nb + v * self.seg[names[1] + s]
                shift = nc - J[c]
                for i in (c,) + tuple(deps):
                    J[i] += shift
                J[b] = nb

    def _limit_hinges(self, J):
        for chain in (ARMS, LEGS):
            for a, b, c, deps in chain.values():
                u, v = J[b] - J[a], J[c] - J[b]
                lu, lv = np.linalg.norm(u), np.linalg.norm(v)
                if lu < 1e-6 or lv < 1e-6:
                    continue
                un, vn = u / lu, v / lv
                ang = math.acos(float(np.clip(np.dot(un, vn), -1.0, 1.0)))
                axis = np.cross(un, vn)
                if ang > self.MAX_FLEX and np.linalg.norm(axis) > 1e-6:
                    new_v = q_rotate(q_axis_angle(axis, self.MAX_FLEX), un) * lv
                    shift = (J[b] + new_v) - J[c]
                    for i in (c,) + tuple(deps):
                        J[i] += shift

    # ------------------------------------------------------------------------------------------
    def _foot_lock(self, J, dt):
        for s, (hip, knee, ank, _) in LEGS.items():
            heel, toe = HEEL[s], TOE[s]
            pts = J[[ank, heel, toe]].copy()
            prev = self.prev_feet.get(s)
            self.prev_feet[s] = pts
            if prev is None or dt <= 0:
                continue
            vel = float(np.linalg.norm(pts - prev, axis=1).max()) / dt
            sole = float(min(pts[1, 1], pts[2, 1]))
            low = vel < self.V_LOCK and sole < self.LOCK_HEIGHT
            self.low_count[s] = self.low_count[s] + 1 if low else 0
            if not self.locked[s] and self.low_count[s] >= self.LOCK_FRAMES:
                self.locked[s] = True
                self.anchor[s] = np.array([pts[0, 0], pts[0, 1] - sole, pts[0, 2]])
            elif self.locked[s]:
                drift = float(np.linalg.norm(pts[0, [0, 2]] - self.anchor[s][[0, 2]]))
                if drift > self.LOCK_DRIFT or vel > self.V_RELEASE:
                    self.locked[s], self.low_count[s] = False, 0
            w = self.lock_w[s]
            w = min(1.0, w + 0.34) if self.locked[s] else max(0.0, w - 0.34)
            self.lock_w[s] = w
            if w <= 0.0:
                continue
            target = pts[0] * (1.0 - w) + self.anchor[s] * w
            l1, l2 = float(np.linalg.norm(J[knee] - J[hip])), float(np.linalg.norm(J[ank] - J[knee]))
            new_knee, new_ank = two_bone_ik(J[hip], target, l1, l2, J[knee])
            shift = new_ank - J[ank]
            for i in (heel, toe):
                J[i] += shift
            J[ank], J[knee] = new_ank, new_knee

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _limb_world(dirv, rest_dir, rest_axis, meas_axis, fade):
        """Swing (rest_dir -> dirv) followed by a twist about dirv that aligns the rest bend axis with the
        measured one. fade (0..1) fades the twist out when the joint is straight and the axis is unreliable."""
        qs = q_from_vectors(rest_dir, dirv)
        if meas_axis is not None and fade > 0.0:
            r = q_rotate(qs, rest_axis)
            r = r - dirv * float(np.dot(r, dirv))
            a = meas_axis - dirv * float(np.dot(meas_axis, dirv))
            if np.linalg.norm(r) > 1e-6 and np.linalg.norm(a) > 1e-6:
                r, a = unit(r), unit(a)
                ang = math.atan2(float(np.dot(np.cross(r, a), dirv)), float(np.dot(r, a)))
                qs = q_mul(q_axis_angle(dirv, ang * fade), qs)
        return qs

    def _hinge(self, parent_world, parent_dir, child_dir, rest_axis):
        ax = np.cross(parent_dir, child_dir)
        ang = min(math.acos(float(np.clip(np.dot(parent_dir, child_dir), -1.0, 1.0))), self.MAX_FLEX)
        axis_w = unit(ax, q_rotate(parent_world, rest_axis))
        return q_axis_angle(q_rotate(q_conj(parent_world), axis_w), ang)

    @staticmethod
    def _head_raw(J, chest_w):
        eye_mid = (J[2] + J[5]) / 2.0
        ear_mid = J[EAR_MID]
        head_c = (eye_mid + ear_mid) / 2.0
        v_eye = J[2] - J[5]
        v_ear = J[EAR_L] - J[EAR_R]
        x = unit(v_eye * 0.6 + v_ear * 0.4, [1.0, 0.0, 0.0])
        z_hint = J[NOSE] - head_c
        z = unit(z_hint - x * float(np.dot(z_hint, x)), FWD)
        y = np.cross(z, x)
        head_w = mat_to_quat(np.stack([x, y, z], axis=1))
        return q_mul(q_conj(chest_w), head_w)

    def preset_from_sequence(self, Js, absolute=False):
        """Offline calibration from a whole take: Js is a list of (36,3) joint arrays (filtered, hip-centred or
        absolute). Sets bone lengths, ground height and the neutral head from the take itself."""
        ms = [self._measure(J) for J in Js]
        self.seg = {k: float(np.median([m[k] for m in ms])) for k in ms[0]}
        self.ground_fixed = None if absolute else float(np.percentile([J[list(FOOT_IDS), 1].min() for J in Js], 20))
        heads = []
        for J in Js[:self.CALIB_FRAMES]:
            spine = unit(J[SHO_MID] - J[HIP_MID])
            heads.append(self._head_raw(J, frame_quat(J[SHO_L] - J[SHO_R], spine)))
        self.head_neutral = q_mean(heads)
        self.calib_left = 0

    def _pose(self, J, vis=None):
        spine = unit(J[SHO_MID] - J[HIP_MID])
        hips_w = frame_quat(J[HIP_L] - J[HIP_R], spine)
        chest_w = frame_quat(J[SHO_L] - J[SHO_R], spine)

        head_raw = self._head_raw(J, chest_w)
        head_loc = head_raw if self.head_neutral is None else q_mul(head_raw, q_conj(self.head_neutral))
        if self.calib_left > 0:
            self._cal_head.append(head_raw)

        quats = {"Hips": hips_w, "spine": q_mul(q_conj(hips_w), chest_w), "head": head_loc}
        world = {"Hips": hips_w, "chest": chest_w}
        for s in "LR":
            for chain, parent_w, rest_axis, up_n, lo_n in ((ARMS, chest_w, REST_AXIS_ARM, "upper_arm_", "forearm_"),
                                                           (LEGS, hips_w, REST_AXIS_LEG, "thigh_", "shin_")):
                a, b, c, _ = chain[s]
                ud = unit(J[b] - J[a], DOWN)
                fd = unit(J[c] - J[b], ud)
                ax = np.cross(ud, fd)
                sin_f = float(np.linalg.norm(ax))
                meas = ax / sin_f if sin_f > 1e-3 else None
                qw = self._limb_world(ud, DOWN, rest_axis, meas, min(1.0, sin_f / 0.35))
                q_lo = self._hinge(qw, ud, fd, rest_axis)

                # Confidence blending: relax unobserved / hidden limbs to clean straight rest pose
                limb_conf = 1.0
                if vis is not None:
                    try:
                        v_b, v_c = float(vis[b]), float(vis[c])
                        limb_conf = float(np.clip((min(v_b, v_c) - 0.25) / 0.35, 0.0, 1.0))
                    except (IndexError, TypeError):
                        limb_conf = 1.0

                q_up = _canon(q_mul(q_conj(parent_w), qw))
                q_down = _canon(q_lo)
                if limb_conf < 1.0:
                    target_up = RELAXED_DEFAULTS.get(up_n + s, IDENTITY_Q)
                    target_down = RELAXED_DEFAULTS.get(lo_n + s, IDENTITY_Q)
                    q_up = _canon(q_slerp(target_up, q_up, limb_conf))
                    q_down = _canon(q_slerp(target_down, q_down, limb_conf))
                    qw = q_mul(parent_w, q_up)

                quats[up_n + s] = q_up
                quats[lo_n + s] = q_down
                world[up_n + s], world[lo_n + s] = qw, q_mul(qw, q_down)
                if chain is LEGS:                                   # foot, parented to the shin
                    shin_w = q_mul(qw, q_down)
                    ank, heel, toe = c, HEEL[s], TOE[s]
                    fdir = unit(J[toe] - J[ank], FWD)
                    up_hint = J[ank] - (J[heel] + J[toe]) / 2
                    fw = self._limb_world(fdir, FWD, UP, unit(up_hint, UP), 1.0)
                    q_foot = _canon(q_mul(q_conj(shin_w), fw))
                    if limb_conf < 1.0:
                        target_foot = RELAXED_DEFAULTS.get("foot_" + s, IDENTITY_Q)
                        q_foot = _canon(q_slerp(target_foot, q_foot, limb_conf))
                        fw = q_mul(shin_w, q_foot)
                    quats["foot_" + s] = q_foot
                    world["foot_" + s] = fw
        return {k: _canon(quats[k]) for k in JOINT_ORDER}, world

    # ------------------------------------------------------------------------------------------
    def process(self, J33, t, root=None, vis=None, absolute=False, ground=None, prefiltered=False):
        """J33: (33,3) y-up joints. root: (x, z) pelvis translation for hip-centred input.
        absolute=True means J33 already lives in world space (multi camera) and ground defaults to y=0."""
        f = np.asarray(J33, dtype=np.float64) if prefiltered else self.filter(J33, t)
        J = np.empty((36, 3))
        J[:33] = f
        J[HIP_MID] = (f[HIP_L] + f[HIP_R]) / 2
        J[SHO_MID] = (f[SHO_L] + f[SHO_R]) / 2
        J[EAR_MID] = (f[EAR_L] + f[EAR_R]) / 2

        msg = None
        foot_min = float(J[list(FOOT_IDS), 1].min())
        if self.calib_left > 0:
            good = vis is None or float(np.mean(np.asarray(vis)[[11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]])) > 0.6
            if good:
                self._cal_len.append(self._measure(J))
                self._cal_ground.append(foot_min)
                self.calib_left -= 1
                if self.calib_left == 0:
                    self.seg = {k: float(np.median([m[k] for m in self._cal_len])) for k in self._cal_len[0]}
                    g = float(np.median(self._cal_ground))
                    self.ground_hist.clear()
                    self.ground_hist.extend([g] * 40)
                    msg = "calib_done"
        if ground is None:
            ground = self.ground_fixed
        if ground is not None:
            self.ground_y = float(ground)
        elif absolute:
            self.ground_y = 0.0
        else:
            self.ground_hist.append(foot_min)
            self.ground_y = float(np.percentile(self.ground_hist, 20))
        J[:, 1] -= self.ground_y

        dt = (t - self.prev_t) if self.prev_t is not None else 0.0
        self.prev_t = t
        if self.use_physics and not self.calibrating:
            J = self.physics.step(J, dt=dt if dt > 0 else 1.0 / 60.0)
            J[HIP_MID] = (J[HIP_L] + J[HIP_R]) / 2
            J[SHO_MID] = (J[SHO_L] + J[SHO_R]) / 2
            J[EAR_MID] = (J[EAR_L] + J[EAR_R]) / 2

        self._enforce_lengths(J)
        self._limit_hinges(J)
        if not absolute:
            if root is not None:
                self.root = self.root_filter(np.asarray(root, dtype=np.float64).reshape(1, 2), t)[0]
            J[:, 0] += self.root[0]
            J[:, 2] += self.root[1]
        self._foot_lock(J, dt)
        min_foot_y = float(J[list(FOOT_IDS), 1].min())
        if min_foot_y < -0.005:
            J[:, 1] -= min_foot_y
        self.last_J = J.copy()
        quats, world = self._pose(J, vis=vis)
        if msg and self._cal_head:
            self.head_neutral = q_mean(self._cal_head)
            self._cal_head = []
            quats["head"] = _canon(q_mul(quats["head"], q_conj(self.head_neutral)))
        return {"joints": J, "quats": quats, "world": world, "locks": np.array([self.locked["L"], self.locked["R"]]),
                "calib_msg": msg, "lengths": self.lengths(), "ground_y": self.ground_y}
