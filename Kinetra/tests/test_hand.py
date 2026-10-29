import numpy as np
from kinetra.core.hand import *
from kinetra.core.mathx import q_axis_angle, q_mul, q_to_mat


def make_hand(rng, curl=1.0):
    solver = HandSolver()
    off = solver.rest_offsets()
    quats = {}
    for f in FINGERS:
        a1, a2, a3 = rng.uniform(0, 80, 3) * curl * np.radians(1) * np.array([1, 1.2, 0.8])
        abd = np.radians(rng.uniform(-12, 12))
        quats[f + "_1"] = q_mul(q_axis_angle([0, 0, 1], a1), q_axis_angle([1, 0, 0], abd))
        quats[f + "_2"] = q_axis_angle([0, 0, 1], a2)
        quats[f + "_3"] = q_axis_angle([0, 0, 1], a3)
    return solver, off, quats


def random_rot(rng):
    q = rng.normal(size=4)
    return q_to_mat(q / np.linalg.norm(q))


def test_fingers_roundtrip_with_random_palm_pose():
    rng = np.random.default_rng(5)
    for trial in range(20):
        solver, off, quats = make_hand(rng, curl=rng.uniform(0.0, 1.0))
        Pl = hand_fk(quats, off)
        R0, t = random_rot(rng), rng.normal(size=3)
        P = Pl @ R0.T + t
        out = solver.solve(P, learn=False)
        assert np.allclose(out["palm_R"], R0, atol=1e-6)
        rec = hand_fk(out["quats"], off)
        assert np.abs(rec - Pl).max() < 1e-4, (trial, np.abs(rec - Pl).max())


def test_template_is_learned_from_the_users_hand():
    rng = np.random.default_rng(1)
    solver, off, quats = make_hand(rng)
    Pl = hand_fk(quats, off) * 1.15                     # a bigger hand
    for _ in range(60):
        solver.solve(Pl + rng.normal(0, 1e-4, Pl.shape))
    bones = solver.bone_lengths()
    assert abs(bones["middle"][0] - DEFAULT_BONES["middle"][0] * 1.15) < 2e-3
    assert abs(solver.palm_offsets()["middle"][1] + 0.092 * 1.15) < 2e-3


def test_curl_is_zero_for_an_open_hand_and_high_for_a_fist():
    rng = np.random.default_rng(2)
    s = HandSolver()
    off = s.rest_offsets()
    open_q = {j: np.array([1.0, 0, 0, 0]) for j in FINGER_JOINTS}
    fist = {}
    for f in FINGERS:
        fist[f + "_1"], fist[f + "_2"], fist[f + "_3"] = (q_axis_angle([0, 0, 1], np.radians(a)) for a in (85, 95, 65))
    assert max(s.solve(hand_fk(open_q, off), learn=False)["curl"].values()) < 0.05
    assert min(v for k, v in s.solve(hand_fk(fist, off), learn=False)["curl"].items() if k != "thumb") > 0.6
