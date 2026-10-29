import numpy as np
from kinetra.ai import synth
from kinetra.ai.hand_prior import *
from kinetra.ai.body_prior import *
from kinetra.core.hand import HandSolver, hand_fk, FINGERS
from kinetra.core.mathx import q_axis_angle


def test_features_roundtrip_through_quaternions():
    rng = np.random.default_rng(0)
    for side in "LR":
        x = synth.random_hand_pose(rng)
        x2 = hand_features(features_to_quats(x, side), side)
        assert np.abs(x - x2).max() < 1e-6, (side, np.abs(x - x2).max())


def test_synthetic_poses_are_anatomically_possible():
    rng = np.random.default_rng(1)
    X = np.array([synth.random_hand_pose(rng) for _ in range(500)])
    assert X[:, 2::4].max() < np.radians(125) and X[:, 2::4].min() > np.radians(-5)      # PIP flexion range


_HAND = None


def hand_prior():
    global _HAND
    if _HAND is None:
        rng = np.random.default_rng(0)
        X = np.array([synth.random_hand_pose(rng) for _ in range(6000)])
        _HAND = HandPosePrior.train(X, synth.hand_sequences(rng, 30, 150), K=32, iters=25)
    return _HAND


def test_hidden_fingers_are_inferred_from_visible_ones_not_frozen():
    prior = hand_prior()
    rng = np.random.default_rng(3)
    err_ai, err_freeze = [], []
    for name in ("fist", "open", "peace", "grab", "rock"):
        f = prior.new_filter()
        for _ in range(8):                                        # long open hand first: last known pose = open
            f.update(synth.gesture_vec("open", rng, 0.01), np.ones(5))
        truth = synth.gesture_vec(name, rng, 0.01)
        conf = np.array([1.0, 1.0, 1.0, 0.0, 0.0])               # ring and pinky are hidden, the rest is seen
        for _ in range(10):
            out = f.update(truth, conf)
        hidden = np.r_[12:20]
        last_known = synth.gesture_vec("open", rng, 0.0)
        err_ai.append(np.abs(out["pose"][hidden] - truth[hidden]).mean())
        err_freeze.append(np.abs(last_known[hidden] - truth[hidden]).mean())
    assert np.mean(err_ai) < 0.6 * np.mean(err_freeze), (np.mean(err_ai), np.mean(err_freeze))


def test_reappearing_fingers_do_not_continue_from_the_old_pose():
    prior = hand_prior()
    rng = np.random.default_rng(4)
    f = prior.new_filter()
    for _ in range(10):
        f.update(synth.gesture_vec("fist", rng, 0.01), np.ones(5))
    for _ in range(40):
        f.update(None, None)                                      # hand leaves the picture
    truth = synth.gesture_vec("open", rng, 0.01)
    errs = [np.abs(f.update(truth, np.ones(5))["pose"] - truth).mean() for _ in range(4)]
    assert errs[-1] < 0.05 and errs[-1] < errs[0] + 1e-9


def test_untrusted_measurements_are_pulled_toward_plausible_poses():
    prior = hand_prior()
    rng = np.random.default_rng(5)
    f = prior.new_filter()
    truth = synth.gesture_vec("fist", rng, 0.01)
    for _ in range(10):
        f.update(truth, np.ones(5))
    bad = truth.copy()
    bad[4::4] = np.radians(-60)                                   # impossible hyper-extended fingers, low trust
    out = f.update(bad, np.full(5, 0.15))
    assert np.abs(out["pose"] - truth).mean() < np.abs(bad - truth).mean() * 0.6


def test_finger_confidence_flags_wrong_bone_lengths():
    s = HandSolver()
    P = hand_fk({j: np.array([1.0, 0, 0, 0]) for j in __import__("kinetra.core.hand", fromlist=["FINGER_JOINTS"]).FINGER_JOINTS}, s.rest_offsets())
    bones = s.bone_lengths()
    assert finger_confidence(P, bones).min() > 0.95
    bad = P.copy()
    bad[8] = bad[7] + (bad[8] - bad[7]) * 2.0                     # index tip stretched
    c = finger_confidence(bad, bones)
    assert c[1] < 0.5 and c[2] > 0.95


def test_body_prior_predicts_hidden_joints_better_than_holding_the_last_value():
    rng = np.random.default_rng(0)
    seqs = [synth.body_sequence(rng, T=90) for _ in range(60)]
    prior = BodyPrior.train(seqs, n_windows=12000, epochs=16, hidden=192)
    test = [synth.body_sequence(np.random.default_rng(100 + i), T=60) for i in range(15)]
    e_ai, e_hold = [], []
    for s in test:
        for e in range(20, 60, 6):
            w = s[e - T_WIN:e]
            pel, B, sc = frame_basis(w[-1])
            C = to_canon(w, pel, B, sc)
            M = np.ones((T_WIN, NJ))
            hide = [4, 2]                                         # left wrist and elbow vanish for the last 5 frames
            M[-5:, hide] = 0
            done = prior.complete(C, M)
            held = C.copy()
            held[-5:, hide] = C[-6, hide]
            e_ai.append(np.abs(done[-1, hide] - C[-1, hide]).mean())
            e_hold.append(np.abs(held[-1, hide] - C[-1, hide]).mean())
    assert np.mean(e_ai) < np.mean(e_hold), (np.mean(e_ai), np.mean(e_hold))


def test_live_body_prior_fills_only_hidden_joints():
    rng = np.random.default_rng(1)
    seqs = [synth.body_sequence(rng, T=60) for _ in range(20)]
    live = LiveBodyPrior(BodyPrior.train(seqs, n_windows=1500, epochs=3, hidden=96))
    s = synth.body_sequence(np.random.default_rng(9), T=40)
    out_changed = []
    for t in range(40):
        J33 = np.zeros((33, 3))
        J33[list(MP_INDEX)] = s[t]
        vis = np.ones(33)
        if t >= 20:
            vis[16] = 0.0                                         # right wrist hidden
        J, rep = live.fill(J33, vis)
        if t >= 20:
            out_changed.append(rep[16])
            assert not rep[[11, 12, 13, 14, 15, 23, 24]].any()
            assert np.allclose(J[15], J33[15])
    assert all(out_changed)
