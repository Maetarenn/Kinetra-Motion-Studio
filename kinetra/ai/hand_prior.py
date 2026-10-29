"""Finger pose prior: a Gaussian-mixture pose model with a hidden-Markov temporal model.

States are typical hand configurations (open, relaxed, fist, point, pinch, ...). Each frame the filter
  1. propagates the belief with the learned transition matrix (no freezing when fingers are hidden: the belief
     drifts toward the poses that are really likely),
  2. multiplies by how well the VISIBLE, confident finger angles fit each state,
  3. returns the most probable state's pose for every finger angle that could not be observed, and a
     confidence-weighted blend for the ones that were.
When a finger reappears it is compared against all states at once, so it can never 'continue from the old pose'."""
import math

import numpy as np

from ..core.hand import FINGERS, FINGER_JOINTS
from ..core.mathx import q_axis_angle, q_norm, IDENTITY_Q

NDIM = 20                                     # per finger: [mcp flexion, mcp abduction, pip flexion, dip flexion]
SIDE_SIGN = {"L": -1.0, "R": 1.0}             # flexion axis points to -Z on a left hand, +Z on a right hand
LIMITS = np.radians(np.array([[-10, 95, -35, 35, 0, 110, 0, 90],     # index..pinky: mcp lo/hi, abd lo/hi, pip, dip
                              [-10, 95, -35, 35, 0, 110, 0, 90]]))


def _rotvec(q):
    q = q_norm(q)
    if q[0] < 0:
        q = -q
    ang = 2.0 * math.acos(min(q[0], 1.0))
    s = math.sqrt(max(1.0 - q[0] ** 2, 0.0))
    return q[1:] / s * ang if s > 1e-9 else np.zeros(3)


def _from_rotvec(r):
    a = float(np.linalg.norm(r))
    return IDENTITY_Q.copy() if a < 1e-9 else q_axis_angle(r / a, a)


def _hinge_angle(q):
    q = q_norm(q)
    return 2.0 * math.atan2(q[3], q[0])         # signed rotation about Z


def hand_features(quats, side):
    """15 finger quaternions -> 20 side-normalised angles (radians)."""
    sg = SIDE_SIGN[side]
    x = np.zeros(NDIM)
    for i, f in enumerate(FINGERS):
        rv = _rotvec(quats[f + "_1"])
        x[4 * i:4 * i + 4] = [sg * rv[2], rv[0], sg * _hinge_angle(quats[f + "_2"]), sg * _hinge_angle(quats[f + "_3"])]
    return x


def features_to_quats(x, side):
    sg = SIDE_SIGN[side]
    q = {}
    for i, f in enumerate(FINGERS):
        m, a, p, d = x[4 * i:4 * i + 4]
        q[f + "_1"] = _from_rotvec(np.array([a, 0.0, sg * m]))
        q[f + "_2"] = q_axis_angle([0, 0, 1], sg * p)
        q[f + "_3"] = q_axis_angle([0, 0, 1], sg * d)
    return q


def finger_confidence(P, bones, prev=None):
    """Per-finger trust in the tracked landmarks (0..1): bone lengths must match the learned hand and the
    fingertips must not teleport. P: (21,3) wrist-relative landmarks, bones: {finger: (l1,l2,l3)}."""
    from ..core.hand import CHAINS
    conf = np.ones(5)
    for i, f in enumerate(FINGERS):
        c = CHAINS[f]
        lens = np.array([np.linalg.norm(P[c[k + 1]] - P[c[k]]) for k in range(3)])
        err = float(np.mean(np.abs(lens - np.array(bones[f])) / np.array(bones[f])))
        conf[i] = math.exp(-(err / 0.30) ** 2)
        if prev is not None:
            jump = float(np.linalg.norm(P[c[3]] - prev[c[3]]))
            if jump > 0.06:
                conf[i] *= math.exp(-((jump - 0.06) / 0.05) ** 2)
    return conf


# ----------------------------------------------------------------------------------------------
class HandPosePrior:
    def __init__(self, mu, var, w, A, pi):
        self.mu, self.var, self.w, self.A, self.pi = mu, var, w, A, pi
        self.K = len(mu)
        self._logdet = np.log(var).sum(axis=1)

    # -- training -----------------------------------------------------------------------------------
    @classmethod
    def train(cls, X, sequences=(), K=48, iters=40, seed=0, progress=None):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, float)
        mu = X[rng.choice(len(X), K, replace=False)].copy()                       # k-means++ style spread
        for _ in range(6):
            lab = np.argmin(((X[:, None] - mu[None]) ** 2).sum(-1), axis=1)
            for k in range(K):
                if (lab == k).any():
                    mu[k] = X[lab == k].mean(0)
        var = np.tile(X.var(axis=0) * 0.2 + 1e-3, (K, 1))
        w = np.full(K, 1.0 / K)
        for it in range(iters):
            ll = cls._loglik_all(X, mu, var, w)
            ll -= ll.max(axis=1, keepdims=True)
            r = np.exp(ll)
            r /= r.sum(axis=1, keepdims=True)
            nk = r.sum(0) + 1e-6
            mu = (r.T @ X) / nk[:, None]
            var = np.maximum((r.T @ (X ** 2)) / nk[:, None] - mu ** 2, 4e-4)
            w = nk / nk.sum()
            if progress:
                progress((it + 1) / iters)
        A = np.full((K, K), 0.02 / K)
        for seq in sequences:
            lab = np.argmax(cls._loglik_all(np.asarray(seq, float), mu, var, w), axis=1)
            for a, b in zip(lab[:-1], lab[1:]):
                A[a, b] += 1.0
        A += np.eye(K) * 1.0
        A /= A.sum(axis=1, keepdims=True)
        pi = np.linalg.matrix_power(A, 64)[0]
        return cls(mu, var, w, A, pi / pi.sum())

    @staticmethod
    def _loglik_all(X, mu, var, w):
        d = X[:, None, :] - mu[None]
        return np.log(w + 1e-12)[None] - 0.5 * ((d * d / var[None]).sum(-1) + np.log(var).sum(-1)[None])

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, var=self.var, w=self.w, A=self.A, pi=self.pi)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mu"], z["var"], z["w"], z["A"], z["pi"])

    # -- inference ----------------------------------------------------------------------------------
    def new_filter(self):
        return HandPoseFilter(self)


class HandPoseFilter:
    TEMPER = 0.4                    # emission temperature: avoids over-confident states from noisy landmarks

    def __init__(self, prior):
        self.p = prior
        self.belief = prior.pi.copy()
        self.pose = prior.mu[int(np.argmax(prior.pi))].copy()
        self.missing = 0.0

    def update(self, x=None, conf=None, motion=0.0, dt=1 / 30.0):
        """x: (20,) observed angles or None; conf: (5,) per-finger trust or None; motion: wrist speed (m/s)."""
        p = self.p
        b = self.belief @ p.A
        if x is None or conf is None or float(np.max(conf)) < 0.05:
            self.missing += dt
            eps = min(0.02 + 0.10 * self.missing + 0.04 * min(motion, 2.0), 0.5)      # unobserved: drift toward likely poses
            b = (1.0 - eps * dt * 30.0 * 0.1) * b + (eps * dt * 30.0 * 0.1) * p.pi
            observed = np.zeros(NDIM)
        else:
            self.missing = 0.0
            wt = np.repeat(np.asarray(conf, float), 4)
            d = x[None] - p.mu
            ll = -0.5 * (wt[None] * (d * d / p.var + np.log(p.var))).sum(-1) * self.TEMPER
            b = b * np.exp(ll - ll.max())
            observed = wt
        b = b + 1e-12
        self.belief = b / b.sum()
        k = int(np.argmax(self.belief))
        target = p.mu[k].copy()                                   # most probable state for unobserved angles
        if x is not None:
            # a pose that no state explains even loosely (> ~4 sigma on average) is mostly tracking error: trust it less
            fit = float(np.max(-0.5 * ((x[None] - p.mu) ** 2 / p.var).sum(-1) / NDIM))
            trust = observed * (1.0 if fit > -8.0 else math.exp((fit + 8.0) / 8.0))
            target = trust * x + (1.0 - trust) * target
        a = 0.85 if observed.max() > 0.5 else 0.25                 # smooth switches between states while hidden
        self.pose = a * target + (1.0 - a) * self.pose
        return {"pose": self.pose.copy(), "state": k, "confidence": float(self.belief[k]), "missing": self.missing}
