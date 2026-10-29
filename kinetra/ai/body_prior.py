"""Body motion prior: a masked denoising network that completes joints the camera cannot see.

It learns how the 12 main joints (shoulders, elbows, wrists, hips, knees, ankles) move together, in a frame
that follows the pelvis (position, heading and size removed). Given the last T frames with a visibility mask it
predicts the hidden joints of the current frame - e.g. a wrist behind the back continues along a plausible arm
path instead of freezing."""
import numpy as np

from .nn import MLP

T_WIN = 9
NJ = 12                                  # order: shoulder L,R  elbow L,R  wrist L,R  hip L,R  knee L,R  ankle L,R
MP_INDEX = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)      # the same joints in MediaPipe numbering


def frame_basis(J):
    """pelvis (3,), basis (3,3 columns x,y,z), scale for a (12,3) pose (y up)."""
    pelvis = (J[6] + J[7]) / 2
    ex = J[6] - J[7]
    ex = np.array([ex[0], 0.0, ex[2]])
    n = np.linalg.norm(ex)
    ex = ex / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.cross(ex, ey)
    scale = float(np.linalg.norm((J[0] + J[1]) / 2 - pelvis))
    return pelvis, np.stack([ex, ey, ez], axis=1), max(scale, 0.1)


def to_canon(J, pelvis, B, scale):
    return ((J - pelvis) @ B) / scale


def from_canon(C, pelvis, B, scale):
    return (C * scale) @ B.T + pelvis


class BodyPrior:
    def __init__(self, net, mean=None):
        self.net = net

    @staticmethod
    def make_windows(seqs, rng, n):
        """seqs: list of (T,12,3) metric sequences -> canonical windows (n, T_WIN, 12, 3) in the last frame's frame."""
        out = []
        for _ in range(n):
            s = seqs[rng.integers(len(seqs))]
            if len(s) <= T_WIN:
                continue
            e = int(rng.integers(T_WIN, len(s) + 1))
            w = s[e - T_WIN:e]
            pel, B, sc = frame_basis(w[-1])
            out.append(to_canon(w, pel, B, sc))
        return np.array(out)

    @staticmethod
    def _held(W, M):
        """Forward-fill hidden joints with their last visible value (zeros if never seen in the window)."""
        H = W * M[..., None]
        for t in range(1, W.shape[1]):
            H[:, t] = np.where(M[:, t, :, None] > 0, W[:, t], H[:, t - 1])
        return H

    @classmethod
    def _inputs(cls, W, M):
        H = cls._held(W, M)
        return np.concatenate([H.reshape(len(W), -1), M.reshape(len(W), -1)], axis=1), H

    @classmethod
    def train(cls, seqs, n_windows=30000, epochs=25, batch=256, hidden=256, seed=0, progress=None):
        rng = np.random.default_rng(seed)
        W = cls.make_windows(seqs, rng, n_windows)
        net = MLP([T_WIN * NJ * 4, hidden, hidden // 2, hidden, T_WIN * NJ * 3], seed)
        steps = epochs * (len(W) // batch)
        k = 0
        for ep in range(epochs):
            order = rng.permutation(len(W))
            for i in range(0, len(W) - batch + 1, batch):
                x = W[order[i:i + batch]]
                M = np.ones((len(x), T_WIN, NJ))
                for r in range(len(x)):                       # hide 1-4 joints, whole window or only the recent frames
                    js = rng.choice(NJ, rng.integers(1, 5), replace=False)
                    t0 = 0 if rng.random() < 0.5 else int(rng.integers(1, T_WIN - 1))
                    M[r, t0:, js] = 0.0
                noisy = x + rng.normal(0, 0.01, x.shape)
                wgt = np.where(np.broadcast_to(M[..., None], x.shape) > 0, 1.0, 4.0).reshape(len(x), -1)
                inp, held = cls._inputs(noisy, M)
                net.train_step(inp, (x - held).reshape(len(x), -1), wgt, lr=2e-3 * (1 - 0.8 * k / steps))   # learn the correction to "hold"
                k += 1
            if progress:
                progress((ep + 1) / epochs)
        return cls(net)

    def complete(self, W, M):
        """W: (T,12,3) canonical window, M: (T,12) 1 = observed. Returns the completed window."""
        inp, held = self._inputs(W[None], M[None])
        pred = held[0] + self.net.forward(inp)[0].reshape(T_WIN, NJ, 3)
        return np.where(M[..., None] > 0, W, pred)

    def save(self, path):
        np.savez_compressed(path, **self.net.state())

    @classmethod
    def load(cls, path):
        return cls(MLP.from_state(np.load(path)))


class LiveBodyPrior:
    """Streaming wrapper: feed 33 MediaPipe-style joints + visibility every frame, get the completed joints."""
    VIS = 0.5

    def __init__(self, prior):
        self.prior, self.buf, self.mbuf = prior, [], []
        self.basis = None

    def reset(self):
        self.buf, self.mbuf, self.basis = [], [], None

    def fill(self, J33, vis):
        """J33: (33,3) y-up metres; vis: (33,) -> (J33 with hidden joints predicted, bool mask of replaced joints)."""
        J = np.asarray(J33, float).copy()
        cur = J[list(MP_INDEX)]
        m = (np.asarray(vis)[list(MP_INDEX)] > self.VIS).astype(float)
        core = m[[0, 1, 6, 7]].all()                          # shoulders + hips decide the frame
        if core or self.basis is None:
            self.basis = frame_basis(cur) if core else frame_basis(np.where(np.isnan(cur), 0, cur))
        pel, B, sc = self.basis
        self.buf.append(to_canon(np.nan_to_num(cur), pel, B, sc) if self.basis else cur)
        self.mbuf.append(m)
        del self.buf[:-T_WIN], self.mbuf[:-T_WIN]
        replaced = np.zeros(33, dtype=bool)
        if len(self.buf) < T_WIN or m.all():
            return J, replaced
        W = np.array([to_canon(from_canon(b, *self._frame_of(i)), pel, B, sc) for i, b in enumerate(self.buf)])
        M = np.array(self.mbuf)
        done = self.prior.complete(W, M)[-1]
        world = from_canon(done, pel, B, sc)
        for k, mp in enumerate(MP_INDEX):
            if m[k] < 0.5:
                J[mp], replaced[mp] = world[k], True
        return J, replaced

    def _frame_of(self, i):
        return self.basis                                       # window already stored in the current frame's basis
