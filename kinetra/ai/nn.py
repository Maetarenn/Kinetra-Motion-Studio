"""Minimal numpy neural network (dense layers, ReLU, Adam). No framework needed, so the AI features install
everywhere and train on any CPU."""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, x, keep=False):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        return (x, acts) if keep else x

    def train_step(self, x, y, w=None, lr=1e-3):
        """One Adam step on weighted MSE. Returns the loss."""
        out, acts = self.forward(x, keep=True)
        w = np.ones_like(y) if w is None else w
        err = (out - y) * w
        loss = float(np.mean(err * (out - y)))
        d = 2.0 * err / err.size
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i], gb[i] = acts[i].T @ d, d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        self._t += 1
        params, grads = self.W + self.b, gW + gb
        for k, (p, g) in enumerate(zip(params, grads)):
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * g * g
            mh, vh = self._m[k] / (1 - 0.9 ** self._t), self._v[k] / (1 - 0.999 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def state(self):
        d = {"sizes": np.array(self.sizes)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        return d

    @classmethod
    def from_state(cls, d):
        m = cls([int(s) for s in d["sizes"]])
        m.W = [np.array(d[f"W{i}"]) for i in range(len(m.W))]
        m.b = [np.array(d[f"b{i}"]) for i in range(len(m.b))]
        return m
