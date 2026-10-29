"""Pencil-dot / sticker marker tracking: black-hat detection + pyramidal LK flow with forward-backward check,
fused back into the face mesh."""
import cv2
import numpy as np

from .models import face_connections


def _conn_indices(name):
    return sorted({i for pair in face_connections(name) for i in pair})


def _hull(P, idxs):
    return cv2.convexHull(P[idxs, :2].astype(np.float32)).astype(np.int32)


class MarkerTracker:
    MAX_POINTS, ALPHA = 150, 0.6
    LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def __init__(self):
        self.clear()
        self._excl = None

    def clear(self):
        self.pts = None
        self.idx = self.w = self.r0 = self.prev = None
        self.n0, self.jitter = 0, 0.0

    @property
    def active(self):
        return self.pts is not None and len(self.pts) > 0

    @property
    def points(self):
        return None if not self.active else self.pts.reshape(-1, 2)

    def register(self, gray, P):
        """Detect dark marks on skin (excluding eyes/brows/lips), anchor each to its 3 nearest mesh vertices."""
        if self._excl is None:
            self._excl = [_conn_indices(n) for n in ("LIPS", "LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW")]
            self._oval = _conn_indices("FACE_OVAL")
        self.clear()
        fw = float(P[:468, 0].max() - P[:468, 0].min())
        if fw < 80:
            return 0
        mask = np.zeros(gray.shape, np.uint8)
        if self._oval:
            cv2.fillConvexPoly(mask, _hull(P, self._oval), 255)
            mask = cv2.erode(mask, np.ones((int(fw * 0.06) | 1,) * 2, np.uint8))
        else:
            cv2.fillConvexPoly(mask, _hull(P, list(range(468))), 255)
        for s in self._excl:
            if s:
                tmp = np.zeros_like(mask)
                cv2.fillConvexPoly(tmp, _hull(P, s), 255)
                mask[cv2.dilate(tmp, np.ones((int(fw * 0.03) | 1,) * 2, np.uint8)) > 0] = 0
        k = max(5, int(fw * 0.045)) | 1
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bh = cv2.GaussianBlur(bh, (3, 3), 0)
        bh[mask == 0] = 0
        if bh.max() < 30:
            return 0
        cand = cv2.goodFeaturesToTrack(bh, self.MAX_POINTS, 0.15, max(4, fw / 50), mask=mask, blockSize=5)
        if cand is None:
            return 0
        pts = cand.reshape(-1, 2)
        pts = pts[bh[pts[:, 1].astype(int), pts[:, 0].astype(int)] > max(22, 0.45 * float(bh.max()))]
        if len(pts) == 0:
            return 0
        V = P[:468, :2]
        idx, w, r0 = [], [], []
        for p in pts:
            d = np.linalg.norm(V - p, axis=1)
            nn = np.argpartition(d, 3)[:3]
            wk = 1.0 / (d[nn] + 1.0)
            wk /= wk.sum()
            idx.append(nn)
            w.append(wk)
            r0.append(p - (wk[:, None] * V[nn]).sum(0))
        self.pts = pts.astype(np.float32).reshape(-1, 1, 2)
        self.idx, self.w, self.r0 = np.array(idx), np.array(w), np.array(r0)
        self.prev, self.n0 = gray, len(pts)
        return len(pts)

    def update(self, gray, P):
        """LK flow with forward-backward check; the residual between tracked marks and the positions the mesh
        predicts is pushed back into the nearby landmarks."""
        if not self.active:
            return P
        p0 = self.pts
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **self.LK)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.LK)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 1.0)
        self.pts, self.idx, self.w, self.r0 = p1[good], self.idx[good], self.w[good], self.r0[good]
        self.prev = gray
        if not self.active:
            return P
        cur = self.pts.reshape(-1, 2)
        pred = (self.w[:, :, None] * P[self.idx, :2]).sum(1) + self.r0
        e = cur - pred
        self.jitter = float(np.linalg.norm(e, axis=1).mean())
        num, den = np.zeros((478, 2)), np.zeros(478)
        for k in range(3):
            np.add.at(num, self.idx[:, k], self.w[:, k, None] * e)
            np.add.at(den, self.idx[:, k], self.w[:, k])
        m = den > 1e-6
        out = P.copy()
        out[m, :2] += self.ALPHA * num[m] / den[m, None]
        return out
