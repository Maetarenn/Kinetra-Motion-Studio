"""Quaternion algebra (w, x, y, z), One-Euro filter and small geometry helpers. No Qt, no I/O."""
import math
import numpy as np

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
FWD = np.array([0.0, 0.0, 1.0])
BACK = -FWD


def unit(v, fallback=None):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array(fallback if fallback is not None else [0.0, 1.0, 0.0], dtype=np.float64)
    return v / n


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_norm(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else IDENTITY_Q.copy()


def q_axis_angle(axis, angle):
    a = unit(axis, [1.0, 0.0, 0.0])
    s = math.sin(angle * 0.5)
    return np.array([math.cos(angle * 0.5), a[0] * s, a[1] * s, a[2] * s])


def q_from_vectors(u, v):
    """Shortest-arc rotation taking direction u onto direction v."""
    u, v = unit(u), unit(v)
    d = float(np.dot(u, v))
    if d < -0.999999:
        axis = np.cross([1.0, 0.0, 0.0], u)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], u)
        return np.array([0.0, *unit(axis)])
    return q_norm(np.array([1.0 + d, *np.cross(u, v)]))


def q_rotate(q, v):
    qv = q[1:]
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def q_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(M):
    t = float(np.trace(M))
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = math.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = [(M[2, 1] - M[1, 2]) / s, .25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
    elif M[1, 1] > M[2, 2]:
        s = math.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, .25 * s, (M[1, 2] + M[2, 1]) / s]
    else:
        s = math.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, .25 * s]
    return q_norm(np.array(q))


def frame_quat(x_hint, y_hint, x_fallback=(1.0, 0.0, 0.0)):
    """Orientation of a right-handed frame: y axis follows y_hint, x axis is x_hint made orthogonal to y."""
    y = unit(y_hint)
    x = np.asarray(x_hint, dtype=np.float64)
    x = unit(x - y * float(np.dot(x, y)), x_fallback)
    z = np.cross(x, y)
    return mat_to_quat(np.stack([x, y, z], axis=1))


def q_mean(quats):
    ref = quats[0]
    acc = np.zeros(4)
    for q in quats:
        acc += q if float(np.dot(q, ref)) >= 0 else -q
    return q_norm(acc)


def q_slerp(q0, q1, t):
    """Spherical linear interpolation between q0 and q1 with factor t in [0, 1]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q_norm(q0 + t * (q1 - q0))
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-6:
        return q0.copy()
    return (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / sin_theta


def q_pow(q, p):
    """Raise quaternion to power p (SLERP fractional rotation towards IDENTITY_Q)."""
    w = float(np.clip(q[0], -1.0, 1.0))
    ang = 2.0 * math.acos(abs(w))
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return IDENTITY_Q.copy()
    axis = q[1:] / s * (1.0 if w >= 0 else -1.0)
    return q_axis_angle(axis, ang * p)


def q_to_euler_zxy(q):
    """Quaternion -> BVH Z,X,Y Euler angles in degrees (R = Rz * Rx * Ry)."""
    R = q_to_mat(q)
    b = math.asin(float(np.clip(R[2, 1], -1.0, 1.0)))
    if abs(math.cos(b)) > 1e-6:
        a, c = math.atan2(-R[0, 1], R[1, 1]), math.atan2(-R[2, 0], R[2, 2])
    else:
        a, c = math.atan2(R[1, 0], R[0, 0]), 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def two_bone_ik(root, target, l1, l2, pole_point):
    """Analytic two-bone IK. Returns (mid_joint, end_joint); the mid joint bends toward pole_point."""
    root, target = np.asarray(root, float), np.asarray(target, float)
    to_t = target - root
    d = float(np.linalg.norm(to_t))
    u = unit(to_t, [0.0, -1.0, 0.0])
    d = min(max(d, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pv = np.asarray(pole_point, float) - root
    pv = pv - u * float(np.dot(pv, u))
    pv = unit(pv, [0.0, 0.0, 1.0])
    return root + u * a + pv * h, root + u * d


class OneEuroFilter:
    """One Euro adaptive low-pass filter, vectorised over any array shape.
    fc = f_min + beta*|v| ; tau = 1/(2*pi*fc) ; alpha = 1/(1 + tau/dt)."""

    def __init__(self, f_min=1.0, beta=5.0, d_cutoff=1.0, vector=True):
        self.f_min, self.beta, self.d_cutoff, self.vector = f_min, beta, d_cutoff, vector
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._x.shape != x.shape:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) if self.vector else np.abs(dx_hat)
        a = self._alpha(self.f_min + self.beta * speed, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()
