"""Art-mannequin and Ball Joint Doll (BJD) characters.

Provides:
  * BallJointDollModel: Premium stylized Ball Joint Doll basemesh (ChamberSu1996 aesthetic),
    featuring porcelain ivory resin panels, sleek titanium ball-and-socket articulations,
    sculpted anime/doll head profile, and articulated hands and feet.
  * MannequinModel: Smooth gray art-mannequin figure.
"""
import math
import numpy as np

from .bodymesh import BodyModel, stylized_head_mesh
from .fullbody import SIDES
from .hand import FINGERS
from .mathx import unit

BODY = (0.66, 0.66, 0.68)
BODY_DARK = (0.58, 0.58, 0.60)
JOINT = (0.36, 0.36, 0.39)

# BJD Porcelain Doll Palette (ChamberSu1996 Aesthetic)
BJD_BODY = (0.91, 0.90, 0.88)
BJD_BODY_ACCENT = (0.84, 0.83, 0.81)
BJD_JOINT = (0.24, 0.24, 0.27)
BJD_SOCKET = (0.35, 0.35, 0.38)

# torso cross-sections, metres for a 0.50 m spine and 0.19 m half shoulder width:
# (height above the hips joint, half width, depth in front, depth behind)
TORSO_TABLE = {
    "female": [(-0.10, 0.100, 0.075, 0.085), (-0.06, 0.138, 0.090, 0.108), (0.00, 0.156, 0.098, 0.120), (0.06, 0.150, 0.094, 0.114),
               (0.12, 0.126, 0.082, 0.092), (0.17, 0.108, 0.074, 0.078), (0.22, 0.106, 0.073, 0.075), (0.27, 0.116, 0.082, 0.077),
               (0.33, 0.134, 0.100, 0.081), (0.385, 0.148, 0.112, 0.084), (0.43, 0.156, 0.100, 0.086), (0.475, 0.164, 0.086, 0.080),
               (0.515, 0.110, 0.064, 0.062), (0.54, 0.055, 0.048, 0.048)],
    "male": [(-0.10, 0.100, 0.075, 0.085), (-0.06, 0.128, 0.086, 0.100), (0.00, 0.142, 0.092, 0.108), (0.06, 0.140, 0.090, 0.104),
             (0.12, 0.130, 0.084, 0.092), (0.17, 0.124, 0.080, 0.084), (0.22, 0.126, 0.081, 0.083), (0.27, 0.136, 0.088, 0.085),
             (0.33, 0.152, 0.100, 0.088), (0.385, 0.168, 0.108, 0.090), (0.43, 0.178, 0.102, 0.090), (0.475, 0.188, 0.088, 0.082),
             (0.515, 0.126, 0.066, 0.064), (0.54, 0.058, 0.050, 0.050)],
    "bjd": [(-0.10, 0.092, 0.070, 0.080), (-0.06, 0.128, 0.084, 0.098), (0.00, 0.146, 0.090, 0.110), (0.06, 0.138, 0.085, 0.102),
            (0.12, 0.114, 0.074, 0.082), (0.17, 0.096, 0.066, 0.070), (0.22, 0.095, 0.066, 0.068), (0.27, 0.108, 0.076, 0.072),
            (0.33, 0.126, 0.092, 0.076), (0.385, 0.142, 0.106, 0.080), (0.43, 0.150, 0.094, 0.082), (0.475, 0.158, 0.080, 0.075),
            (0.515, 0.102, 0.058, 0.058), (0.54, 0.050, 0.044, 0.044)]
}


def _smooth(a, k=5):
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def loft(sections, nseg=24):
    """sections: list of (y, half_width_x, depth_front, depth_back, z_shift). Elliptical rings joined into a closed
    surface with fan caps. Axis is +Y. Returns verts (V,3), tris (F,3)."""
    ang = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    rings, tris = [], []
    for y, rx, zf, zb, cz in sections:
        a, d = (zf + zb) / 2, (zf - zb) / 2
        rings.append(np.stack([rx * c, np.full(nseg, y), cz + s * (a + d * s)], axis=1))
    V = np.concatenate(rings)
    n = len(sections)
    for i in range(n - 1):
        for k in range(nseg):
            a_, b_ = i * nseg + k, i * nseg + (k + 1) % nseg
            c_, d_ = (i + 1) * nseg + k, (i + 1) * nseg + (k + 1) % nseg
            tris += [(a_, c_, b_), (b_, c_, d_)]
    lo, hi = len(V), len(V) + 1
    V = np.vstack([V, [0.0, sections[0][0], sections[0][4]], [0.0, sections[-1][0], sections[-1][4]]])
    for k in range(nseg):
        tris.append((lo, k, (k + 1) % nseg))
        tris.append((hi, (n - 1) * nseg + (k + 1) % nseg, (n - 1) * nseg + k))
    return V, np.array(tris, dtype=np.int64)


def round_ends(radius_fn, y0, y1, n=12, cap=0.35):
    """Section list from y0 to y1 whose radius follows radius_fn(t) with rounded ends (cap = rounding depth in radii)."""
    out = []
    for i in range(n + 1):
        t = i / n
        r = radius_fn(t)
        e = min(t, 1 - t) / 0.22
        f = math.sqrt(max(0.0, 1 - max(0.0, 1 - e) ** 2)) if e < 1 else 1.0
        out.append((y0 + (y1 - y0) * t, r * (0.6 + 0.4 * f)))          # blunt ends: the ball joints hide the seams
    return out


class MannequinModel(BodyModel):
    def __init__(self, skel, style="female"):
        self.style = style if style in TORSO_TABLE else "female"
        super().__init__(skel)

    # -- helpers ----------------------------------------------------------------------------------
    def _loft_part(self, joint, sections, color, origin=(0, 0, 0), align=None, nseg=24):
        v, t = loft(sections, nseg)
        self._add(joint, v, t, color, origin, align)

    def _ball(self, joint, center, r, color=JOINT):
        self._ell(joint, center, (r, r, r), color, 16, 10)

    def _limb_loft(self, joint, vec, radius_fn, color=BODY, front=1.0, back=1.0, cz=0.0, gap=0.0):
        L = float(np.linalg.norm(vec))
        secs = []
        for y, r in round_ends(radius_fn, gap * L, (1 - gap) * L, 14):
            secs.append((y, r, r * front, r * back, cz * r))
        self._loft_part(joint, secs, color, (0, 0, 0), unit(vec))

    def _torso(self, joint, y_lo, y_hi, y0, sx, sy, color=BODY):
        """Torso slice between table heights y_lo..y_hi (metres at default size) bound to `joint` whose origin sits at y0."""
        tab = np.array(TORSO_TABLE[self.style])
        ys = np.linspace(y_lo, y_hi, 22)
        cols = [_smooth(np.interp(ys, tab[:, 0], tab[:, k]), 5) for k in range(1, 4)]
        secs = [((y * sy) - y0, cols[0][i] * sx, cols[1][i] * sx, cols[2][i] * sx, 0.0) for i, y in enumerate(ys)]
        self._loft_part(joint, secs, color, nseg=28)

    def _stylized_head(self, color=BODY, H=0.225, n_rings=32, n_seg=36):
        """Sculpted mannequin head with clean planar faceplate and smooth cranium."""
        V, T = stylized_head_mesh(H=H, n_rings=n_rings, n_seg=n_seg)
        self._add("head", V, T, color)

    # -- the model --------------------------------------------------------------------------------
    def _build(self):
        sk, o = self.skel, self.skel.offsets
        sp = 2 * o["chest"][1]
        sw, hw = o["upper_arm_L"][0], o["thigh_L"][0]
        sy = sp / 0.5
        sx = ((sw / 0.19) + (hw / 0.10)) / 2
        y_chest, y_neck = o["chest"][1], sp
        # torso: pelvis | waist ball | abdomen | chest | neck base ball
        self._torso("Hips", -0.10, 0.112, 0.0, sx, sy)
        self._ball("spine", (0, 0.124 * sy, 0), 0.074 * sx)
        self._torso("spine", 0.136, 0.250, 0.0, sx, sy)
        self._torso("chest", 0.244, 0.540, y_chest, sx, sy)
        self._ball("neck", (0, 0.035, 0), 0.044)
        # neck and head (sculpted stylized mannequin head)
        self._limb_loft("neck", o["head"], lambda t: 0.037 - 0.003 * t, BODY_DARK, gap=0.12)
        self._stylized_head(color=BODY)
        # arms and hands
        for s in SIDES:
            self._ball("upper_arm_" + s, (0, 0, 0), 0.052)
            self._limb_loft("upper_arm_" + s, o["forearm_" + s], lambda t: 0.044 - 0.011 * t + 0.003 * math.sin(math.pi * t), gap=0.06)
            self._ball("forearm_" + s, (0, 0, 0), 0.033)
            self._limb_loft("forearm_" + s, o["hand_" + s], lambda t: 0.035 - 0.013 * t + 0.004 * math.sin(math.pi * min(1, t * 1.6)), gap=0.07)
            self._ball("hand_" + s, (0, 0, 0), 0.025)
            mcp = np.array([o[f"{f}_1_{s}"] for f in FINGERS[1:]])
            c = mcp.mean(axis=0) * 0.52
            self._ell("hand_" + s, c, (0.016, abs(mcp[:, 1]).mean() * 0.62, (mcp[:, 2].max() - mcp[:, 2].min()) / 2 + 0.012), BODY, 16, 10)
            for f, r in zip(FINGERS, (0.0112, 0.0100, 0.0104, 0.0096, 0.0084)):
                for i in (1, 2, 3):
                    j = f"{f}_{i}_{s}"
                    vec = np.array(sk.offsets[f"{f}_{i + 1}_{s}"]) if i < 3 else np.array(sk.ends[j])
                    rr = r * (1.0, 0.92, 0.82)[i - 1]
                    self._limb(j, vec, rr, rr * 0.9, BODY, 10, 3)
                    self._ell(j, (0, 0, 0), (rr * 1.05,) * 3, JOINT, 8, 5)
                self._ell(f"{f}_3_{s}", sk.ends[f"{f}_3_{s}"], (r * 0.78,) * 3, BODY, 8, 5)
        # legs and feet
        for s in SIDES:
            self._ball("thigh_" + s, (0, 0, 0), 0.060)
            self._limb_loft("thigh_" + s, o["shin_" + s], lambda t: 0.080 - 0.030 * t ** 0.8 + 0.003 * math.sin(math.pi * t), front=1.06, back=1.0, gap=0.045)
            self._ball("shin_" + s, (0, 0, 0), 0.047)
            self._limb_loft("shin_" + s, o["foot_" + s], lambda t: 0.050 - 0.020 * t + 0.010 * math.sin(math.pi * min(1, t / 0.7)) * (1 - t),
                            front=0.92, back=1.12, gap=0.06)
            self._ball("foot_" + s, (0, 0, 0), 0.032)
            tl = o["toe_" + s][2]
            self._ell("foot_" + s, (0, -0.040, 0.40 * tl), (0.040, 0.040, 0.60 * tl + 0.02), BODY, 16, 10)          # foot
            self._ell("foot_" + s, (0, -0.038, -0.022), (0.038, 0.044, 0.045), BODY, 14, 9)                         # heel
            self._ell("toe_" + s, (0, -0.046, 0.35 * tl), (0.037, 0.021, 0.40 * tl), BODY_DARK, 14, 9)              # toes


class BallJointDollModel(MannequinModel):
    """Ball Joint Doll (BJD) Basemesh - inspired by ChamberSu1996 design.

    Features ivory resin shell parts, prominent spherical ball-joints at every articulation,
    sculpted anime/doll head profile with chin and jawline, and articulated jointed limbs.
    """
    def __init__(self, skel):
        super().__init__(skel, style="bjd")

    def _bjd_head(self):
        self._stylized_head(color=BJD_BODY)

    def _build(self):
        sk, o = self.skel, self.skel.offsets
        sp = 2 * o["chest"][1]
        sw, hw = o["upper_arm_L"][0], o["thigh_L"][0]
        sy = sp / 0.5
        sx = ((sw / 0.19) + (hw / 0.10)) / 2
        y_chest, y_neck = o["chest"][1], sp

        # 1. Torso: Pelvic shell, lower lumbar ball, midriff, chest shell, neck base ball
        self._torso("Hips", -0.09, 0.105, 0.0, sx, sy, color=BJD_BODY)
        self._ball("Hips", (0, 0.080 * sy, 0), 0.066 * sx, color=BJD_JOINT)
        self._ball("spine", (0, 0.120 * sy, 0), 0.070 * sx, color=BJD_JOINT)
        self._torso("spine", 0.130, 0.245, 0.0, sx, sy, color=BJD_BODY)
        self._torso("chest", 0.240, 0.535, y_chest, sx, sy, color=BJD_BODY)
        self._ball("neck", (0, 0.032, 0), 0.042, color=BJD_JOINT)

        # 2. Neck and Sculpted BJD Doll Head Profile (Anime/BJD Basemesh Chin & Crown)
        self._limb_loft("neck", o["head"], lambda t: 0.034 - 0.004 * t, BJD_BODY_ACCENT, gap=0.10)
        self._ball("head", (0, 0.025, 0), 0.038, color=BJD_JOINT)
        self._bjd_head()

        # 3. Arms and Hands with BJD Ball-and-Socket Joint Cups
        for s in SIDES:
            # Shoulder ball & socket
            self._ball("upper_arm_" + s, (0, 0, 0), 0.050, color=BJD_JOINT)
            self._limb_loft("upper_arm_" + s, o["forearm_" + s],
                            lambda t: 0.042 - 0.010 * t + 0.003 * math.sin(math.pi * t),
                            color=BJD_BODY, gap=0.065)

            # Elbow ball joint
            self._ball("forearm_" + s, (0, 0, 0), 0.034, color=BJD_JOINT)
            self._limb_loft("forearm_" + s, o["hand_" + s],
                            lambda t: 0.033 - 0.012 * t + 0.003 * math.sin(math.pi * min(1, t * 1.5)),
                            color=BJD_BODY, gap=0.07)

            # Wrist ball joint
            self._ball("hand_" + s, (0, 0, 0), 0.024, color=BJD_JOINT)
            mcp = np.array([o[f"{f}_1_{s}"] for f in FINGERS[1:]])
            c = mcp.mean(axis=0) * 0.52
            self._ell("hand_" + s, c, (0.015, abs(mcp[:, 1]).mean() * 0.60, (mcp[:, 2].max() - mcp[:, 2].min()) / 2 + 0.011),
                      BJD_BODY, 16, 10)

            # Articulated doll fingers with knuckle sphere joints
            for f, r in zip(FINGERS, (0.0108, 0.0098, 0.0102, 0.0094, 0.0082)):
                for i in (1, 2, 3):
                    j = f"{f}_{i}_{s}"
                    vec = np.array(sk.offsets[f"{f}_{i + 1}_{s}"]) if i < 3 else np.array(sk.ends[j])
                    rr = r * (1.0, 0.90, 0.80)[i - 1]
                    self._limb(j, vec, rr, rr * 0.88, BJD_BODY, 10, 3)
                    self._ell(j, (0, 0, 0), (rr * 1.08,) * 3, BJD_JOINT, 8, 5)
                self._ell(f"{f}_3_{s}", sk.ends[f"{f}_3_{s}"], (r * 0.75,) * 3, BJD_BODY_ACCENT, 8, 5)

        # 4. Legs and Feet with BJD Ball-and-Socket Joint Cups
        for s in SIDES:
            # Hip ball joint
            self._ball("thigh_" + s, (0, 0, 0), 0.058, color=BJD_JOINT)
            self._limb_loft("thigh_" + s, o["shin_" + s],
                            lambda t: 0.076 - 0.028 * (t ** 0.8) + 0.003 * math.sin(math.pi * t),
                            color=BJD_BODY, front=1.05, back=1.0, gap=0.05)

            # Knee ball joint
            self._ball("shin_" + s, (0, 0, 0), 0.046, color=BJD_JOINT)
            self._limb_loft("shin_" + s, o["foot_" + s],
                            lambda t: 0.048 - 0.019 * t + 0.009 * math.sin(math.pi * min(1, t / 0.7)) * (1 - t),
                            color=BJD_BODY, front=0.92, back=1.10, gap=0.065)

            # Ankle ball joint and sculpted doll foot
            self._ball("foot_" + s, (0, 0, 0), 0.031, color=BJD_JOINT)
            tl = o["toe_" + s][2]
            self._ell("foot_" + s, (0, -0.038, 0.40 * tl), (0.038, 0.038, 0.58 * tl + 0.02), BJD_BODY, 16, 10)
            self._ell("foot_" + s, (0, -0.036, -0.020), (0.036, 0.042, 0.042), BJD_BODY, 14, 9)
            self._ell("toe_" + s, (0, -0.044, 0.35 * tl), (0.035, 0.020, 0.38 * tl), BJD_BODY_ACCENT, 14, 9)


def make_model(skel, style="bjd"):
    """Creates a 3D procedural character model bound to `skel`.
    Default and unified model is the Ball Joint Doll basemesh.
    """
    if style == "detailed":
        return BodyModel(skel)
    return BallJointDollModel(skel)
