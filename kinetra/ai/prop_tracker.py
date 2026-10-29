"""Color-Tape Prop & Weapon Tracker for Markerless Mocap.

Tracks rigid handheld objects (swords, daggers, pistols, shields, bo-staffs) in real time
using a small piece of colored tape (e.g. Neon Green, Orange, or Cyan) attached to the tip/barrel.

Uses the known wrist joint as the anchor socket and resolves the 6-DoF prop transform
from the wrist-to-tip vector with kinematic fallback to hand orientation when occluded.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..core.mathx import OneEuroFilter, unit, q_from_vectors, q_norm, IDENTITY_Q


@dataclass
class PropResult:
    detected: bool
    kind: str                                  # "sword", "knife", "pistol", "shield", "staff"
    side: str                                  # "R" or "L"
    tip_2d: Optional[Tuple[float, float]]     # Normalized (0..1) screen coords for crosshair
    tip_3d: np.ndarray                         # 3D tip position in scene space
    base_3d: np.ndarray                        # 3D base/wrist anchor position in scene space
    quat: np.ndarray                           # 4D quaternion orientation (w, x, y, z)
    length: float                              # Measured/configured prop length in metres
    confidence: float                          # Detection confidence [0..1]
    color_name: str                            # "green", "orange", "blue"


class PropTracker:
    """Real-time color tape detector and 6-DoF prop solver."""

    COLOR_RANGES = {
        "green": {
            "lower": np.array([35, 80, 70], dtype=np.uint8),
            "upper": np.array([88, 255, 255], dtype=np.uint8),
            "name": "Neon Yeşil",
            "hex": "#00FF66",
        },
        "orange": {
            "lower": np.array([5, 110, 90], dtype=np.uint8),
            "upper": np.array([24, 255, 255], dtype=np.uint8),
            "name": "Neon Turuncu",
            "hex": "#FF7700",
        },
        "blue": {
            "lower": np.array([88, 110, 80], dtype=np.uint8),
            "upper": np.array([130, 255, 255], dtype=np.uint8),
            "name": "Neon Mavi",
            "hex": "#00E5FF",
        },
    }

    PROP_CONFIGS = {
        "sword": {"length": 0.88, "radius": 0.035, "label": "Kılıç / Katana"},
        "knife": {"length": 0.28, "radius": 0.025, "label": "Bıçak / Hançer"},
        "pistol": {"length": 0.22, "radius": 0.030, "label": "Tabanca / Silah"},
        "shield": {"length": 0.38, "radius": 0.24, "label": "Savaş Kalkanı"},
        "staff": {"length": 1.65, "radius": 0.022, "label": "Dövüş Asası (Bo-Staff)"},
    }

    def __init__(self, color="green", kind="sword", side="R"):
        self.color_key = color
        self.kind = kind
        self.side = side
        self.enabled = True
        self._filter_tip = OneEuroFilter(f_min=1.8, beta=6.0)
        self._filter_q = OneEuroFilter(f_min=1.8, beta=6.0)
        self.last_result: Optional[PropResult] = None

    def set_config(self, kind: Optional[str] = None, color: Optional[str] = None, side: Optional[str] = None):
        if kind and kind in self.PROP_CONFIGS:
            self.kind = kind
        if color and color in self.COLOR_RANGES:
            self.color_key = color
        if side and side in ("R", "L"):
            self.side = side

    def reset(self):
        self._filter_tip.reset()
        self._filter_q.reset()
        self.last_result = None

    def process(self, rgb_frame: np.ndarray, wrist_img: Optional[np.ndarray],
                wrist_3d: Optional[np.ndarray], hand_quat: Optional[np.ndarray] = None,
                t_now: float = 0.0) -> Optional[PropResult]:
        """Detects the prop tip from color tape or falls back to hand pose."""
        if not self.enabled or wrist_3d is None:
            return None

        cfg = self.PROP_CONFIGS.get(self.kind, self.PROP_CONFIGS["sword"])
        prop_len = cfg["length"]
        h_frame, w_frame = rgb_frame.shape[:2]

        tip_2d = None
        detected = False
        conf = 0.0

        crange = self.COLOR_RANGES.get(self.color_key, self.COLOR_RANGES["green"])
        lower = crange["lower"]
        upper = crange["upper"]

        # 1. Search for color tape marker
        if wrist_img is not None and wrist_img[3] > 0.3:
            # ROI centered at wrist, radius proportional to prop length in screen space
            wx, wy = int(wrist_img[0] * w_frame), int(wrist_img[1] * h_frame)
            roi_r = int(max(w_frame, h_frame) * 0.45)
            x0 = max(0, wx - roi_r)
            x1 = min(w_frame, wx + roi_r)
            y0 = max(0, wy - roi_r)
            y1 = min(h_frame, wy + roi_r)

            crop = rgb_frame[y0:y1, x0:x1]
            if crop.size > 100:
                hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, lower, upper)

                # Morphology: remove speckles and join contiguous patch
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask, kernel, iterations=2)

                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    best_cnt = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(best_cnt)
                    if area >= 12:
                        M = cv2.moments(best_cnt)
                        if M["m00"] > 1e-4:
                            cx = int(M["m10"] / M["m00"]) + x0
                            cy = int(M["m01"] / M["m00"]) + y0
                            tip_2d = (float(cx / w_frame), float(cy / h_frame))
                            detected = True
                            conf = min(1.0, float(area / 160.0))

        # 2. Compute 3D direction vector
        base_3d = np.asarray(wrist_3d[:3], dtype=np.float64)

        if detected and tip_2d is not None and wrist_img is not None:
            # 2D screen direction
            dx = tip_2d[0] - wrist_img[0]
            dy = -(tip_2d[1] - wrist_img[1])  # Y is up in 3D scene space
            norm_2d = math.hypot(dx, dy)
            if norm_2d > 1e-3:
                # Estimate Z inclination: if projected length is shorter than full prop, prop has depth
                max_proj = 0.35 * (prop_len / 0.88)
                ratio = min(1.0, norm_2d / max(max_proj, 1e-3))
                dz = math.sqrt(max(0.0, 1.0 - ratio * ratio))
                # If pointing forwards
                dir_vec = unit(np.array([dx / norm_2d * ratio, dy / norm_2d * ratio, dz]))
            else:
                dir_vec = np.array([0.0, 1.0, 0.0])
        elif hand_quat is not None:
            # Kinematic fallback: weapon points along hand/palm forward axis
            from ..core.mathx import q_rotate
            dir_vec = unit(q_rotate(hand_quat, np.array([0.0, 1.0, 0.0])))
        else:
            dir_vec = np.array([0.0, 1.0, 0.0] if self.side == "R" else [0.0, 1.0, 0.0])

        # 3. Calculate Orientation Quaternion
        # Base prop model points along +Y
        raw_quat = q_from_vectors(np.array([0.0, 1.0, 0.0]), dir_vec)
        smooth_q = q_norm(self._filter_q(raw_quat, t_now))

        # Calculate Tip 3D Position
        raw_tip = base_3d + dir_vec * prop_len
        smooth_tip = self._filter_tip(raw_tip, t_now)

        res = PropResult(
            detected=detected,
            kind=self.kind,
            side=self.side,
            tip_2d=tip_2d,
            tip_3d=smooth_tip,
            base_3d=base_3d,
            quat=smooth_q,
            length=prop_len,
            confidence=conf,
            color_name=crange["name"],
        )
        self.last_result = res
        return res
