"""Multi-Actor Re-Identification (ReID) & Appearance Tracker.

Distinguishes and tracks multiple actors simultaneously using:
1. HSV Clothing Color Histograms (torso & pants patches)
2. Biometric Ratios (shoulder-to-torso, arm reach, relative proportions)
3. Kinematic Spatial Coherence (velocity & spatial distance)

Prevents actor identity swapping (cross-over / occlusion swaps) during multi-person mocap.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def color_name_from_rgb(r: int, g: int, b: int) -> str:
    """Categorizes dominant RGB color into clear Turkish names."""
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    # Grayscale / Low saturation
    if diff < 28:
        if max_c < 55:
            return "Siyah"
        elif max_c > 200:
            return "Beyaz"
        else:
            return "Gri"

    # Chromatic
    if r > 160 and g < 90 and b < 90:
        return "Kırmızı"
    elif b > 140 and r < 100:
        return "Mavi"
    elif g > 140 and r < 110 and b < 120:
        return "Yeşil"
    elif r > 180 and g > 160 and b < 90:
        return "Sarı"
    elif r > 180 and g > 90 and b < 50:
        return "Turuncu"
    elif r > 140 and b > 140 and g < 100:
        return "Mor"
    elif b > 120 and g > 120 and r < 80:
        return "Turkuaz"
    elif r > 100 and g > 60 and b < 50:
        return "Kahverengi"
    return "Renkli"


@dataclass
class ActorProfile:
    actor_id: int                              # 1 or 2
    label: str                                 # "Aktör 1" / "Aktör 2"
    color_hex: str                             # Hex code (e.g. "#1E88E5")
    color_name: str                            # e.g. "Koyu Mavi"
    theme_rgb: Tuple[float, float, float]      # (r, g, b) in [0, 1]
    torso_hist: Optional[np.ndarray] = None    # 16x16 normalized HSV histogram
    shoulder_ratio: float = 0.55
    arm_ratio: float = 0.65
    last_center: Tuple[float, float] = (0.5, 0.5)
    last_seen_t: float = 0.0
    tracked_frames: int = 0
    joints_3d: Optional[np.ndarray] = None
    quats: Optional[dict] = None
    quats_full: Optional[dict] = None


class MultiActorTracker:
    """Live Person Re-Identification & Appearance Tracker for multi-actor mocap."""

    THEMES = [
        (1, "Aktör 1", (0.0, 0.90, 1.0)),     # Cyan theme
        (2, "Aktör 2", (0.98, 0.55, 0.12)),    # Amber theme
    ]

    def __init__(self, max_actors: int = 2):
        self.max_actors = max_actors
        self.profiles: Dict[int, ActorProfile] = {}
        self._next_id = 1

    def reset(self):
        self.profiles.clear()
        self._next_id = 1

    @staticmethod
    def extract_appearance(rgb_frame: np.ndarray, landmarks_2d: np.ndarray) -> Optional[dict]:
        """Samples the torso clothing patch and extracts HSV histogram and biometric ratios."""
        h, w = rgb_frame.shape[:2]
        # Landmarks: 11, 12 shoulders; 23, 24 hips
        if landmarks_2d.shape[0] < 25:
            return None

        # Check visibility
        vis_sho = min(landmarks_2d[11, 3], landmarks_2d[12, 3]) if landmarks_2d.shape[1] > 3 else 1.0
        vis_hip = min(landmarks_2d[23, 3], landmarks_2d[24, 3]) if landmarks_2d.shape[1] > 3 else 1.0
        if vis_sho < 0.25 or vis_hip < 0.25:
            return None

        p11, p12 = landmarks_2d[11, :2], landmarks_2d[12, :2]
        p23, p24 = landmarks_2d[23, :2], landmarks_2d[24, :2]

        # Torso bounding box
        x_min = int(max(0, min(p11[0], p12[0], p23[0], p24[0]) * w))
        x_max = int(min(w, max(p11[0], p12[0], p23[0], p24[0]) * w))
        y_min = int(max(0, min(p11[1], p12[1]) * h))
        y_max = int(min(h, max(p23[1], p24[1]) * h))

        if (x_max - x_min) < 16 or (y_max - y_min) < 20:
            return None

        # Crop inner 80% to avoid background bleed
        margin_x = int((x_max - x_min) * 0.12)
        margin_y = int((y_max - y_min) * 0.12)
        crop = rgb_frame[y_min + margin_y:y_max - margin_y, x_min + margin_x:x_max - margin_x]
        if crop.size < 64:
            return None

        # Compute HSV histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Dominant color
        med_rgb = np.median(crop.reshape(-1, 3), axis=0).astype(int)
        hex_color = f"#{med_rgb[0]:02X}{med_rgb[1]:02X}{med_rgb[2]:02X}"
        color_name = color_name_from_rgb(int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2]))

        # Biometrics
        sho_w = float(np.linalg.norm(p11 - p12))
        torso_h = float(np.linalg.norm((p11 + p12) / 2.0 - (p23 + p24) / 2.0))
        sho_ratio = sho_w / (torso_h + 1e-5)

        center = (float((x_min + x_max) / (2.0 * w)), float((y_min + y_max) / (2.0 * h)))

        return {
            "hist": hist,
            "hex": hex_color,
            "color_name": color_name,
            "sho_ratio": sho_ratio,
            "center": center,
        }

    def match_and_update(self, rgb_frame: np.ndarray, detected_bodies: List[dict], t_now: float) -> List[ActorProfile]:
        """Matches candidate bodies to registered actors using Appearance + Biometrics + Spatial tracking."""
        candidates = []
        for body in detected_bodies:
            if body is None or "image" not in body:
                continue
            app = self.extract_appearance(rgb_frame, body["image"])
            candidates.append({"body": body, "app": app})

        if not candidates:
            return list(self.profiles.values())

        # If no profiles registered yet, initialize up to max_actors
        if not self.profiles:
            for i, cand in enumerate(candidates[:self.max_actors]):
                theme = self.THEMES[i] if i < len(self.THEMES) else (i + 1, f"Aktör {i + 1}", (0.5, 0.5, 0.5))
                app = cand["app"] or {}
                prof = ActorProfile(
                    actor_id=theme[0],
                    label=theme[1],
                    color_hex=app.get("hex", "#00E5FF" if theme[0] == 1 else "#FFB300"),
                    color_name=app.get("color_name", "Koyu Mavi" if theme[0] == 1 else "Turuncu"),
                    theme_rgb=theme[2],
                    torso_hist=app.get("hist"),
                    shoulder_ratio=app.get("sho_ratio", 0.55),
                    last_center=app.get("center", (0.5, 0.5)),
                    last_seen_t=t_now,
                    tracked_frames=1,
                    joints_3d=cand["body"].get("world"),
                )
                self.profiles[theme[0]] = prof
            return list(self.profiles.values())

        # Build cost matrix between registered profiles and candidate detections
        active_pids = list(self.profiles.keys())
        N, M = len(active_pids), len(candidates)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for i, pid in enumerate(active_pids):
            prof = self.profiles[pid]
            for j, cand in enumerate(candidates):
                app = cand["app"]
                if app is None:
                    # Spatial distance only
                    dx = cand["body"]["image"][11, 0] - prof.last_center[0]
                    dy = cand["body"]["image"][11, 1] - prof.last_center[1]
                    cost_matrix[i, j] = min(1.0, math.hypot(dx, dy) * 1.5)
                    continue

                # 1. Clothing HSV Histogram Bhattacharyya distance [0, 1]
                if prof.torso_hist is not None:
                    d_color = float(cv2.compareHist(prof.torso_hist, app["hist"], cv2.HISTCMP_BHATTACHARYYA))
                else:
                    d_color = 0.3

                # 2. Biometric shoulder ratio difference [0, 1]
                d_bio = min(1.0, abs(prof.shoulder_ratio - app["sho_ratio"]) / (prof.shoulder_ratio + 1e-4) * 2.0)

                # 3. Spatial displacement [0, 1]
                d_spatial = min(1.0, math.hypot(prof.last_center[0] - app["center"][0],
                                               prof.last_center[1] - app["center"][1]) * 1.8)

                # Weighted fusion: Clothing color is the primary discriminator against cross-over swap
                cost = 0.55 * d_color + 0.25 * d_bio + 0.20 * d_spatial
                cost_matrix[i, j] = cost

        # Solve assignment
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # Greedy matching for 2x2
            row_ind = list(range(min(N, M)))
            col_ind = list(range(min(N, M)))
            if N == 2 and M == 2:
                if cost_matrix[0, 1] + cost_matrix[1, 0] < cost_matrix[0, 0] + cost_matrix[1, 1]:
                    col_ind = [1, 0]

        assigned_cands = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.85: # Valid match threshold
                pid = active_pids[r]
                prof = self.profiles[pid]
                cand = candidates[c]
                app = cand["app"]
                if app is not None:
                    # Update profile with exponential moving average
                    if prof.torso_hist is not None:
                        prof.torso_hist = 0.90 * prof.torso_hist + 0.10 * app["hist"]
                    else:
                        prof.torso_hist = app["hist"]
                    prof.color_hex = app["hex"]
                    prof.color_name = app["color_name"]
                    prof.shoulder_ratio = 0.90 * prof.shoulder_ratio + 0.10 * app["sho_ratio"]
                    prof.last_center = app["center"]

                prof.last_seen_t = t_now
                prof.tracked_frames += 1
                prof.joints_3d = cand["body"].get("world")
                assigned_cands.add(c)

        # If a candidate was unassigned and we have space for Actor 2, register them
        if len(self.profiles) < self.max_actors:
            for j, cand in enumerate(candidates):
                if j not in assigned_cands:
                    new_id = 2 if 1 in self.profiles else 1
                    theme = self.THEMES[new_id - 1]
                    app = cand["app"] or {}
                    self.profiles[new_id] = ActorProfile(
                        actor_id=new_id,
                        label=theme[1],
                        color_hex=app.get("hex", "#FFB300"),
                        color_name=app.get("color_name", "Turuncu"),
                        theme_rgb=theme[2],
                        torso_hist=app.get("hist"),
                        shoulder_ratio=app.get("sho_ratio", 0.55),
                        last_center=app.get("center", (0.5, 0.5)),
                        last_seen_t=t_now,
                        tracked_frames=1,
                        joints_3d=cand["body"].get("world"),
                    )
                    break

        return [self.profiles[k] for k in sorted(self.profiles.keys())]
