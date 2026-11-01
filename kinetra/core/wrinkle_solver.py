"""Dynamic Facial Wrinkle & Skin Tension Solver.
Computes real-time micro-wrinkle intensities (forehead, glabellar frown, crow's feet,
nasolabial folds, and chin pebbling) from ARKit 52 blendshapes and landmark metrics."""
from typing import Dict, Optional, Tuple
import numpy as np

from .constants import BLENDSHAPE_NAMES


class WrinkleSolver:
    """Calculates normalized tension and wrinkle activation weights [0.0 - 1.0]

    compatible with MetaHuman dynamic wrinkle maps and Blender displacement rigs.
    """

    WRINKLE_CHANNELS = (
        "forehead_horizontal",
        "glabellar_frown",
        "crows_feet_left",
        "crows_feet_right",
        "nasolabial_left",
        "nasolabial_right",
        "chin_pebble",
        "marionette_left",
        "marionette_right",
    )

    def __init__(self, smoothing: float = 0.25):
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._bs_map = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}
        self._prev_weights: Dict[str, float] = {k: 0.0 for k in self.WRINKLE_CHANNELS}
        self.neutral_brow_dist: Optional[float] = None
        self.neutral_naso_dist: Optional[float] = None

    def _get_bs(self, bs_vec: np.ndarray, name: str) -> float:
        idx = self._bs_map.get(name)
        if idx is not None and idx < len(bs_vec):
            return float(np.clip(bs_vec[idx], 0.0, 1.0))
        return 0.0

    def solve(
        self,
        blendshapes: np.ndarray,
        landmarks_3d: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculates dynamic wrinkle weights.

        blendshapes: 52-dim array of Apple ARKit blendshapes.
        landmarks_3d: optional (468, 3) facial landmark array.
        Returns dict of wrinkle channel names to floats in [0.0, 1.0].
        """
        if blendshapes is None or len(blendshapes) < 52:
            return {k: 0.0 for k in self.WRINKLE_CHANNELS}

        # 1. Blendshape-driven tension kinematics
        brow_inner_up = self._get_bs(blendshapes, "browInnerUp")
        brow_outer_up_l = self._get_bs(blendshapes, "browOuterUpLeft")
        brow_outer_up_r = self._get_bs(blendshapes, "browOuterUpRight")
        brow_down_l = self._get_bs(blendshapes, "browDownLeft")
        brow_down_r = self._get_bs(blendshapes, "browDownRight")

        eye_squint_l = self._get_bs(blendshapes, "eyeSquintLeft")
        eye_squint_r = self._get_bs(blendshapes, "eyeSquintRight")
        cheek_squint_l = self._get_bs(blendshapes, "cheekSquintLeft")
        cheek_squint_r = self._get_bs(blendshapes, "cheekSquintRight")

        nose_sneer_l = self._get_bs(blendshapes, "noseSneerLeft")
        nose_sneer_r = self._get_bs(blendshapes, "noseSneerRight")
        mouth_smile_l = self._get_bs(blendshapes, "mouthSmileLeft")
        mouth_smile_r = self._get_bs(blendshapes, "mouthSmileRight")
        mouth_frown_l = self._get_bs(blendshapes, "mouthFrownLeft")
        mouth_frown_r = self._get_bs(blendshapes, "mouthFrownRight")
        mouth_pucker = self._get_bs(blendshapes, "mouthPucker")
        mouth_shrug_u = self._get_bs(blendshapes, "mouthShrugUpper")
        mouth_stretch_l = self._get_bs(blendshapes, "mouthStretchLeft")
        mouth_stretch_r = self._get_bs(blendshapes, "mouthStretchRight")

        # Forehead: Frontalis muscle tension when elevating brows
        forehead = brow_inner_up * 0.70 + max(brow_outer_up_l, brow_outer_up_r) * 0.30

        # Glabellar frown: Corrugator supercilii contraction (furrowed brow)
        frown = ((brow_down_l + brow_down_r) * 0.50) * 0.85 + ((nose_sneer_l + nose_sneer_r) * 0.50) * 0.15

        # Crow's feet (orbicularis oculi compression on lateral eye orbits)
        crows_l = eye_squint_l * 0.65 + cheek_squint_l * 0.35
        crows_r = eye_squint_r * 0.65 + cheek_squint_r * 0.35

        # Nasolabial folds (levator labii & zygomaticus contraction)
        naso_l = mouth_smile_l * 0.60 + nose_sneer_l * 0.30 + cheek_squint_l * 0.10
        naso_r = mouth_smile_r * 0.60 + nose_sneer_r * 0.30 + cheek_squint_r * 0.10

        # Chin pebble (mentalis muscle contraction)
        chin = mouth_pucker * 0.55 + mouth_shrug_u * 0.45

        # Marionette lines
        marionette_l = mouth_frown_l * 0.70 + mouth_stretch_l * 0.30
        marionette_r = mouth_frown_r * 0.70 + mouth_stretch_r * 0.30

        # 2. Metric boost from 3D Landmark compression if available
        if landmarks_3d is not None and len(landmarks_3d) >= 468:
            # Medial eyebrow landmarks: Left 107, Right 336
            brow_dist = float(np.linalg.norm(landmarks_3d[107] - landmarks_3d[336]))
            if self.neutral_brow_dist is None or self.neutral_brow_dist <= 1e-4:
                self.neutral_brow_dist = brow_dist
            else:
                ratio = brow_dist / self.neutral_brow_dist
                if ratio < 0.90:
                    compression_factor = float(np.clip((0.90 - ratio) / 0.20, 0.0, 1.0))
                    frown = max(frown, compression_factor)

        raw_weights = {
            "forehead_horizontal": float(np.clip(forehead, 0.0, 1.0)),
            "glabellar_frown": float(np.clip(frown, 0.0, 1.0)),
            "crows_feet_left": float(np.clip(crows_l, 0.0, 1.0)),
            "crows_feet_right": float(np.clip(crows_r, 0.0, 1.0)),
            "nasolabial_left": float(np.clip(naso_l, 0.0, 1.0)),
            "nasolabial_right": float(np.clip(naso_r, 0.0, 1.0)),
            "chin_pebble": float(np.clip(chin, 0.0, 1.0)),
            "marionette_left": float(np.clip(marionette_l, 0.0, 1.0)),
            "marionette_right": float(np.clip(marionette_r, 0.0, 1.0)),
        }

        # Temporal smoothing
        out = {}
        for k, v in raw_weights.items():
            prev = self._prev_weights[k]
            filtered = prev * self.smoothing + v * (1.0 - self.smoothing)
            out[k] = round(float(filtered), 4)
            self._prev_weights[k] = filtered

        return out

    def get_metahuman_curves(self, wrinkle_weights: Dict[str, float]) -> Dict[str, float]:
        """Maps computed wrinkle weights to Unreal Engine MetaHuman dynamic wrinkle curve names."""
        return {
            "CTRL_Wrinkle_Forehead": wrinkle_weights.get("forehead_horizontal", 0.0),
            "CTRL_Wrinkle_Glabella": wrinkle_weights.get("glabellar_frown", 0.0),
            "CTRL_Wrinkle_CrowsFeet_L": wrinkle_weights.get("crows_feet_left", 0.0),
            "CTRL_Wrinkle_CrowsFeet_R": wrinkle_weights.get("crows_feet_right", 0.0),
            "CTRL_Wrinkle_Nasolabial_L": wrinkle_weights.get("nasolabial_left", 0.0),
            "CTRL_Wrinkle_Nasolabial_R": wrinkle_weights.get("nasolabial_right", 0.0),
            "CTRL_Wrinkle_Chin": wrinkle_weights.get("chin_pebble", 0.0),
        }
