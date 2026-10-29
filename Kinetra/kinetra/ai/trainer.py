"""Dataset building, training and model loading for the Kinetra AI add-on.

Data sources: procedural anatomically-limited motion (always), BVH motion-capture files (any library, e.g. CMU),
and the takes recorded with Kinetra itself. More and better data -> better predictions."""
import json
from pathlib import Path

import numpy as np

from .body_prior import BodyPrior, MP_INDEX
from .hand_prior import HandPosePrior, hand_features, NDIM
from . import synth

USER_DIR = Path.home() / ".kinetra" / "ai"
PRETRAINED = Path(__file__).resolve().parent / "pretrained"


def takes_body_sequences(project_dir):
    """Raw joint tracks of recorded takes -> list of (T,12,3) sequences."""
    out = []
    for f in sorted(Path(project_dir).glob("takes/take_*/raw.npz")):
        z = np.load(f, allow_pickle=False)
        raw = z["raw"]
        ok = ~np.isnan(raw).any(axis=(1, 2))
        if ok.sum() < 30:
            continue
        seq = raw[ok][:, list(MP_INDEX)]
        if not bool(z["absolute"]):
            seq = seq + np.concatenate([z["root_raw"][ok][:, :1] * 0, np.zeros((ok.sum(), 1, 1))], axis=-1) * 0
        out.append(seq)
    return out


def takes_hand_features(project_dir):
    """Solved finger angles of recorded takes -> (N,20) arrays per hand."""
    from ..core.hand import HandSolver
    X = []
    for f in sorted(Path(project_dir).glob("takes/take_*/raw.npz")):
        z = np.load(f, allow_pickle=False)
        if "hands_raw" not in z.files:
            continue
        for hi, side in enumerate("LR"):
            seq = []
            for P in z["hands_raw"][:, hi]:
                if not np.isnan(P).any():
                    seq.append(hand_features(HandSolver().solve(P, learn=False)["quats"], side))
            if len(seq) > 10:
                X.append(np.array(seq))
    return X


def train_all(n_body_seq=250, n_windows=30000, epochs=25, n_hand=40000, K=48, bvh_dir=None, project_dir=None,
              out_dir=USER_DIR, progress=None, seed=0):
    """Train both models and save them. progress(fraction, text) is optional. Returns a metrics dict."""
    rng = np.random.default_rng(seed)
    say = progress or (lambda f, t: None)
    say(0.0, "hand gestures")
    X = np.array([synth.random_hand_pose(rng) for _ in range(n_hand)])
    seqs = synth.hand_sequences(rng)
    real_hands = takes_hand_features(project_dir) if project_dir else []
    if real_hands:
        X = np.concatenate([X] + real_hands)
        seqs = seqs + real_hands
    hand = HandPosePrior.train(X, seqs, K=K, seed=seed, progress=lambda f: say(0.25 * f, "hand prior"))
    say(0.25, "body motion")
    bodies = [synth.body_sequence(rng) for _ in range(n_body_seq)]
    if project_dir:
        bodies += takes_body_sequences(project_dir)
    if bvh_dir:
        from .bvh_import import load_bvh_folder
        bodies += load_bvh_folder(bvh_dir)
    body = BodyPrior.train(bodies, n_windows=n_windows, epochs=epochs, seed=seed,
                           progress=lambda f: say(0.4 + 0.6 * f, "body prior"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hand.save(out_dir / "hand_prior.npz")
    body.save(out_dir / "body_prior.npz")
    info = {"hand_samples": int(len(X)), "body_sequences": len(bodies), "windows": n_windows, "epochs": epochs,
            "bvh": str(bvh_dir) if bvh_dir else None, "project": str(project_dir) if project_dir else None}
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    say(1.0, "done")
    return info


def load_models(prefer_user=True):
    """(HandPosePrior or None, BodyPrior or None): the user's trained models first, then the shipped ones."""
    hand = body = None
    for d in ([USER_DIR] if prefer_user else []) + [PRETRAINED]:
        if hand is None and (d / "hand_prior.npz").exists():
            hand = HandPosePrior.load(d / "hand_prior.npz")
        if body is None and (d / "body_prior.npz").exists():
            body = BodyPrior.load(d / "body_prior.npz")
    return hand, body
