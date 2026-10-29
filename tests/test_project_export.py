import json, tempfile
from pathlib import Path
import numpy as np
from kinetra.core.body_solver import BodySolver
from kinetra.core.constants import *
from kinetra.core.exporters import *
from kinetra.core.project import *
from test_body_solver import synth_pose


def make_take(n=120, absolute=False):
    rec, s, J = TakeRecorder(), BodySolver(), synth_pose()
    rng = np.random.default_rng(0)
    rec.start()
    for i in range(n):
        t = i / 30.0
        raw = J + rng.normal(0, 0.004, J.shape)
        raw[:, 0] += 0.2 * np.sin(t)             # sway
        out = s.process(raw, t, root=(0.1 * t, 0.0), absolute=absolute)
        rec.add(t, raw, np.ones(33), (0.1 * t, 0.0), out["joints"][HIP_MID], out["quats"],
                np.random.default_rng(i).random(NBS), absolute, out["lengths"])
    return rec.stop()


def test_save_export_and_reprocess(tmp_path):
    proj = Project.create(tmp_path / "demo", "Demo")
    data = make_take()
    d = save_take(proj, data, mode="single")
    for f in ("raw.npz", "meta.json", "motion.bvh", "motion.usda", "face_curves.csv"):
        assert (d / f).exists(), f
    bvh = (d / "motion.bvh").read_text()
    assert bvh.startswith("HIERARCHY") and "Frames: " in bvh
    frames = int(bvh.split("Frames: ")[1].split()[0])
    rows = bvh.split("Frame Time:")[1].strip().split("\n")[1:]
    assert len(rows) == frames and len(rows[0].split()) == 6 + 12 * 3
    n = reprocess_take(d)
    assert n > 100 and (d / "motion.npz").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["processed"].startswith("offline")
    assert len(proj.takes()) == 1
    # a second take gets the next number
    assert save_take(proj, make_take(30), mode="single").name == "take_0002"


def test_zero_phase_has_less_lag_than_causal():
    t = np.arange(0, 4, 1 / 30.0)
    x = np.stack([np.sin(2 * t)] * 3, axis=1)
    from kinetra.core.project import _zero_phase
    from kinetra.core.mathx import OneEuroFilter
    f = OneEuroFilter(1.0, 0.0)
    causal = np.array([f(v, tt) for v, tt in zip(x, t)])
    zp = _zero_phase(t, x, 1.0, 0.0)
    assert np.abs(zp - x).mean() < np.abs(causal - x).mean()


def test_usd_and_osc():
    t = np.arange(10) / 30.0
    q = np.tile(np.array([1.0, 0, 0, 0]), (10, len(JOINT_ORDER), 1))
    p = Path(tempfile.mkdtemp()) / "a.usda"
    write_usda(p, t, np.zeros((10, 3)), q, DEFAULT_LENGTHS)
    txt = p.read_text()
    assert txt.startswith("#usda 1.0") and txt.count("rotations.timeSamples") == 1 and '"Hips/spine/head"' in txt
    m = osc_message("/kinetra/root", [1.0, 2.0, 3.0])
    assert len(m) % 4 == 0 and m.startswith(b"/kinetra/root\0")
    assert osc_bundle([m]).startswith(b"#bundle\0")


def test_resample_preserves_endpoints():
    t = np.array([0.0, 0.1, 0.25, 0.3])
    q = np.tile(np.array([1.0, 0, 0, 0]), (4, 2, 1))
    r = np.arange(12, dtype=float).reshape(4, 3)
    t2, r2, q2 = resample_motion(t, r, q, 30)
    assert abs(t2[0] - 0) < 1e-9 and np.allclose(r2[0], r[0]) and np.allclose(np.linalg.norm(q2, axis=-1), 1)
