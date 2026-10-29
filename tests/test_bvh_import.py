import tempfile
from pathlib import Path
import numpy as np
from kinetra.ai.bvh_import import body_from_bvh
from kinetra.ai.body_prior import frame_basis
from kinetra.core.exporters import write_bvh
from kinetra.core.fullbody import build_skeleton
from kinetra.core.mathx import q_axis_angle


def test_v2_bvh_is_read_back_with_the_right_shape_and_scale():
    sk = build_skeleton()
    n = 60
    q = np.tile(np.array([1.0, 0, 0, 0]), (n, len(sk.order), 1))
    ti = sk.order.index("thigh_L")
    for i in range(n):
        q[i, ti] = q_axis_angle([1, 0, 0], 0.5 * np.sin(i / 6))
    root = np.tile([0.0, 0.95, 0.0], (n, 1))
    p = Path(tempfile.mkdtemp()) / "a.bvh"
    write_bvh(p, np.arange(n) / 30.0, root, q, {}, 30.0, sk)
    seq = body_from_bvh(p.read_text())
    assert seq is not None and seq.shape == (n, 12, 3)
    pel, B, sc = frame_basis(seq[10])
    assert abs(sc * 0.01 - 0.5) < 0.05 or abs(sc - 0.5) < 0.05        # torso length (file is in centimetres)
    knee_y = seq[:, 8, 1]
    assert knee_y.max() - knee_y.min() > 1e-3                         # the swinging thigh moved the left knee
