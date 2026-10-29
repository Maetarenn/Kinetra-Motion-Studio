import numpy as np, cv2
from kinetra.core.multicam import *


def look_at(c, target, up=(0, 1, 0)):
    c, target, up = map(lambda v: np.asarray(v, float), (c, target, up))
    z = target - c; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])          # world -> cam (rows are cam axes); y points down in image
    return R, -R @ c


K = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1.0]])
SIZE = (1280, 720)
CENTERS = [(2.5, 1.6, 3.0), (-2.5, 1.5, 3.2), (0.2, 1.8, -3.5)]


def true_cams():
    out = []
    for i, c in enumerate(CENTERS):
        R, t = look_at(c, (0, 0.9, 0))
        out.append(CameraModel(f"cam{i}", K, np.zeros(5), R, t, SIZE))
    return out


def board_world(cols=9, rows=6, sq=0.05):
    obj = board_points(cols, rows, sq)
    M = M_UP_Z
    return (M @ obj.T).T + np.array([-0.2, 0.0, 0.1])


def test_extrinsics_recover_camera_positions():
    cams = true_cams()
    # board frame origin sits at (-0.2,0,0.1) in the world; solve_extrinsics puts the origin at the board corner
    obs = []
    for cam in cams:
        corners = cam.project(board_world()).reshape(-1, 1, 2).astype(np.float32)
        obs.append((cam.name, corners, K, np.zeros(5), SIZE))
    rig = solve_extrinsics(obs, 9, 6, 0.05)
    shift = np.array([-0.2, 0.0, 0.1])
    for rc, tc in zip(rig.cameras, cams):
        assert np.linalg.norm((rc.center + shift) - tc.center) < 0.02, (rc.center, tc.center)
        assert rc.center[1] > 0
    assert max(rig.rms.values()) < 0.2


def test_triangulation_accuracy_and_outlier_rejection():
    cams = true_cams()
    rig = Rig(cams)
    rng = np.random.default_rng(0)
    X = rng.uniform([-0.5, 0.0, -0.5], [0.5, 1.8, 0.5], (33, 3))
    pts = [c.project(X) + rng.normal(0, 0.3, (33, 2)) for c in cams]
    vis = [np.ones(33) for _ in cams]
    est, used = rig.triangulate(pts, vis)
    assert np.nanmax(np.linalg.norm(est - X, axis=1)) < 0.02
    assert (used == 3).all()
    pts[1][5] += [150, -90]                       # one camera is badly wrong for joint 5
    est, used = rig.triangulate(pts, vis)
    assert np.linalg.norm(est[5] - X[5]) < 0.03 and used[5] == 2
    vis[0][7] = vis[1][7] = 0.0                    # only one camera sees joint 7
    est, used = rig.triangulate(pts, vis)
    assert np.isnan(est[7]).all()


def test_intrinsics_solve_from_views():
    obj = board_points(9, 6, 0.03)
    rng = np.random.default_rng(3)
    col = IntrinsicsCollector(9, 6, 0.03)
    col.size = SIZE
    for _ in range(14):
        rvec = rng.uniform(-0.5, 0.5, 3); tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.5, 0.9)])
        px, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
        col.views.append(px.astype(np.float32))
    K2, dist, rms, size = col.solve()
    assert rms < 0.2 and abs(K2[0, 0] - 900) < 15


def test_find_board_on_rendered_chessboard():
    sq = 60
    img = np.full((8 * sq, 11 * sq), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                img[(r + 1) * sq - sq // 2 + sq // 2 - sq // 2:(r + 1) * sq + sq // 2, (c + 1) * sq - sq // 2:(c + 1) * sq + sq // 2] = 0
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    found = find_board(img, 9, 6)
    assert found is not None and len(found) == 54
