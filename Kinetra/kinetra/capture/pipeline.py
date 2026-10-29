"""Capture worker: N cameras (or video files) -> parallel pose/face inference -> 3D (single-camera world
landmarks or multi-camera triangulation) -> BodySolver + face tracks -> results for UI, recorder, streamer.

One frame set is in flight at a time and always the newest one (no backlog, minimum latency). Body inference of
every camera and the face inference run concurrently on a thread pool; MediaPipe releases the GIL.
"""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6 import QtCore

from ..ai import load_models
from ..ai.actor_tracker import MultiActorTracker
from ..ai.prop_tracker import PropTracker
from ..ai.audio_lipsync import AudioLipSync
from .autocalib import MultiCamAutoSync
from ..ai.body_prior import LiveBodyPrior
from ..ai.hand_prior import hand_features, features_to_quats, finger_confidence
from ..core.anatomy import apply_limits, bone_axes
from ..core.body_solver import BodySolver
from ..core.constants import HIP_MID, NBS
from ..core.exporters import Streamer
from ..core.fullbody import JOINT_ORDER_V2, assemble, build_skeleton, forward_kinematics
from ..core.hand import HandSolver
from ..core.fullbody import q_pow
from ..core.mathx import OneEuroFilter, q_mul, q_conj, IDENTITY_Q
from ..core.project import TakeRecorder
from .cameras import CameraSource
from .face_tracks import FaceTracker, process_face, blend_vector
from . import models as default_models

INFER_WIDTH = 960


@dataclass
class CaptureConfig:
    sources: list = field(default_factory=lambda: [0])
    mode: str = "single"                  # "single" | "multi"
    preset: str = "1280x720 @60"
    pose_variant: str = "heavy"
    num_faces: int = 1
    num_poses: int = 1
    multi_actor: bool = False
    track_prop: bool = True
    prop_color: str = "green"
    prop_kind: str = "sword"
    prop_side: str = "R"
    audio_lipsync: bool = True
    face_cam: int = 0                     # index into sources
    use_gpu: bool = False
    track_body: bool = True
    track_face: bool = True
    track_hands: bool = True
    use_ai: bool = True
    face_rig: object = None               # kinetra.core.facerig.FaceRig (personal face scan)
    markers: bool = False
    rig: object = None                    # kinetra.core.multicam.Rig for mode == "multi"
    focal_factor: float = 0.9
    beta: float = 5.0
    f_min: float = 1.0


def estimate_root(img, wld, frame_w, frame_h, focal_factor, cam_z=3.4):
    """Pinhole estimate of the pelvis position from shoulder width (pixels vs metres)."""
    if min(img[[11, 12, 23, 24], 3]) < 0.5:
        return None
    f = focal_factor * frame_w
    ps = math.hypot((img[11, 0] - img[12, 0]) * frame_w, (img[11, 1] - img[12, 1]) * frame_h)
    ws = float(np.linalg.norm(wld[11, :3] - wld[12, :3]))
    if ps < 8 or ws < 0.1:
        return None
    depth = float(np.clip(f * ws / ps, 0.6, 12.0))
    u = (img[23, 0] + img[24, 0]) / 2 * frame_w
    return ((u - frame_w / 2) * depth / f, cam_z - depth)


class CaptureWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(object)
    message = QtCore.Signal(str, object)        # (i18n key, args tuple)
    state = QtCore.Signal(str)

    def __init__(self, cfg, models=None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.models = models or default_models
        self.solver = BodySolver(cfg.f_min, cfg.beta)
        self.faces = FaceTracker()
        self.recorder = TakeRecorder()
        self.streamer = Streamer()
        self.focal_factor = cfg.focal_factor
        self.markers_enabled = cfg.markers
        self._running = False
        self._calib = False
        self._pend = {"reg": False, "neutral": False, "rom": False}
        self._video = None
        self._vlock = threading.Lock()
        self.sources, self._pose, self._face = [], [], None
        self._last_ts = {}
        self._last_abs = None
        self.last_lengths = None
        self._hands_lm = []
        self.hand_solvers = {"L": HandSolver(), "R": HandSolver()}
        self._hand_f = {"L": OneEuroFilter(2.0, 8.0), "R": OneEuroFilter(2.0, 8.0)}
        self.skel, self._skel_axes, self._skel_n = build_skeleton(), None, 0
        self.live_body, self._hand_ai, self._prev_hand, self._last_local = None, {}, {}, {}
        self.ai_stats = {"body_filled": 0, "hand_states": {}}
        self.actor_tracker = MultiActorTracker(max_actors=2)
        self.prop_tracker = PropTracker(color=getattr(cfg, 'prop_color', 'green'), kind=getattr(cfg, 'prop_kind', 'sword'), side=getattr(cfg, 'prop_side', 'R'))
        self.lipsync = AudioLipSync(enabled=getattr(cfg, 'audio_lipsync', True))
        self.auto_sync = MultiCamAutoSync(focal_factor=cfg.focal_factor)
        self.recorder.order = JOINT_ORDER_V2

    # -- commands from the UI ---------------------------------------------------------------------
    def set_prop_config(self, kind=None, color=None, side=None, enabled=None):
        if enabled is not None:
            self.prop_tracker.enabled = bool(enabled)
            self.cfg.track_prop = bool(enabled)
        self.prop_tracker.set_config(kind=kind, color=color, side=side)

    def set_audio_lipsync(self, on):
        self.cfg.audio_lipsync = bool(on)
        self.lipsync.enabled = bool(on)
        if on and not self.lipsync._running:
            self.lipsync.start()
        elif not on and self.lipsync._running:
            self.lipsync.stop()

    def run_auto_sync_calibration(self, sizes):
        return self.auto_sync.solve(sizes)

    def set_multi_actor(self, on):
        self.cfg.multi_actor = bool(on)
        self.cfg.num_poses = 2 if on else 1
        if not on:
            self.actor_tracker.reset()

    def request_calibration(self):
        self._calib = True

    def request(self, what):
        self._pend[what] = True

    def set_markers(self, on):
        self.markers_enabled = on
        if not on:
            self.faces.clear_markers()

    def set_params(self, beta, f_min, focal):
        self.solver.set_filter(beta, f_min)
        self.focal_factor = focal

    def set_profile_bias(self, bias):
        self.faces.profile_bias = None if bias is None else np.asarray(bias, float)

    def start_video(self, directory):
        with self._vlock:
            self._video = {"dir": directory, "writers": {}}

    def stop_video(self):
        with self._vlock:
            v, self._video = self._video, None
        for w in (v["writers"].values() if v else []):
            w.release()

    def stop(self):
        self._running = False
        self.wait(8000)

    def _msg(self, key, *args):
        self.message.emit(key, args)

    # -- inference tasks (run on the pool) --------------------------------------------------------
    def _ts(self, key, t):
        ts = max(int(t * 1000), self._last_ts.get(key, -1) + 1)
        self._last_ts[key] = ts
        return ts

    def _body_task(self, i, rgb, t):
        import mediapipe as mp
        t0 = time.perf_counter()
        data = None
        all_bodies = []
        try:
            res = self._pose[i].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts(("p", i), t))
            if res.pose_landmarks:
                for k, pl in enumerate(res.pose_landmarks):
                    img = np.array([[l.x, l.y, l.z, l.visibility] for l in pl])
                    wld = (np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_world_landmarks[k]])
                           if res.pose_world_landmarks and k < len(res.pose_world_landmarks) else None)
                    all_bodies.append({"image": img, "world": wld})
                if all_bodies:
                    data = dict(all_bodies[0])
                    data["all"] = all_bodies
        except Exception as exc:
            self._msg("log.fault", f"BODY CAM {i}: {exc}")
        return data, (time.perf_counter() - t0) * 1e6

    def _face_task(self, rgb, t):
        import mediapipe as mp
        t0 = time.perf_counter()
        out = []
        try:
            res = self._face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts("f", t))
            for k, f in enumerate(res.face_landmarks):
                lm = np.array([[p.x, p.y, p.z] for p in f])
                bs = blend_vector(res.face_blendshapes[k]) if res.face_blendshapes else np.zeros(NBS)
                out.append((lm, bs))
        except Exception as exc:
            self._msg("log.fault", f"FACE: {exc}")
        return out, (time.perf_counter() - t0) * 1e6

    def _hand_task(self, k, rgb, t):
        import mediapipe as mp
        t0 = time.perf_counter()
        out = []
        try:
            res = self._hands_lm[k].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts(("h", k), t))
            for n, lms in enumerate(res.hand_landmarks):
                cat = res.handedness[n][0] if res.handedness else None
                out.append({"img": np.array([[l.x, l.y, l.z] for l in lms]),
                            "world": np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[n]]),
                            "label": cat.category_name if cat else "", "score": float(cat.score) if cat else 0.5})
        except Exception as exc:
            self._msg("log.fault", f"HANDS {k}: {exc}")
        return out, (time.perf_counter() - t0) * 1e6

    @staticmethod
    def assign_hands(hands, body_img):
        """Map detections to sides by distance to the body wrists (robust to mirrored handedness labels)."""
        out = {}
        if body_img is not None:
            pairs = sorted(((float(np.linalg.norm(h["img"][0, :2] - body_img[w, :2])), i, s)
                            for i, h in enumerate(hands) for w, s in ((15, "L"), (16, "R"))), key=lambda x: x[0])
            used = set()
            for d, i, s in pairs:
                if d < 0.25 and i not in used and s not in out:
                    out[s] = hands[i]
                    used.add(i)
        else:
            for h in hands:
                s = "R" if h["label"] == "Left" else "L"          # labels assume a mirrored image
                out.setdefault(s, h)
        return out

    def _hands_scene(self, per_cam):
        """per_cam: list of {side: hand} per hand-camera -> {side: (21,3) scene-axes landmarks, wrist-relative}."""
        multi = self.cfg.mode == "multi" and self.cfg.rig is not None and len(self.cfg.rig) >= 2
        flip = np.diag([1.0, -1.0, -1.0])
        res = {}
        for s in "LR":
            seen = [(c, d[s]) for c, d in enumerate(per_cam) if s in d]
            if not seen:
                continue
            P = None
            if multi and len(seen) >= 2:
                pts, vis = [None] * len(self.cfg.rig), [None] * len(self.cfg.rig)
                for c, h in seen:
                    pts[c] = h["img"][:, :2] * np.array(self.cfg.rig.cameras[c].size, float)
                    vis[c] = np.ones(21)
                X, _ = self.cfg.rig.triangulate(pts, vis)
                if not np.isnan(X).any():
                    P = X
            if P is None:
                c, h = max(seen, key=lambda x: x[1]["score"])
                R = self.cfg.rig.cameras[c].R.T if multi else flip
                P = h["world"] @ R.T
            res[s] = P - P[0]
        return res

    # -- 3D body ----------------------------------------------------------------------------------
    def _body_3d(self, bodies, sizes):
        """Returns (J33 y-up, vis, root_xz or None, absolute, tri_used) or None."""
        if self.cfg.mode == "multi" and self.cfg.rig is not None and len(self.cfg.rig) >= 2:
            pts, vis = [], []
            for b, cam in zip(bodies, self.cfg.rig.cameras):
                if b is None:
                    pts.append(None)
                    vis.append(None)
                else:
                    pts.append(b["image"][:, :2] * np.array(cam.size, float))
                    vis.append(b["image"][:, 3])
            if sum(p is not None for p in pts) < 2:
                return None
            X, used = self.cfg.rig.triangulate(pts, vis)
            good = ~np.isnan(X).any(axis=1)
            if good.sum() < 20:
                return None
            if self._last_abs is not None:
                X = np.where(good[:, None], X, self._last_abs)
            elif not good.all():
                X = np.where(good[:, None], X, np.nanmean(X, axis=0))
            if self.live_body is not None:
                X, rep = self.live_body.fill(X, good.astype(float))
                self.ai_stats["body_filled"] = int(rep.sum())
            self._last_abs = X.copy()
            return X, good.astype(float), None, True, float(used[good].mean())
        b = bodies[0]
        if b is None or b["world"] is None:
            return None
        w = b["world"]
        J = w[:, :3] * np.array([1.0, -1.0, -1.0])
        if self.live_body is not None:                         # joints the camera cannot see are predicted, not frozen
            J, rep = self.live_body.fill(J, w[:, 3])
            self.ai_stats["body_filled"] = int(rep.sum())
        fw, fh = sizes[0]
        root = estimate_root(b["image"], w, fw, fh, self.focal_factor)
        return J, w[:, 3], root, False, 1.0

    # -- main loop --------------------------------------------------------------------------------
    def _make_sources(self):
        out = []
        for i, s in enumerate(self.cfg.sources):
            src = s if hasattr(s, "get") else CameraSource(s, self.cfg.preset, f"CAM {i}")
            if not src.open():
                self._msg("log.cam_fail", i)
                return None
            self._msg("log.cam_online", i, src.info)
            out.append(src)
        return out

    def run(self):
        cfg = self.cfg
        self.sources = self._make_sources()
        if not self.sources:
            self.state.emit("ERROR")
            return
        try:
            if cfg.track_body:
                n_poses = 2 if cfg.multi_actor else cfg.num_poses
                try:
                    self._pose = [self.models.create_pose(cfg.pose_variant, num_poses=n_poses,
                                                          log=lambda m: self._msg("log.raw", m))
                                  for _ in self.sources]
                except TypeError:
                    self._pose = [self.models.create_pose(cfg.pose_variant,
                                                          log=lambda m: self._msg("log.raw", m))
                                  for _ in self.sources]
            if cfg.track_hands:
                n_hand_cams = len(self.sources) if cfg.mode == "multi" else 1
                self._hands_lm = [self.models.create_hands(2, lambda m: self._msg("log.raw", m)) for _ in range(n_hand_cams)]
            if cfg.track_face:
                self._face, delegate = self.models.create_face(num_faces=cfg.num_faces, prefer_gpu=cfg.use_gpu,
                                                               log=lambda m: self._msg("log.raw", m))
                self._msg("log.face_ready", delegate, cfg.num_faces)
        except Exception as exc:
            self._msg("log.init_fail", str(exc))
            for s in self.sources:
                s.stop()
            self.state.emit("ERROR")
            return
        if cfg.face_rig is not None:
            rg = cfg.face_rig
            self.faces.profile_bias, self.faces.profile_gain, self.faces.profile_neutral = rg.bias, rg.gain, rg.identity
            self.recorder.face_idx = rg.feat
        if cfg.use_ai:
            try:
                hp, bp = load_models()
                if bp is not None:
                    self.live_body = LiveBodyPrior(bp)
                if hp is not None:
                    self._hand_ai = {side: hp.new_filter() for side in "LR"}
                self._msg("log.ai_ready", int(bp is not None), int(hp is not None))
            except Exception as exc:
                self._msg("log.fault", f"AI: {exc}")
        self.streamer.configure()
        if getattr(cfg, 'audio_lipsync', True):
            self.lipsync.start()
        pool = ThreadPoolExecutor(max_workers=2 * len(self.sources) + 1)
        for s in self.sources:
            s.start()
        self._running = True
        self.state.emit("RUNNING")
        idx = consumed = 0
        stalls = 0
        try:
            while self._running:
                items = [s.get(0.25) for s in self.sources]
                if any(it is None for it in items):
                    stalls += 1
                    if all(s.ended for s in self.sources if s.is_file) and any(s.ended for s in self.sources):
                        self._msg("log.source_ended")
                        break
                    if stalls > 80:
                        self._msg("log.stream_lost")
                        break
                    continue
                stalls = 0
                consumed += 1
                idx += 1
                t_cap = max(t for _, t in items)
                skew = (t_cap - min(t for _, t in items)) * 1e6
                try:
                    self._process(pool, idx, items, t_cap, skew, consumed)
                except Exception as exc:
                    self._msg("log.fault", f"FUSION: {exc!r}")
        finally:
            self._running = False
            pool.shutdown(wait=False)
            self.lipsync.stop()
            self.stop_video()
            for s in self.sources:
                s.stop()
            for lm in list(self._pose) + list(self._hands_lm) + ([self._face] if self._face else []):
                try:
                    lm.close()
                except Exception:
                    pass
            self.streamer.close()
            self.state.emit("STOPPED")

    def _process(self, pool, idx, items, t_cap, skew, consumed):
        cfg = self.cfg
        frames = [f for f, _ in items]
        sizes = [(f.shape[1], f.shape[0]) for f in frames]
        small = []
        for f in frames:
            w = f.shape[1]
            small.append(cv2.resize(f, (INFER_WIDTH, int(round(f.shape[0] * INFER_WIDTH / w)))) if w > INFER_WIDTH else f)
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in small]
        fc = min(cfg.face_cam, len(rgb) - 1)

        if self._calib:
            self._calib = False
            self.solver.start_calibration()
            self._msg("log.calib_armed")
            self._pend["neutral"] = True

        t0 = time.perf_counter()
        body_f = [pool.submit(self._body_task, i, rgb[i], t_cap) for i in range(len(rgb))] if cfg.track_body else []
        face_f = pool.submit(self._face_task, rgb[fc], t_cap) if cfg.track_face else None
        hand_f = [pool.submit(self._hand_task, k, rgb[k], t_cap) for k in range(len(self._hands_lm))]
        bodies, a_us = [], 0.0
        for f in body_f:
            d, us = f.result()
            bodies.append(d)
            a_us = max(a_us, us)
        faces_raw, b_us = face_f.result() if face_f else ([], 0.0)
        hands_det = [f.result()[0] for f in hand_f]
        t_inf = time.perf_counter()

        res = {"idx": idx, "frames": small, "body_img": [b["image"] if b else None for b in bodies] or [None] * len(small),
               "joints": None, "quats": None, "locks": np.zeros(2, dtype=bool), "faces": [], "tri_used": 0.0,
               "calibrating": self.solver.calibrating, "mode": cfg.mode, "root_raw": None, "J_raw": None}

        b3 = self._body_3d(bodies, sizes) if cfg.track_body else None
        if b3 is not None:
            J33, vis, root, absolute, used = b3
            out = self.solver.process(J33, t_cap, root=root, vis=vis, absolute=absolute)
            res.update(joints=out["joints"], quats=out["quats"], locks=out["locks"], tri_used=used,
                       J_raw=J33, vis=vis, root_raw=root, absolute=absolute, lengths=out["lengths"],
                       calibrating=self.solver.calibrating,
                       vis_mean=float(np.mean(vis[11:33])), foot_vis=float(np.min(vis[27:33])))
            res["world"] = out["world"]
            self.last_lengths = out["lengths"]
            if out["calib_msg"]:
                self._msg("log.calib_done")
                self._skel_n = 0

        # Multi-Actor ReID Appearance Matching
        detected_candidates = []
        if cfg.track_body and bodies and bodies[0] is not None:
            if "all" in bodies[0]:
                detected_candidates = bodies[0]["all"]
            else:
                detected_candidates = [bodies[0]]

        tracked_actors = []
        if (cfg.multi_actor or getattr(self.cfg, "multi_actor", False)) and len(small) > 0:
            if detected_candidates:
                tracked_actors = self.actor_tracker.match_and_update(rgb[0], detected_candidates, t_cap)
            else:
                tracked_actors = list(self.actor_tracker.profiles.values())
        res["actors"] = tracked_actors

        if len(tracked_actors) > 1 and tracked_actors[1].joints_3d is not None:
            sec_w = tracked_actors[1].joints_3d
            res["secondary_joints"] = sec_w[:, :3] * np.array([1.0, -1.0, -1.0])


        # Multi-Cam Auto-Sync Collection
        if cfg.mode == "multi" and bodies:
            for k, b in enumerate(bodies):
                if b is not None and "image" in b:
                    self.auto_sync.add_frame_landmarks(k, b["image"])

        if cfg.track_face:
            gray = cv2.cvtColor(small[fc], cv2.COLOR_BGR2GRAY) if self.markers_enabled else None
            cmds = dict(self._pend)
            for k in self._pend:
                self._pend[k] = False
            ev = lambda e: self._msg("log.face_event", *e)
            assign = self.faces.match([lm[:468, :2].mean(0) for lm, _ in faces_raw], t_cap, ev)
            fh, fw = small[fc].shape[:2]
            for i, (lm, bs) in enumerate(faces_raw):
                pk = process_face(self.faces.tracks[assign[i]], lm, bs, (fw, fh), gray, t_cap, cmds, self.markers_enabled)
                for e in pk.pop("events"):
                    ev(e)
                res["faces"].append(pk)
            self.faces.prune(t_cap, ev)

            slot = self.faces.slots()
            for pk in res["faces"]:
                pk["slot"] = slot.get(pk["id"], 0)
            res["faces"].sort(key=lambda p: p["slot"])
            if cfg.face_rig is not None and res["faces"]:
                res["faces"][0]["personal"] = cfg.face_rig.solve(res["faces"][0]["L"][cfg.face_rig.feat])

        primary = res["faces"][0]["bs"] if res["faces"] else None
        audio_lipsync_active = False
        if primary is None and getattr(cfg, "audio_lipsync", True):
            bs_audio = self.lipsync.get_blendshapes()
            if float(np.max(bs_audio)) > 0.01:
                primary = bs_audio
                audio_lipsync_active = True
        res["audio_lipsync_active"] = audio_lipsync_active
        face_head_q = res["faces"][0]["quat"] if res["faces"] else None

        # hands (finger solve) and the full 52-joint pose
        hands_res, hands_raw = {}, np.full((2, 21, 3), np.nan)
        if cfg.track_hands and res["joints"] is not None:
            per_cam = [self.assign_hands(h, bodies[k]["image"] if k < len(bodies) and bodies[k] else None)
                       for k, h in enumerate(hands_det)]
            scene = self._hands_scene(per_cam)
            fa_world = res.get("world") or {}
            for hi, side in enumerate("LR"):
                ai = self._hand_ai.get(side)
                if side in scene:
                    P = self._hand_f[side](scene[side], t_cap)
                    hands_raw[hi] = P
                    hr = self.hand_solvers[side].solve(P)
                    if ai is not None:
                        x = hand_features(hr["quats"], side)
                        conf = finger_confidence(P, self.hand_solvers[side].bone_lengths(), self._prev_hand.get(side))
                        out = ai.update(x, conf)
                        pq = features_to_quats(out["pose"], side)
                        for i, f in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                            sl = slice(4 * i, 4 * i + 4)
                            if conf[i] < 0.9 or float(np.abs(out["pose"][sl] - x[sl]).max()) > 0.1:
                                for j in (1, 2, 3):
                                    hr["quats"][f"{f}_{j}"] = pq[f"{f}_{j}"]
                        hr["ai"] = out
                    self._prev_hand[side] = P
                    hands_res[side] = hr
                    if "forearm_" + side in fa_world:
                        self._last_local[side] = q_mul(q_conj(fa_world["forearm_" + side]), hr["palm_q"])
                else:
                    self._hand_f[side].reset()
                    self._prev_hand.pop(side, None)
                    if ai is not None and side in self._last_local and "forearm_" + side in fa_world:
                        out = ai.update(None, None)                    # hidden hand: most probable pose, not a frozen one
                        k = min(1.0, out["missing"] / 1.5)
                        local = q_mul(q_pow(self._last_local[side], 1.0 - k), q_pow(IDENTITY_Q, k))
                        hands_res[side] = {"palm_q": q_mul(fa_world["forearm_" + side], local),
                                           "quats": features_to_quats(out["pose"], side), "ai": out, "curl": {}}
        res["hands"], res["quats_full"], res["fk"], res["skel"], res["limit_hits"] = hands_res, None, None, self.skel, []

        # Prop / Weapon Tracking
        prop_res = None
        if getattr(cfg, "track_prop", True) and res.get("joints") is not None and len(small) > 0:
            side = getattr(cfg, "prop_side", "R")
            wrist_idx = 16 if side == "R" else 15
            wrist_img = bodies[0]["image"][wrist_idx] if bodies and bodies[0] is not None else None
            wrist_3d = res["joints"][wrist_idx]
            hand_q = hands_res.get(side, {}).get("palm_q")
            prop_res = self.prop_tracker.process(rgb[0], wrist_img, wrist_3d, hand_quat=hand_q, t_now=t_cap)
        res["prop"] = prop_res
        if res["quats"] is not None:
            self._skel_n += 1
            if self._skel_n % 30 == 1:
                self.skel = build_skeleton(res["lengths"], self.hand_solvers)
                self._skel_axes = bone_axes(self.skel)
            if self._skel_axes is None:
                self._skel_axes = bone_axes(self.skel)
            qd = assemble(res["quats"], res["world"], hands_res, primary, head_q=face_head_q)
            qd, hit = apply_limits(qd, self.skel, self._skel_axes)
            res.update(quats_full=qd, skel=self.skel, limit_hits=hit,
                       fk=forward_kinematics(self.skel, qd, res["joints"][HIP_MID]))

        if self.recorder.active and (res["joints"] is not None or primary is not None):
            res_j = res["joints"]
            self.recorder.add(t_cap, res.get("J_raw"), res.get("vis"), res.get("root_raw"),
                              None if res_j is None else res_j[HIP_MID], res["quats_full"], primary,
                              res.get("absolute", False), res.get("lengths"), hands_raw, self.skel,
                              res["faces"][0]["L"] if res["faces"] else None)
        if self.streamer.enabled and res["quats_full"] is not None:
            self.streamer.send(idx, res["joints"][HIP_MID], res["quats_full"], [p["bs"] for p in res["faces"]], prop=res.get("prop"))
        with self._vlock:
            if self._video is not None:
                self._write_video(frames, t_cap)

        t_now = time.perf_counter()
        res.update(a_us=a_us, b_us=b_us, infer_us=(t_inf - t0) * 1e6, total_us=(t_now - t_cap) * 1e6, sync_us=skew,
                   cam_fps=[s.fps for s in self.sources], drops=max(self.sources[0].produced - consumed, 0)
                   if hasattr(self.sources[0], "produced") else 0, bs=primary if primary is not None else np.zeros(NBS))
        self.frame_ready.emit(res)

    def _write_video(self, frames, t):
        v = self._video
        for i, f in enumerate(frames):
            w = v["writers"].get(i)
            if w is None:
                fps = self.sources[i].fps or 30.0
                w = cv2.VideoWriter(str(v["dir"] / f"cam{i}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                    (f.shape[1], f.shape[0]))
                v["writers"][i] = w
            w.write(f)
