"""Guided face scan: the camera shows a guide circle and tells the actor what to do (hold the head inside the
circle, turn left / right, open the mouth, smile ...). When every step is captured the personal face rig is built."""
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..capture.cameras import CameraSource
from ..capture.face_regions import feature_regions, feature_triangles
from ..capture.face_tracks import blend_vector
from ..capture.models import create_face
from ..core.face import to_face_frame, euler_from_R
from ..core.facerig import FaceEnrollment, FaceRig
from ..i18n import t
from .theme import C_AMBER, C_BG, C_CYAN, C_GREEN, C_DIM


class _ScanThread(QtCore.QThread):
    info = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, src, enrollment, models=None, parent=None):
        super().__init__(parent)
        self.src, self.enr, self.models = src, enrollment, models
        self.running = True
        self.cmds = []                                # "skip" / "now" / "restart" requested by the UI

    def run(self):
        import mediapipe as mp
        try:
            lm, _ = (self.models.create_face if self.models else create_face)(num_faces=1)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        last_ts = -1
        while self.running:
            item = self.src.get(0.2)
            if item is None:
                continue
            while self.cmds:
                c = self.cmds.pop(0)
                {"skip": self.enr.skip, "now": self.enr.capture_now, "restart": self.enr.restart}[c]()
            frame, tcap = item
            h0, w0 = frame.shape[:2]
            if w0 > 640:
                frame = cv2.resize(frame, (640, int(h0 * 640 / w0)))
            h, w = frame.shape[:2]
            last_ts = max(int(tcap * 1000), last_ts + 1)
            try:
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), last_ts)
            except Exception:
                continue
            out = {"frame": frame, "face": False, "state": self.enr.state(False), "in_circle": False}
            if res.face_landmarks:
                pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])
                P = np.column_stack([pts[:, 0] * w, pts[:, 1] * h, pts[:, 2] * w])
                L, R, _o, s = to_face_frame(P)
                yaw, pitch, _roll = euler_from_R(R)
                cx, cy = pts[:468, 0].mean(), pts[:468, 1].mean()
                fw = pts[:468, 0].max() - pts[:468, 0].min()
                inside = abs(cx - 0.5) < 0.14 and abs(cy - 0.5) < 0.18 and 0.20 < fw < 0.55
                bs = blend_vector(res.face_blendshapes[0]) if res.face_blendshapes else np.zeros(52)
                out.update(face=True, in_circle=inside, state=self.enr.update(L, bs, yaw, pitch, inside),
                           yaw=yaw, pitch=pitch)
            self.info.emit(out)
        lm.close()


class ScanView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame, self.state, self.in_circle, self.face = None, None, False, False
        self.setMinimumSize(640, 480)

    def set_data(self, frame, state, in_circle, face):
        self.frame, self.state, self.in_circle, self.face = frame, state, in_circle, face
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(C_BG))
        if self.frame is None:
            p.setPen(QtGui.QColor(C_AMBER))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, t("nosignal"))
            return
        f = self.frame
        fh, fw = f.shape[:2]
        img = QtGui.QImage(f.data, fw, fh, f.strides[0], QtGui.QImage.Format_BGR888)
        sc = min(self.width() / fw, self.height() / fh)
        dw, dh = fw * sc, fh * sc
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        p.drawImage(QtCore.QRectF(ox, oy, dw, dh), img)
        cx, cy, rx, ry = ox + dw / 2, oy + dh / 2, dw * 0.19, dh * 0.34
        ok = self.state and self.state["ok"]
        col = QtGui.QColor(C_GREEN if ok else (C_CYAN if self.in_circle else C_AMBER))
        p.setBrush(QtCore.Qt.NoBrush)
        pen = QtGui.QPen(col, 3, QtCore.Qt.SolidLine if self.in_circle else QtCore.Qt.DashLine)
        p.setPen(pen)
        p.drawEllipse(QtCore.QPointF(cx, cy), rx, ry)
        if self.state and self.state["progress"] > 0:
            p.setPen(QtGui.QPen(QtGui.QColor(C_GREEN), 8))
            p.drawArc(QtCore.QRectF(cx - rx - 8, cy - ry - 8, 2 * rx + 16, 2 * ry + 16), 90 * 16, int(-360 * 16 * self.state["progress"]))
        if self.state:
            font = p.font()
            font.setPointSize(15)
            font.setBold(True)
            p.setFont(font)
            msg = t("scan.finished") if self.state["done"] else t(self.state["hint"])
            if not self.face or not self.in_circle:
                msg = t("scan.circle_hint")
            p.setPen(QtGui.QColor("#000000"))
            r = QtCore.QRectF(ox, oy + 8, dw, 60)
            p.drawText(r.translated(2, 2), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, msg)
            p.setPen(col)
            p.drawText(r, QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, msg)


class FaceScanDialog(QtWidgets.QDialog):
    rig_ready = QtCore.Signal(object)

    def __init__(self, source, preset, save_path, parent=None, models=None, src=None):
        super().__init__(parent)
        self.setWindowTitle(t("act.face_scan"))
        self.resize(900, 760)
        self.save_path, self.rig, self.models = Path(save_path) if save_path else None, None, models
        self.enr = FaceEnrollment()
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(t("scan.intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.view = ScanView()
        lay.addWidget(self.view, 1)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, len(self.enr.steps))
        self.bar.setTextVisible(True)
        lay.addWidget(self.bar)
        self.score = QtWidgets.QProgressBar()                    # how close the current pose / expression is
        self.score.setRange(0, 100)
        self.score.setTextVisible(True)
        self.score.setFormat(t("scan.score") + " %p%")
        lay.addWidget(self.score)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setStyleSheet(f"color:{C_DIM};")
        lay.addWidget(self.lbl)
        row = QtWidgets.QHBoxLayout()
        self.btn_restart = QtWidgets.QPushButton(t("scan.restart"))
        self.btn_restart.clicked.connect(self._restart)
        self.btn_skip = QtWidgets.QPushButton(t("scan.skip"))
        self.btn_skip.clicked.connect(lambda: self._cmd("skip"))
        self.btn_now = QtWidgets.QPushButton(t("scan.capture_now"))
        self.btn_now.clicked.connect(lambda: self._cmd("now"))
        self.btn_close = QtWidgets.QPushButton(t("btn.close"))
        self.btn_close.clicked.connect(self.accept)
        for b in (self.btn_restart, self.btn_skip, self.btn_now):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.src = src or CameraSource(source, preset, "SCAN")
        if not self.src.open():
            self.lbl.setText(t("log.cam_fail", 0))
            self.thread = None
            return
        self.src.start()
        self.thread = _ScanThread(self.src, self.enr, models)
        self.thread.info.connect(self._on_info)
        self.thread.failed.connect(lambda m: self.lbl.setText(t("log.init_fail", m)))
        self.thread.start()

    def _cmd(self, c):
        if self.thread is not None:
            self.thread.cmds.append(c)

    def _restart(self):
        self.rig = None
        self.bar.setValue(0)
        self._cmd("restart")

    def _on_info(self, d):
        st = d["state"]
        self.view.set_data(d["frame"], st, d["in_circle"], d["face"])
        self.bar.setValue(st["idx"])
        self.score.setValue(int(min(st["score"], 1.0) * 100))
        self.btn_skip.setEnabled(st["skippable"] and not st["done"])
        self.btn_now.setEnabled(not st["done"])
        self.bar.setFormat(t("scan.step", min(st["idx"] + 1, st["total"]), st["total"]))
        if st["done"] and self.rig is None:
            self._finish()

    def _finish(self):
        try:
            regions = feature_regions()
            self.rig = FaceRig.from_captures(self.enr.captures, regions, feature_triangles(regions))
            if self.save_path:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.rig.save(self.save_path)
            self.lbl.setText(t("scan.saved", str(self.save_path) if self.save_path else ""))
            self.rig_ready.emit(self.rig)
        except Exception as exc:
            self.lbl.setText(t("scan.fail", exc))

    def done(self, r):
        if getattr(self, "thread", None) is not None:
            self.thread.running = False
            self.thread.wait(2000)
        self.src.stop()
        super().done(r)
