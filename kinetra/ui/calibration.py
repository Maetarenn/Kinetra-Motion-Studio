"""Multi-camera calibration wizard: lens calibration per camera + floor board for the shared world frame."""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..capture.cameras import CameraSource
from ..core.multicam import (IntrinsicsCollector, find_board, solve_extrinsics, default_intrinsics)
from ..i18n import t
from .widgets import CameraGrid


class _Detector(threading.Thread):
    """Looks for the chessboard in the newest frame of every camera without blocking the UI."""

    def __init__(self, sources, get_pattern):
        super().__init__(daemon=True)
        self.sources, self.get_pattern = sources, get_pattern
        self.corners = [None] * len(sources)
        self.sizes = [None] * len(sources)
        self.collect = None                       # (camera index, IntrinsicsCollector) or None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            cols, rows = self.get_pattern()
            for i, src in enumerate(self.sources):
                item = src.peek()
                if item is None:
                    continue
                gray = cv2.cvtColor(item[0], cv2.COLOR_BGR2GRAY)
                c = find_board(gray, cols, rows)
                with self.lock:
                    self.corners[i], self.sizes[i] = c, (gray.shape[1], gray.shape[0])
                    col = self.collect
                    if col is not None and col[0] == i and c is not None:
                        col[1].add_corners(c, self.sizes[i])
            time.sleep(0.02)


class CalibrationDialog(QtWidgets.QDialog):
    rig_saved = QtCore.Signal(str)

    def __init__(self, sources, preset, rig_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("cal.title"))
        self.resize(1100, 720)
        self.sources_spec, self.preset, self.rig_path = sources, preset, Path(rig_path) if rig_path else None
        self.cams, self.detector = [], None
        self.intr, self.collectors, self.floor_obs, self.rig = {}, {}, None, None
        self._build()
        self._open_cameras()
        self.timer = QtCore.QTimer(self, interval=100, timeout=self._tick)
        self.timer.start()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(t("cal.intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QtWidgets.QHBoxLayout()
        self.sp_cols, self.sp_rows = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sp, v in ((self.sp_cols, 9), (self.sp_rows, 6)):
            sp.setRange(3, 30)
            sp.setValue(v)
        self.sp_sq = QtWidgets.QDoubleSpinBox()
        self.sp_sq.setRange(5, 500)
        self.sp_sq.setValue(30.0)
        for lab, w in ((t("cal.cols"), self.sp_cols), (t("cal.rows"), self.sp_rows), (t("cal.square"), self.sp_sq)):
            row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.view = CameraGrid("view.optical")
        self.view.show_body = self.view.show_face = False
        lay.addWidget(self.view, 1)
        g1 = QtWidgets.QGroupBox(t("cal.s1"))
        h1 = QtWidgets.QHBoxLayout(g1)
        self.cb_cam = QtWidgets.QComboBox()
        self.btn_collect = QtWidgets.QPushButton(t("cal.grab"))
        self.btn_collect.setCheckable(True)
        self.btn_collect.toggled.connect(self._toggle_collect)
        self.lbl_views = QtWidgets.QLabel(t("cal.views", 0))
        self.btn_lens = QtWidgets.QPushButton(t("cal.solve"))
        self.btn_lens.clicked.connect(self._solve_lens)
        for w in (QtWidgets.QLabel(t("cal.camera")), self.cb_cam, self.btn_collect, self.lbl_views, self.btn_lens):
            h1.addWidget(w)
        h1.addStretch(1)
        lay.addWidget(g1)
        g2 = QtWidgets.QGroupBox(t("cal.s2"))
        h2 = QtWidgets.QHBoxLayout(g2)
        self.btn_floor = QtWidgets.QPushButton(t("cal.floor"))
        self.btn_floor.clicked.connect(self._capture_floor)
        self.btn_rig = QtWidgets.QPushButton(t("cal.solve_rig"))
        self.btn_rig.clicked.connect(self._solve_rig)
        self.btn_save = QtWidgets.QPushButton(t("cal.save"))
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self._save)
        self.btn_rig.setEnabled(False)
        self.btn_save.setEnabled(False)
        hint = QtWidgets.QLabel(t("cal.floor_hint"))
        for w in (self.btn_floor, self.btn_rig, self.btn_save, hint):
            h2.addWidget(w)
        h2.addStretch(1)
        lay.addWidget(g2)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        lay.addWidget(self.log)

    def _say(self, msg):
        self.log.appendPlainText(msg)

    def _open_cameras(self):
        for i, s in enumerate(self.sources_spec):
            src = CameraSource(s, self.preset, f"CAM {i}")
            if not src.open():
                self._say(t("log.cam_fail", i))
                continue
            src.start()
            self.cams.append(src)
            self.cb_cam.addItem(f"CAM {i}", len(self.cams) - 1)
        self.detector = _Detector(self.cams, lambda: (self.sp_cols.value(), self.sp_rows.value()))
        self.detector.start()

    def _tick(self):
        frames, k = [], []
        for i, src in enumerate(self.cams):
            item = src.peek()
            if item is None:
                continue
            f = item[0].copy()
            with self.detector.lock:
                c = self.detector.corners[i]
            if c is not None:
                cv2.drawChessboardCorners(f, (self.sp_cols.value(), self.sp_rows.value()), c, True)
            frames.append(f)
        self.view.set_data(frames, [None] * len(frames), [], 0)
        idx = self.cb_cam.currentData()
        if idx in self.collectors:
            self.lbl_views.setText(t("cal.views", len(self.collectors[idx].views)))

    def _toggle_collect(self, on):
        idx = self.cb_cam.currentData()
        if idx is None:
            self.btn_collect.setChecked(False)
            self._say(t("cal.open"))
            return
        with self.detector.lock:
            if on:
                col = self.collectors.get(idx) or IntrinsicsCollector(self.sp_cols.value(), self.sp_rows.value(),
                                                                      self.sp_sq.value() / 1000.0)
                self.collectors[idx] = col
                self.detector.collect = (idx, col)
            else:
                self.detector.collect = None
        self.cb_cam.setEnabled(not on)

    def _solve_lens(self):
        idx = self.cb_cam.currentData()
        col = self.collectors.get(idx)
        if col is None:
            self._say(t("cal.open"))
            return
        self.btn_collect.setChecked(False)
        try:
            K, dist, rms, size = col.solve()
        except Exception as exc:
            self._say(t("cal.fail", exc))
            return
        self.intr[idx] = (K, dist, size)
        self._say(t("cal.rms", f"CAM {idx}", rms))

    def _capture_floor(self):
        with self.detector.lock:
            corners, sizes = list(self.detector.corners), list(self.detector.sizes)
        if len(self.cams) < 2:
            self._say(t("cal.need_two"))
            return
        if any(c is None for c in corners):
            self._say(t("cal.fail", t("cal.floor_hint")))
            return
        obs = []
        for i, (c, sz) in enumerate(zip(corners, sizes)):
            K, dist, size = self.intr.get(i) or (*default_intrinsics(sz), sz)
            obs.append((f"CAM {i}", c.copy(), K, dist, size))
        self.floor_obs = obs
        self.btn_rig.setEnabled(True)
        self._say(f"OK: {len(obs)} / {len(self.cams)}")

    def _solve_rig(self):
        try:
            self.rig = solve_extrinsics(self.floor_obs, self.sp_cols.value(), self.sp_rows.value(), self.sp_sq.value() / 1000.0)
        except Exception as exc:
            self._say(t("cal.fail", exc))
            return
        for name, rms in self.rig.rms.items():
            self._say(t("cal.rms", name, rms))
        self.btn_save.setEnabled(True)

    def _save(self):
        if self.rig is None:
            return
        path = self.rig_path
        if path is None:
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("cal.save"), "calibration.json", "JSON (*.json)")
            if not p:
                return
            path = Path(p)
        self.rig.save(path)
        self._say(t("cal.done", len(self.rig)))
        self.rig_saved.emit(str(path))

    def done(self, r):
        self.timer.stop()
        if self.detector:
            self.detector.running = False
        for s in self.cams:
            s.stop()
        super().done(r)
