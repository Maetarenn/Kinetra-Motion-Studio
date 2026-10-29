"""Light-weight custom widgets: bars, sparklines, the camera grid with skeleton / face overlays."""
import math
from collections import deque

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.constants import POSE_CONNECTIONS
from ..i18n import t
from .theme import C_BG, C_CYAN, C_AMBER, C_GREEN, C_DIM, C_PANEL2


class FastBar(QtWidgets.QWidget):
    """Cheap custom bar: 100+ styled QProgressBars repainting at 30-60 Hz cost more than the inference itself."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.v = 0.0
        self.setFixedHeight(10)
        self.setMinimumWidth(60)

    def set_value(self, v):
        if abs(v - self.v) > 0.004 or (v == 0.0) != (self.v == 0.0):
            self.v = v
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#0B0F17"))
        p.fillRect(QtCore.QRect(0, 0, int(self.width() * self.v), self.height()),
                   QtGui.QColor(C_CYAN if self.v > 0.6 else C_GREEN))


class BarList(QtWidgets.QScrollArea):
    """Scrollable list of (name, FastBar, value label) rows."""

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setSpacing(3)
        self.bars, self.vals = [], []
        for n in names:
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(n)
            lab.setMinimumWidth(128)
            bar, val = FastBar(), QtWidgets.QLabel("0.00")
            val.setMinimumWidth(32)
            row.addWidget(lab)
            row.addWidget(bar, 1)
            row.addWidget(val)
            lay.addLayout(row)
            self.bars.append(bar)
            self.vals.append(val)
        lay.addStretch(1)
        self.setWidget(inner)

    def set_values(self, arr):
        for i, (b, l) in enumerate(zip(self.bars, self.vals)):
            v = float(arr[i])
            b.set_value(v)
            l.setText(f"{v:.2f}")


class Sparkline(QtWidgets.QWidget):
    def __init__(self, color=C_CYAN, n=120, parent=None):
        super().__init__(parent)
        self.color, self.data = QtGui.QColor(color), deque(maxlen=n)
        self.setMinimumHeight(24)

    def push(self, v):
        self.data.append(float(v))
        self.update()

    def paintEvent(self, _):
        if len(self.data) < 2:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        lo, hi = min(self.data), max(self.data)
        span, w, h, n = max(hi - lo, 1e-6), self.width(), self.height(), len(self.data)
        pts = [QtCore.QPointF(i * w / (n - 1), h - 2 - (v - lo) / span * (h - 4)) for i, v in enumerate(self.data)]
        p.setPen(QtGui.QPen(self.color, 1.5))
        p.drawPolyline(pts)


class Meter(QtWidgets.QWidget):
    """Label + bar + value in one row (CPU / GPU / VRAM)."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(title)
        lab.setMinimumWidth(40)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.val = QtWidgets.QLabel("N/A")
        self.val.setMinimumWidth(70)
        self.val.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(lab)
        row.addWidget(self.bar, 1)
        row.addWidget(self.val)

    def set(self, frac, text):
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.val.setText(text)


class CameraGrid(QtWidgets.QWidget):
    """All camera images in a grid with 2D skeletons, face boxes and face landmarks painted on top."""

    def __init__(self, title_key="view.optical", parent=None):
        super().__init__(parent)
        self.title_key = title_key
        self.frames, self.bodies, self.faces, self.face_cam = [], [], [], 0
        self.only = None                      # show just this camera index (face view)
        self.show_body, self.show_face = True, True
        self.setMinimumSize(300, 200)

    def set_data(self, frames, bodies, faces, face_cam=0):
        self.frames, self.bodies, self.faces, self.face_cam = frames, bodies, faces, face_cam
        self.update()

    def clear(self):
        self.frames, self.bodies, self.faces = [], [], []
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(C_BG))
        font = p.font()
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(C_CYAN))
        p.drawText(10, 18, t(self.title_key))
        if not self.frames:
            p.setPen(QtGui.QColor(C_AMBER))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, t("nosignal"))
            return
        idxs = [self.only] if self.only is not None and self.only < len(self.frames) else list(range(len(self.frames)))
        n = len(idxs)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        top = 26
        cw, ch = self.width() / cols, (self.height() - top) / rows
        for k, i in enumerate(idxs):
            cell = QtCore.QRectF((k % cols) * cw, top + (k // cols) * ch, cw, ch).adjusted(2, 2, -2, -2)
            self._cell(p, cell, i)

    def _cell(self, p, cell, i):
        f = self.frames[i]
        fh, fw = f.shape[:2]
        img = QtGui.QImage(f.data, fw, fh, f.strides[0], QtGui.QImage.Format_BGR888)
        sc = min(cell.width() / fw, cell.height() / fh)
        dw, dh = fw * sc, fh * sc
        ox, oy = cell.x() + (cell.width() - dw) / 2, cell.y() + (cell.height() - dh) / 2
        p.drawImage(QtCore.QRectF(ox, oy, dw, dh), img)
        mx, my = (lambda x: ox + x * dw), (lambda y: oy + y * dh)
        cy, am = QtGui.QColor(C_CYAN), QtGui.QColor(C_AMBER)
        B = self.bodies[i] if self.show_body and i < len(self.bodies) else None
        if B is not None:
            for a, b in POSE_CONNECTIONS:
                ok = min(B[a, 3], B[b, 3]) > 0.5
                c = QtGui.QColor(cy if ok else am)
                c.setAlphaF(0.95 if ok else 0.4)
                p.setPen(QtGui.QPen(c, 2.2))
                p.drawLine(QtCore.QPointF(mx(B[a, 0]), my(B[a, 1])), QtCore.QPointF(mx(B[b, 0]), my(B[b, 1])))
            for j in range(11, 33):
                p.setPen(QtGui.QPen(cy if B[j, 3] > 0.5 else am, 6))
                p.drawPoint(QtCore.QPointF(mx(B[j, 0]), my(B[j, 1])))
        if self.show_face and i == self.face_cam:
            for tk in self.faces:
                x0, y0, x1, y1 = tk["bbox"]
                sx = dw / fw
                p.setPen(QtGui.QPen(cy, 1.5))
                p.drawRect(QtCore.QRectF(ox + x0 * sx, oy + y0 * sx, (x1 - x0) * sx, (y1 - y0) * sx))
                p.drawText(QtCore.QPointF(ox + x0 * sx, oy + y0 * sx - 4), f"FACE {tk['slot']}")
                p.setPen(QtGui.QPen(cy, 2))
                p.drawPoints(QtGui.QPolygonF([QtCore.QPointF(ox + x * sx, oy + y * sx) for x, y in tk["lm"][:468:3, :2]]))
                if tk["markers"] is not None:
                    p.setPen(QtGui.QPen(QtGui.QColor(C_GREEN), 3))
                    for x, y in tk["markers"]:
                        p.drawPoint(QtCore.QPointF(ox + x * sx, oy + y * sx))
        p.setPen(QtGui.QColor(C_DIM))
        p.drawText(QtCore.QPointF(cell.x() + 6, cell.bottom() - 6), f"CAM {i}")
