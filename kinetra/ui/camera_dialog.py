"""Visual Camera Setup & Multi-Camera / Phone Connection Manager Dialog.

Supports any number of cameras (1 to N) for studio-grade multi-view
triangulation.  Each camera can be a local device index or an IP stream URL.
"""
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from ..capture.cameras import get_camera_devices, open_capture
from ..i18n import t
from . import icons


class CameraPreviewWidget(QtWidgets.QLabel):
    """Small thumbnail widget displaying camera preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 190)
        self.setStyleSheet(
            "background: #080C14; border: 1px solid #1E293B; border-radius: 10px;"
        )
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(t("cam.preview_hint"))
        self._cap = None
        self._timer = QtCore.QTimer(self, timeout=self._tick, interval=33)

    def start_preview(self, source):
        self.stop_preview()
        try:
            src = int(source) if (isinstance(source, str) and source.isdigit()) else source
            self._cap, _ = open_capture(src, "640x480 @60")
            if self._cap and self._cap.isOpened():
                self._timer.start()
            else:
                self.setText(t("cam.open_fail"))
        except Exception as e:
            self.setText(f"{t('status.error')}: {e}")

    def stop_preview(self):
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.setText(t("cam.preview_stopped"))

    def _tick(self):
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_preview()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
_ROW_STYLE = """
    QWidget#camRow {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #162032, stop:1 #131D30);
        border: 1px solid #1E293B;
        border-radius: 8px;
    }
    QWidget#camRow:hover { border-color: #00F0FF; }
"""


def _role_labels():
    return [
        t("cam.role_main"), t("cam.role_side"), t("cam.role_top"),
        t("cam.role_back"), t("cam.role_detail"), t("cam.role_extra"),
    ]


class _CameraRow(QtWidgets.QWidget):
    """One camera entry inside the scroll-area."""
    remove_requested = QtCore.Signal(object)
    preview_requested = QtCore.Signal(object)

    def __init__(self, index: int, devices: list, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("camRow")
        self.setStyleSheet(_ROW_STYLE)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setPixmap(icons.camera().pixmap(18, 18))
        icon_lbl.setFixedWidth(22)
        icon_lbl.setStyleSheet("border: none;")
        row.addWidget(icon_lbl)

        self.lbl = QtWidgets.QLabel(f"{t('cam.camera')} {index + 1}")
        self.lbl.setStyleSheet("font-weight: bold; color: #00F0FF; min-width: 72px; border: none;")
        row.addWidget(self.lbl)

        self.cb_role = QtWidgets.QComboBox()
        self.cb_role.addItems(_role_labels() + [t("cam.role_custom")])
        self.cb_role.setCurrentIndex(min(index, len(_role_labels())))
        self.cb_role.setFixedWidth(150)
        row.addWidget(self.cb_role)

        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._populate(devices)
        row.addWidget(self.cb_device)

        self.ed_ip = QtWidgets.QLineEdit()
        self.ed_ip.setPlaceholderText("http://192.168.1.50:4747/video")
        self.ed_ip.setFixedWidth(230)
        self.ed_ip.hide()
        row.addWidget(self.ed_ip)

        self.chk_ip = QtWidgets.QCheckBox("IP")
        self.chk_ip.setToolTip(t("cam.ip_tooltip"))
        self.chk_ip.setStyleSheet("border: none;")
        self.chk_ip.toggled.connect(self._toggle_ip)
        row.addWidget(self.chk_ip)

        btn_preview = QtWidgets.QPushButton(icons.preview(), t("cam.preview"))
        btn_preview.setFixedWidth(85)
        btn_preview.setStyleSheet("border: none; padding: 4px 8px;")
        btn_preview.clicked.connect(lambda: self.preview_requested.emit(self))
        row.addWidget(btn_preview)

        self.btn_remove = QtWidgets.QPushButton(icons.remove(), t("cam.remove"))
        self.btn_remove.setFixedWidth(80)
        self.btn_remove.setStyleSheet("color: #F87171; border: none; padding: 4px 8px;")
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row.addWidget(self.btn_remove)

    def _populate(self, devices):
        self.cb_device.clear()
        for d in devices:
            self.cb_device.addItem(d["description"], d["index"])
        if self.index < self.cb_device.count():
            self.cb_device.setCurrentIndex(self.index)

    def _toggle_ip(self, on):
        self.ed_ip.setVisible(on)
        self.cb_device.setVisible(not on)

    def get_source(self) -> str:
        if self.chk_ip.isChecked():
            url = self.ed_ip.text().strip()
            return url if url else "0"
        return str(self.cb_device.currentData())

    def set_source(self, val_str: str, devices: list):
        if val_str.startswith("http://") or val_str.startswith("rtsp://"):
            self.chk_ip.setChecked(True)
            self.ed_ip.setText(val_str)
        elif val_str.isdigit():
            idx = int(val_str)
            for i in range(self.cb_device.count()):
                if self.cb_device.itemData(i) == idx:
                    self.cb_device.setCurrentIndex(i)
                    return


class CameraSetupDialog(QtWidgets.QDialog):
    """N-Camera Visual Setup Dialog."""

    def __init__(self, current_sources_str: str = "0", parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("cam.title"))
        self.resize(820, 600)
        self.setStyleSheet("""
            QDialog { background: #0E1726; color: #E2E8F0; }
            QComboBox, QLineEdit {
                background: #162032; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 10px; color: #FFFFFF;
                selection-background-color: #0284C7;
            }
            QComboBox:hover, QLineEdit:hover { border-color: #00F0FF; }
            QComboBox::drop-down { border: none; padding-right: 8px; }
            QPushButton {
                background: #1E293B; border: 1px solid #334155;
                border-radius: 6px; padding: 8px 16px; color: #E2E8F0; font-weight: bold;
            }
            QPushButton:hover { background: #0284C7; border-color: #38BDF8; color: white; }
            QPushButton:pressed { background: #0369A1; }
            QCheckBox { color: #CBD5E1; spacing: 6px; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 3px;
                border: 1px solid #475569; background: #1E293B;
            }
            QCheckBox::indicator:checked { background: #0284C7; border-color: #38BDF8; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #0E1726; width: 8px; border: none; }
            QScrollBar::handle:vertical { background: #334155; border-radius: 4px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #475569; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        self.current_sources_str = current_sources_str
        self.selected_sources: list[str] = []
        self.auto_sync_enabled = True
        self._rows: list[_CameraRow] = []
        self._devices: list[dict] = []
        self._build_ui()
        self._refresh_devices()
        self._restore_from_sources(current_sources_str)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        hdr = QtWidgets.QHBoxLayout()
        il = QtWidgets.QLabel()
        il.setPixmap(icons.camera().pixmap(28, 28))
        hdr.addWidget(il)
        hl = QtWidgets.QLabel(t("cam.title"))
        hl.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF; margin-left: 8px;")
        hdr.addWidget(hl)
        hdr.addStretch()
        root.addLayout(hdr)

        info = QtWidgets.QLabel(t("cam.info"))
        info.setStyleSheet("color: #94A3B8; font-size: 12px; padding: 4px 0;")
        info.setWordWrap(True)
        root.addWidget(info)

        cr = QtWidgets.QHBoxLayout()
        self.lbl_count = QtWidgets.QLabel(f"{t('cam.cameras')}: 0")
        self.lbl_count.setStyleSheet("color: #00F0FF; font-weight: bold; font-size: 13px;")
        cr.addWidget(self.lbl_count)
        cr.addStretch()
        ba = QtWidgets.QPushButton(icons.add(), t("cam.add"))
        ba.setStyleSheet("background: #0F766E; border-color: #14B8A6; padding: 6px 14px;")
        ba.clicked.connect(self._add_camera)
        cr.addWidget(ba)
        br = QtWidgets.QPushButton(icons.refresh(), t("cam.rescan"))
        br.setStyleSheet("padding: 6px 14px;")
        br.clicked.connect(self._refresh_devices)
        cr.addWidget(br)
        root.addLayout(cr)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setStyleSheet("background: transparent;")
        self.cam_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.cam_layout.setSpacing(6)
        self.cam_layout.setContentsMargins(4, 4, 4, 4)
        self.cam_layout.addStretch()
        self.scroll.setWidget(self.scroll_content)
        root.addWidget(self.scroll, 1)

        self.chk_autosync = QtWidgets.QCheckBox(t("cam.autosync"))
        self.chk_autosync.setChecked(True)
        self.chk_autosync.setStyleSheet("color: #38BDF8; font-weight: bold; margin-top: 4px;")
        root.addWidget(self.chk_autosync)

        pl = QtWidgets.QHBoxLayout()
        pl.addStretch()
        self.preview_box = CameraPreviewWidget(self)
        pl.addWidget(self.preview_box)
        pl.addStretch()
        root.addLayout(pl)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #1E293B;")
        root.addWidget(sep)

        bb = QtWidgets.QHBoxLayout()
        bc = QtWidgets.QPushButton(t("cam.cancel"))
        bc.setStyleSheet("padding: 8px 20px;")
        bc.clicked.connect(self.reject)
        bb.addWidget(bc)
        bb.addStretch()
        bap = QtWidgets.QPushButton(t("cam.apply"))
        bap.setStyleSheet("background: #0284C7; border: 1px solid #38BDF8; padding: 8px 24px; font-size: 13px;")
        bap.clicked.connect(self._on_apply)
        bb.addWidget(bap)
        root.addLayout(bb)

    def _refresh_devices(self):
        self.preview_box.stop_preview()
        self._devices = get_camera_devices()
        if not self._devices:
            self._devices = [{"index": 0, "name": t("cam.camera") + " 0",
                              "description": "[0] " + t("cam.camera") + " 0"}]
        for row in self._rows:
            row._populate(self._devices)

    def _add_camera(self, source_str: str | None = None):
        idx = len(self._rows)
        row = _CameraRow(idx, self._devices, self)
        row.remove_requested.connect(self._remove_camera)
        row.preview_requested.connect(self._preview_row)
        if idx == 0:
            row.btn_remove.setEnabled(False)
            row.btn_remove.hide()
        self._rows.append(row)
        self.cam_layout.insertWidget(self.cam_layout.count() - 1, row)
        if source_str:
            row.set_source(source_str, self._devices)
        self._update_labels()

    def _remove_camera(self, row: _CameraRow):
        if row in self._rows:
            self._rows.remove(row)
            self.cam_layout.removeWidget(row)
            row.deleteLater()
            self._update_labels()

    def _update_labels(self):
        for i, row in enumerate(self._rows):
            row.index = i
            row.lbl.setText(f"{t('cam.camera')} {i + 1}")
        n = len(self._rows)
        self.lbl_count.setText(f"{t('cam.cameras')}: {n}")
        self.chk_autosync.setVisible(n >= 2)

    def _preview_row(self, row: _CameraRow):
        self.preview_box.start_preview(row.get_source())

    def _restore_from_sources(self, sources_str: str):
        parts = [p.strip() for p in sources_str.split(",") if p.strip()]
        if not parts:
            parts = ["0"]
        for src in parts:
            self._add_camera(src)

    def _on_apply(self):
        self.preview_box.stop_preview()
        self.selected_sources = [row.get_source() for row in self._rows]
        if not self.selected_sources:
            self.selected_sources = ["0"]
        self.auto_sync_enabled = self.chk_autosync.isChecked()
        self.accept()

    def get_sources_string(self) -> str:
        return ",".join(self.selected_sources) if self.selected_sources else "0"

    def closeEvent(self, event):
        self.preview_box.stop_preview()
        super().closeEvent(event)
