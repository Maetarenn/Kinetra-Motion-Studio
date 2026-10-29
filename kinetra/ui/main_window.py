"""Kinetra main window: one workspace for body, face, single and multi camera capture."""
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtSvgWidgets, QtWidgets

from .. import APP_NAME, APP_VERSION, i18n
from ..capture.cameras import CAPTURE_PRESETS, scan_cameras
from ..capture.models import POSE_VARIANTS
from ..capture.pipeline import CaptureConfig, CaptureWorker
from ..core.constants import BLENDSHAPE_NAMES
from ..core.face import MUSCLE_NAMES
from ..core.facerig import FaceRig
from ..core.multicam import Rig
from ..core.project import Project, save_take, reprocess_take
from ..i18n import t
from .ai_dialog import AITrainerDialog
from .calibration import CalibrationDialog
from .scan_dialog import FaceScanDialog
from .face_view import FaceViewport, FaceScene
from .stage import StageViewport
from .export_dialog import TakeExportDialog
from .theme import C_AMBER, C_CYAN, C_GREEN, C_RED, C_DIM, C_TEXT, QSS
from .widgets import BarList, CameraGrid, Meter, Sparkline

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DATA_DIR = Path.home() / ".kinetra"


class HardwareProbe:
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self.psutil = None
        self.gpu_name, self._h, self.nv = None, None, None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            n = pynvml.nvmlDeviceGetName(self._h)
            self.gpu_name = n.decode() if isinstance(n, bytes) else n
        except Exception:
            self._h = None

    def read(self):
        cpu = self.psutil.cpu_percent(None) if self.psutil else 0.0
        gpu, used, total = None, 0.0, 0.0
        if self._h is not None:
            try:
                gpu = float(self.nv.nvmlDeviceGetUtilizationRates(self._h).gpu)
                m = self.nv.nvmlDeviceGetMemoryInfo(self._h)
                used, total = m.used / 2 ** 20, m.total / 2 ** 20
            except Exception:
                gpu = None
        return cpu, gpu, used, total


class _Job(QtCore.QThread):
    """Run a function off the UI thread; result or exception comes back as a signal."""
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(f"{APP_NAME} {t('app.subtitle')}")
        ico_path = ASSETS / 'icon.ico'
        png_path = ASSETS / 'icon.png'
        self.setWindowIcon(QtGui.QIcon(str(ico_path if ico_path.exists() else png_path)))
        self.resize(1720, 1000)
        self.worker, self.project, self.rig, self.profile, self.face_rig = None, None, None, None, None
        self._jobs, self._takes = [], []
        self._latest, self._fresh = None, False
        self._frames_sec, self._last_drops, self._drops_ps, self._n = 0, 0, 0, 0
        self._roll = {k: [] for k in ("a", "b", "t")}
        self._bindings = []
        self.facescene = FaceScene()
        self.probe = HardwareProbe()
        self._recording = False
        self._rec_shortcut = str(self.settings.value("rec_shortcut", "Space"))
        self._countdown_timer = QtCore.QTimer(self, timeout=self._countdown_tick, interval=1000)
        self._countdown_count = 0
        self._rec_start_time = 0.0
        self._build_menu()
        self._build_ui()
        self._load_settings()
        self.retranslate()
        self.log(t("log.ready"))
        if self.probe.gpu_name:
            self.log(f"CUDA: {self.probe.gpu_name}")
        self.ui_timer = QtCore.QTimer(self, timeout=self._tick_ui, interval=16)
        self.ui_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.ui_timer.start()
        self.stat_timer = QtCore.QTimer(self, timeout=self._tick_stats, interval=1000)
        self.stat_timer.start()

    # ================================================================================== i18n helpers
    def _bind(self, key, setter, *args):
        self._bindings.append((key, setter, args))
        setter(t(key, *args))

    def retranslate(self):
        for key, setter, args in self._bindings:
            setter(t(key, *args))
        self._on_state_text()
        self.setWindowTitle(f"{APP_NAME} {t('app.subtitle')}")
        for w in (self.cam_grid, self.face_grid, self.stage, self.view3d):
            w.update()

    def _label(self, key, *args):
        lab = QtWidgets.QLabel()
        self._bind(key, lab.setText, *args)
        return lab

    def _button(self, key, slot, checkable=False, name=None):
        b = QtWidgets.QPushButton()
        b.setCheckable(checkable)
        if name:
            b.setObjectName(name)
        (b.toggled if checkable else b.clicked).connect(slot)
        self._bind(key, b.setText)
        return b

    def _group(self, key):
        g = QtWidgets.QGroupBox()
        self._bind(key, g.setTitle)
        return g

    def _check(self, key, checked=False):
        c = QtWidgets.QCheckBox()
        c.setChecked(checked)
        self._bind(key, c.setText)
        return c

    def _combo_items(self, combo, keys):
        for i, k in enumerate(keys):
            combo.addItem("")
            self._bind(k, lambda s, i=i, c=combo: c.setItemText(i, s))

    # ======================================================================================= menus
    def _action(self, key, slot, shortcut=None):
        a = QtGui.QAction(self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(shortcut)
        self._bind(key, a.setText)
        return a

    def _menu(self, key, actions):
        m = self.menuBar().addMenu("")
        self._bind(key, m.setTitle)
        for a in actions:
            m.addSeparator() if a is None else m.addAction(a)
        return m

    def _build_menu(self):
        A = self._action
        self._menu("menu.file", [A("act.new_project", self.new_project, "Ctrl+N"),
                                 A("act.open_project", self.open_project, "Ctrl+O"),
                                 A("act.open_video", self.open_video), None, A("act.exit", self.close, "Ctrl+Q")])
        self.act_start, self.act_stop = A("act.start", self.start_capture, "F5"), A("act.stop", self.stop_capture, "F6")
        self.act_rec_toggle = A("act.record_toggle", self.trigger_record_with_countdown, self._rec_shortcut)
        self._menu("menu.capture", [self.act_start, self.act_stop, None, self.act_rec_toggle,
                                    A("act.set_rec_key", self.change_record_hotkey), None,
                                    A("act.calib_body", self.calibrate),
                                    A("act.calib_face", lambda: self._face_cmd("neutral")),
                                    A("act.rom", lambda: self._face_cmd("rom")),
                                    A("act.markers", lambda: self._face_cmd("reg"))])
        self.act_multi_actor = A("act.multi_actor", self.toggle_multi_actor)
        self.act_props_weapon = A("act.props_weapon", self.toggle_props_weapon)
        self.act_props_weapon.setCheckable(True)
        self.act_props_weapon.setChecked(True)
        self.act_audio_lipsync = A("act.audio_lipsync", self.toggle_audio_lipsync)
        self.act_audio_lipsync.setCheckable(True)
        self.act_audio_lipsync.setChecked(True)
        self.act_auto_sync = A("act.auto_sync_cams", self.auto_sync_cameras)
        self.act_multi_actor.setCheckable(True)
        self._menu("menu.tools", [A("act.cam_wizard", self.open_wizard), self.act_auto_sync, A("act.face_scan", self.open_face_scan),
                                  A("act.train_ai", self.open_ai_trainer), self.act_multi_actor, self.act_props_weapon, self.act_audio_lipsync,
                                  A("act.model", self.load_body_model),
                                  A("act.model_reset", self.reset_body_model),
                                  A("act.photo", self.photo_analysis),
                                  A("act.character", self.load_character), A("act.profile", self.load_profile),
                                  None, A("act.reprocess", self.reprocess_selected)])
        self.lang_menu = self.menuBar().addMenu("")
        self._bind("act.language", self.lang_menu.setTitle)
        grp = QtGui.QActionGroup(self)
        for code, name in i18n.available().items():
            a = QtGui.QAction(name, self, checkable=True)
            a.setData(code)
            a.triggered.connect(lambda _=False, c=code: self.set_language(c))
            grp.addAction(a)
            self.lang_menu.addAction(a)
        self._lang_actions = grp.actions()
        self._menu("menu.help", [A("act.about", self.about)])

    # ========================================================================================== UI
    def _build_ui(self):
        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.addWidget(self._build_header())
        self.main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_split.addWidget(self._build_left())
        self.main_split.addWidget(self._build_center())
        self.main_split.addWidget(self._build_right())
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([330, 1050, 340])
        outer.addWidget(self.main_split, 1)
        outer.addWidget(self._build_deck())
        self.setCentralWidget(root)

    def _build_header(self):
        bar = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(4, 0, 4, 0)
        logo = QtSvgWidgets.QSvgWidget(str(ASSETS / "icon.svg"))
        logo.setFixedSize(42, 42)
        title = QtWidgets.QLabel(APP_NAME.upper())
        title.setStyleSheet("font-size:22px; font-weight:bold; letter-spacing:5px;")
        sub = QtWidgets.QLabel()
        sub.setStyleSheet(f"color:{C_CYAN}; font-size:10px; letter-spacing:4px; font-weight:bold;")
        self._bind("app.subtitle", lambda s: sub.setText(s.upper()))
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(title)
        col.addWidget(sub)
        lay.addWidget(logo)
        lay.addLayout(col)
        lay.addStretch(1)
        self.lbl_project = QtWidgets.QLabel("")
        self.lbl_project.setStyleSheet(f"color:{C_DIM};")
        lay.addWidget(self.lbl_project)
        self.cb_lang = QtWidgets.QComboBox()
        for code, name in i18n.available().items():
            self.cb_lang.addItem(name, code)
        self.cb_lang.currentIndexChanged.connect(lambda _: self.set_language(self.cb_lang.currentData()))
        lay.addWidget(self.cb_lang)
        return bar

    def _build_left(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        g = self._group("panel.session")
        f = QtWidgets.QFormLayout(g)
        f.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.cb_mode = QtWidgets.QComboBox()
        self._combo_items(self.cb_mode, ("mode.single", "mode.multi"))
        self.cb_mode.currentIndexChanged.connect(self._mode_changed)
        self.ed_sources = QtWidgets.QLineEdit("0")
        self._bind("lbl.sources_hint", self.ed_sources.setPlaceholderText)
        btn_scan = self._button("btn.scan", self._scan)
        btn_scan.setStyleSheet("padding:3px 8px;")
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(self.ed_sources, 1)
        src_row.addWidget(btn_scan)
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItems(list(CAPTURE_PRESETS))
        self.cb_quality = QtWidgets.QComboBox()
        self.cb_quality.addItems(POSE_VARIANTS)
        self.cb_quality.setCurrentText("heavy")
        self.sp_faces = QtWidgets.QSpinBox()
        self.sp_faces.setRange(1, 4)
        self.chk_body, self.chk_face = self._check("chk.body", True), self._check("chk.face", True)
        self.chk_markers, self.chk_gpu = self._check("chk.markers"), self._check("chk.gpu")
        self.chk_video = self._check("chk.video")
        self.chk_hands, self.chk_ai = self._check("chk.hands", True), self._check("chk.ai", True)
        self.chk_multi_actor = self._check("act.multi_actor", False)
        self.chk_multi_actor.toggled.connect(self.toggle_multi_actor)
        self.chk_skel = self._check("chk.skeleton")
        self.chk_skel.toggled.connect(lambda on: setattr(self.stage, "show_rig", on) if hasattr(self, "stage") else None)
        self.lbl_facerig = QtWidgets.QLabel("-")
        self.chk_markers.toggled.connect(lambda on: self.worker and self.worker.set_markers(on))
        self.lbl_rig = QtWidgets.QLabel("-")
        for key, w in (("lbl.mode", self.cb_mode), ("lbl.sources", src_row), ("lbl.preset", self.cb_preset),
                       ("lbl.quality", self.cb_quality), ("lbl.faces", self.sp_faces), ("lbl.rig", self.lbl_rig),
                       ("lbl.facerig", self.lbl_facerig)):
            f.addRow(self._label(key), w)
        for c in (self.chk_body, self.chk_face, self.chk_hands, self.chk_ai, self.chk_markers, self.chk_gpu,
                  self.chk_video, self.chk_skel, self.chk_multi_actor):
            f.addRow(c)
        lay.addWidget(g)

        g = self._group("panel.params")
        f = QtWidgets.QFormLayout(g)
        self.sp_beta = self._dspin(0.0, 50.0, 0.5, 5.0)
        self.sp_fmin = self._dspin(0.1, 5.0, 0.1, 1.0)
        self.sp_focal = self._dspin(0.4, 1.6, 0.05, 0.9)
        for key, w in (("lbl.beta", self.sp_beta), ("lbl.fmin", self.sp_fmin), ("lbl.focal", self.sp_focal)):
            f.addRow(self._label(key), w)
            w.valueChanged.connect(self._apply_params)
        lay.addWidget(g)

        g = self._group("panel.stream")
        f = QtWidgets.QFormLayout(g)
        self.sp_port = QtWidgets.QSpinBox()
        self.sp_port.setRange(1024, 65535)
        self.sp_port.setValue(11111)
        self.cb_fmt = QtWidgets.QComboBox()
        self.cb_fmt.addItems(["ue5_livelink", "vmc", "json", "osc"])
        self.cb_fmt.currentIndexChanged.connect(self._on_stream_fmt_changed)
        f.addRow(self._label("lbl.stream_fmt"), self.cb_fmt)
        f.addRow(self._label("lbl.stream_port"), self.sp_port)
        lay.addWidget(g)

        g = self._group("panel.telemetry")
        v = QtWidgets.QVBoxLayout(g)
        self.m_cpu, self.m_gpu, self.m_vram = Meter("CPU"), Meter("GPU"), Meter("VRAM")
        for m in (self.m_cpu, self.m_gpu, self.m_vram):
            v.addWidget(m)
        self.lbl_fps = QtWidgets.QLabel("0.0 FPS")
        self.lbl_fps.setStyleSheet(f"color:{C_GREEN}; font-size:18px; font-weight:bold;")
        v.addWidget(self.lbl_fps)
        lay.addWidget(g)

        g = self._group("panel.latency")
        v = QtWidgets.QVBoxLayout(g)
        self.lat = {}
        for key, tkey, color in (("a", "lat.body", C_CYAN), ("b", "lat.face", C_CYAN), ("t", "lat.total", C_GREEN)):
            row = QtWidgets.QHBoxLayout()
            val = QtWidgets.QLabel("---")
            val.setStyleSheet(f"color:{color}; font-weight:bold; font-size:13px;")
            val.setAlignment(QtCore.Qt.AlignRight)
            row.addWidget(self._label(tkey))
            row.addWidget(val)
            v.addLayout(row)
            sp = Sparkline(color)
            v.addWidget(sp)
            self.lat[key] = (val, sp)
        lay.addWidget(g)

        g = self._group("panel.status")
        v = QtWidgets.QVBoxLayout(g)
        self.lbl_state, self.lbl_track = QtWidgets.QLabel(), QtWidgets.QLabel()
        self.lbl_lock, self.lbl_drop, self.lbl_sync = QtWidgets.QLabel(), QtWidgets.QLabel(), QtWidgets.QLabel()
        self.lbl_ai = QtWidgets.QLabel()
        for w in (self.lbl_state, self.lbl_track, self.lbl_lock, self.lbl_sync, self.lbl_ai, self.lbl_drop):
            v.addWidget(w)
        lay.addWidget(g)
        lay.addStretch(1)
        self._state = "status.standby"
        self._track_text = ("status.no_subject", C_AMBER, "")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(310)
        return scroll

    @staticmethod
    def _dspin(lo, hi, step, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _build_center(self):
        self.tabs = QtWidgets.QTabWidget()
        # capture tab
        self.stage_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.cam_grid, self.stage = CameraGrid("view.optical"), StageViewport()
        self.stage.maximize_requested.connect(self.toggle_maximize_stage)
        self.stage_split.addWidget(self.cam_grid)
        self.stage_split.addWidget(self.stage)
        self.stage_split.setSizes([420, 460])
        self.tabs.addTab(self.stage_split, "")
        # face tab
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        opts = QtWidgets.QHBoxLayout()
        self.cb_faceview = QtWidgets.QComboBox()
        self._combo_items(self.cb_faceview, ("view.mesh", "view.character", "view.both", "view.features"))
        self.chk_heat, self.chk_wire = self._check("chk.heat", True), self._check("chk.wire")
        opts.addWidget(self._label("lbl.mannequin"))
        for x in (self.cb_faceview, self.chk_heat, self.chk_wire):
            opts.addWidget(x)
        opts.addStretch(1)
        v.addLayout(opts)
        fs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.face_grid, self.view3d = CameraGrid("view.face"), FaceViewport()
        self.face_grid.show_body = False
        fs.addWidget(self.face_grid)
        fs.addWidget(self.view3d)
        fs.setSizes([420, 620])
        v.addWidget(fs, 1)
        self.chk_heat.toggled.connect(lambda on: setattr(self.view3d, "heat", on))
        self.chk_wire.toggled.connect(lambda on: setattr(self.view3d, "wire", on))
        self.tabs.addTab(w, "")
        # takes tab
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().hide()
        self._bind_headers()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("takes.open", self.open_take_folder))
        row.addWidget(self._button("takes.reprocess", self.reprocess_selected))
        row.addStretch(1)
        self.lbl_takes_empty = QtWidgets.QLabel()
        self._bind("takes.empty", self.lbl_takes_empty.setText)
        v.addWidget(self.tbl, 1)
        v.addWidget(self.lbl_takes_empty)
        v.addLayout(row)
        self.tabs.addTab(w, "")
        for i, k in enumerate(("tab.capture", "tab.face", "tab.takes")):
            self._bind(k, lambda s, i=i: self.tabs.setTabText(i, s))
        return self.tabs

    def _bind_headers(self):
        keys = ("takes.name", "takes.mode", "takes.frames", "takes.duration", "takes.processed")
        self._bindings.append(("", lambda _s: self.tbl.setHorizontalHeaderLabels([t(k) for k in keys]), ()))
        self.tbl.setHorizontalHeaderLabels([t(k) for k in keys])

    def _build_right(self):
        self.rtabs = QtWidgets.QTabWidget()
        from ..core.facerig import build_steps
        self.bs_list, self.mu_list = BarList(BLENDSHAPE_NAMES), BarList(MUSCLE_NAMES)
        self.pe_list = BarList([st.key for st in build_steps() if st.key != "center"])
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        for i, (w, k) in enumerate(((self.bs_list, "tab.blend"), (self.mu_list, "tab.muscles"), (self.pe_list, "tab.personal"),
                                    (self.report, "tab.report"))):
            self.rtabs.addTab(w, "")
            self._bind(k, lambda s, i=i: self.rtabs.setTabText(i, s))
        self.rtabs.setMinimumWidth(320)
        return self.rtabs

    def _build_deck(self):
        deck = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(deck)
        lay.setContentsMargins(0, 0, 0, 0)
        grid = QtWidgets.QGridLayout()
        self.btn_start = self._button("btn.start", self.start_capture, name="primary")
        self.btn_stop = self._button("btn.stop", self.stop_capture, name="abort")
        self.btn_calib = self._button("btn.calibrate", self.calibrate)
        self.btn_neutral = self._button("btn.neutral", lambda: self._face_cmd("neutral"))
        self.btn_rom = self._button("btn.rom", lambda: self._face_cmd("rom"))
        self.btn_markers = self._button("btn.markers", lambda: self._face_cmd("reg"))
        self.btn_rec = self._button("btn.record", self._toggle_record, checkable=True, name="rec")
        self.btn_stream = self._button("btn.stream", self._toggle_stream, checkable=True)
        for i, b in enumerate((self.btn_start, self.btn_stop, self.btn_calib, self.btn_neutral, self.btn_rom,
                               self.btn_markers, self.btn_rec, self.btn_stream)):
            b.setMinimumHeight(38)
            grid.addWidget(b, i // 4, i % 4)
        self._set_running(False)
        lay.addLayout(grid, 0)
        self.term = QtWidgets.QPlainTextEdit()
        self.term.setReadOnly(True)
        self.term.setMaximumBlockCount(800)
        self.term.setMinimumHeight(100)
        self.term.setMaximumHeight(130)
        lay.addWidget(self.term, 1)
        return deck

    # ============================================================================= settings / project
    def _load_settings(self):
        s = self.settings
        lang = s.value("language") or i18n.system_language()
        i18n.set_language(lang)
        self._sync_language_widgets()
        self.ed_sources.setText(s.value("sources", "0"))
        self.cb_preset.setCurrentText(s.value("preset", "1280x720 @60"))
        self.cb_quality.setCurrentText(s.value("quality", "heavy"))
        geo = s.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        self.stage.model_style = s.value("model_style", "bjd")
        mp = s.value("model_path")
        default_model = ASSETS / "models" / "default_character.glb"
        target_model = mp if (mp and Path(mp).exists()) else (str(default_model) if default_model.exists() else None)
        if target_model:
            QtCore.QTimer.singleShot(0, lambda: self.set_body_model(target_model))
        last = s.value("project")
        if last and (Path(last) / "project.json").exists():
            self._set_project(Project.open(last))
        elif s.value("default_project_dir"):
            def_p = Path(str(s.value("default_project_dir")))
            if def_p.exists():
                proj_p = def_p / "DefaultProject"
                if (proj_p / "project.json").exists():
                    self._set_project(Project.open(proj_p))
                else:
                    self._set_project(Project.create(proj_p, "DefaultProject"))

    def toggle_maximize_stage(self):
        if not hasattr(self, "_stage_maximized"):
            self._stage_maximized = False
            self._saved_main_sizes = [330, 1050, 340]
            self._saved_stage_sizes = [420, 460]

        total_w = sum(self.main_split.sizes())
        total_h = sum(self.stage_split.sizes())

        if not self._stage_maximized:
            self._saved_main_sizes = self.main_split.sizes()
            self._saved_stage_sizes = self.stage_split.sizes()
            start_main = list(self.main_split.sizes())
            target_main = [0, total_w, 0]
            start_stage = list(self.stage_split.sizes())
            target_stage = [0, total_h]
            self._stage_maximized = True
            if hasattr(self.stage, "btn_maximize"):
                self.stage.btn_maximize.is_maximized = True
                self.stage.btn_maximize.setToolTip("Görünümü Eski Boyuta Döndür (Restore - Esc)")
                self.stage.btn_maximize.update()
        else:
            start_main = list(self.main_split.sizes())
            target_main = self._saved_main_sizes if sum(self._saved_main_sizes) > 0 else [330, 1050, 340]
            start_stage = list(self.stage_split.sizes())
            target_stage = self._saved_stage_sizes if sum(self._saved_stage_sizes) > 0 else [420, 460]
            self._stage_maximized = False
            if hasattr(self.stage, "btn_maximize"):
                self.stage.btn_maximize.is_maximized = False
                self.stage.btn_maximize.setToolTip("Görünümü Büyüt / Tam Ekran (Maximize)")
                self.stage.btn_maximize.update()

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(260)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)

        def _step(val):
            m_sizes = [int(s0 + (s1 - s0) * val) for s0, s1 in zip(start_main, target_main)]
            s_sizes = [int(s0 + (s1 - s0) * val) for s0, s1 in zip(start_stage, target_stage)]
            self.main_split.setSizes(m_sizes)
            self.stage_split.setSizes(s_sizes)

        anim.valueChanged.connect(_step)
        anim.finished.connect(lambda: (self.main_split.setSizes(target_main), self.stage_split.setSizes(target_stage)))
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        self._max_anim = anim

    def toggle_multi_actor(self, on=None):
        if on is None:
            on = not getattr(self.stage, "multi_actor", False)
        self.stage.multi_actor = bool(on)
        self.act_multi_actor.setChecked(bool(on))
        if hasattr(self, "chk_multi_actor"):
            self.chk_multi_actor.blockSignals(True)
            self.chk_multi_actor.setChecked(bool(on))
            self.chk_multi_actor.blockSignals(False)
        state_str = "Açık (Active)" if on else "Kapalı (Inactive)"
        if self.worker is not None and hasattr(self.worker, "set_multi_actor"):
            self.worker.set_multi_actor(bool(on))
        self.log(f"Çoklu Oyuncu Modu (Multi-Actor Mocap): {state_str}")
        self.stage.update()

    def toggle_props_weapon(self, on=None):
        if on is None:
            on = self.act_props_weapon.isChecked()
        if self.worker is not None and hasattr(self.worker, 'set_prop_config'):
            self.worker.set_prop_config(enabled=bool(on))
        state_str = 'Açık (Active)' if on else 'Kapalı (Inactive)'
        self.log(f'Aksesuarlar & Kılıç / Silah Takibi: {state_str}')

    def toggle_audio_lipsync(self, on=None):
        if on is None:
            on = self.act_audio_lipsync.isChecked()
        if self.worker is not None and hasattr(self.worker, 'set_audio_lipsync'):
            self.worker.set_audio_lipsync(bool(on))
        state_str = 'Açık (Active)' if on else 'Kapalı (Inactive)'
        self.log(f'Ses Destekli Dudak Senkronizasyonu: {state_str}')

    def auto_sync_cameras(self):
        if self.worker is None or not self.worker.isRunning():
            self.log('Otomatik kamera eşleme için önce çekimi (F5) başlatmalısınız.')
            return
        if self.worker.cfg.mode != 'multi' or len(self.worker.sources) < 2:
            self.log('Otomatik eşleme için en az 2 kamera bağlı olmalıdır.')
            return
        sizes = [(s.cap.get(3), s.cap.get(4)) if hasattr(s, 'cap') else (1280, 720) for s in self.worker.sources]
        rig = self.worker.run_auto_sync_calibration(sizes)
        if rig is not None:
            self.rig = rig
            self.worker.cfg.rig = rig
            self.stage.set_rig(rig)
            self.log(f'Kameralar Başarıyla Otomatik Eşlendi ({len(rig)} Kamera).')
        else:
            self.log('Yeterli ortak vücut noktası toplanıyor, lütfen kameralar önünde hareket edin...')

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape and getattr(self, "_stage_maximized", False):
            self.toggle_maximize_stage()
            return
        super().keyPressEvent(e)

    def _sync_language_widgets(self):
        code = i18n.get_language()
        self.cb_lang.blockSignals(True)
        self.cb_lang.setCurrentIndex(max(self.cb_lang.findData(code), 0))
        self.cb_lang.blockSignals(False)
        for a in self._lang_actions:
            a.setChecked(a.data() == code)

    def set_language(self, code):
        i18n.set_language(code)
        self.settings.setValue("language", code)
        self._sync_language_widgets()
        self.retranslate()
        self._refresh_takes()

    def _set_project(self, proj):
        self.project = proj
        self.settings.setValue("project", str(proj.path))
        self.lbl_project.setText(f"{t('panel.project')}: {proj.meta['name']}")
        self.log(t("log.project_opened", proj.path))
        if proj.rig_path.exists():
            self._load_rig(proj.rig_path)
        fr = proj.path / "face_rig.json"
        if fr.exists():
            self._set_face_rig(FaceRig.load(fr))
        self._refresh_takes()

    def _set_face_rig(self, rig):
        self.face_rig = rig
        self.facescene.rig = rig
        self.lbl_facerig.setText(f"{len(rig.keys)} OK")
        self.log(t("log.face_rig_loaded", len(rig.keys)))

    def new_project(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, t("dlg.new_project"))
        if d:
            self._set_project(Project.create(d, Path(d).name))

    def open_project(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, t("dlg.open_project"))
        if d:
            try:
                self._set_project(Project.open(d))
            except FileNotFoundError:
                self._set_project(Project.create(d, Path(d).name))

    def _load_rig(self, path):
        try:
            self.rig = Rig.load(path)
            self.lbl_rig.setText(f"{len(self.rig)} x")
            self.log(t("log.rig_loaded", len(self.rig)))
        except Exception as exc:
            self.log(t("cal.fail", exc))

    def _refresh_takes(self):
        takes = self.project.takes() if self.project else []
        self.tbl.setRowCount(len(takes))
        for r, m in enumerate(takes):
            for c, v in enumerate((m["name"], m.get("mode", ""), m.get("frames", ""), f"{m.get('duration_s', 0):.1f}",
                                   m.get("processed", ""))):
                self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.lbl_takes_empty.setVisible(not takes)
        self._takes = takes

    def open_take_folder(self):
        r = self.tbl.currentRow()
        if 0 <= r < len(self._takes):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._takes[r]["dir"]))

    def reprocess_selected(self):
        r = self.tbl.currentRow()
        if not (0 <= r < len(getattr(self, "_takes", []))):
            return
        d = self._takes[r]["dir"]
        self.log(t("log.reprocessing", Path(d).name))
        job = _Job(lambda: (Path(d).name, reprocess_take(d, beta=self.sp_beta.value() / 2.5, f_min=self.sp_fmin.value())))
        job.done.connect(lambda res: (self.log(t("log.reprocess_done", *res)), self._refresh_takes()))
        job.failed.connect(lambda m: self.log(t("log.reprocess_fail", m)))
        self._start_job(job)

    def _start_job(self, job):
        self._jobs.append(job)
        job.finished.connect(lambda j=job: self._jobs.remove(j) if j in self._jobs else None)
        job.start()

    # ================================================================================== capture control
    def _parse_sources(self):
        out = []
        for tok in self.ed_sources.text().split(","):
            tok = tok.strip().strip('"')
            if tok:
                out.append(int(tok) if tok.isdigit() else tok)
        return out or [0]

    def _mode_changed(self):
        multi = self.cb_mode.currentIndex() == 1
        if multi and self.ed_sources.text().strip() in ("0", ""):
            self.ed_sources.setText("0,1")

    def _scan(self):
        found = scan_cameras()
        self.ed_sources.setText(",".join(map(str, found)) if found else "0")
        self.cb_mode.setCurrentIndex(1 if len(found) > 1 else 0)

    def start_capture(self):
        if self.worker is not None and self.worker.isRunning():
            return
        sources = self._parse_sources()
        multi = self.cb_mode.currentIndex() == 1
        if multi:
            if len(sources) < 2:
                self.log(t("log.few_sources"))
                return
            if self.rig is None or len(self.rig) != len(sources):
                self.log(t("log.need_rig"))
                return
        is_multi_actor = getattr(self.stage, "multi_actor", False)
        cfg = CaptureConfig(sources=sources, mode="multi" if multi else "single", preset=self.cb_preset.currentText(),
                            pose_variant=self.cb_quality.currentText(), num_faces=self.sp_faces.value(),
                            num_poses=2 if is_multi_actor else 1, multi_actor=is_multi_actor,
                            use_gpu=self.chk_gpu.isChecked(), track_body=self.chk_body.isChecked(),
                            track_face=self.chk_face.isChecked(), track_hands=self.chk_hands.isChecked(),
                            use_ai=self.chk_ai.isChecked(), face_rig=self.face_rig, markers=self.chk_markers.isChecked(),
                            rig=self.rig if multi else None, focal_factor=self.sp_focal.value(),
                            beta=self.sp_beta.value(), f_min=self.sp_fmin.value())
        self.worker = CaptureWorker(cfg)
        if self.profile:
            self.worker.set_profile_bias(self.profile["bias"])
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.message.connect(self._on_message)
        self.worker.state.connect(self._on_state)
        self.worker.finished.connect(self._on_finished)
        self.stage.set_rig(self.rig if multi else None)
        for k, v in (("sources", self.ed_sources.text()), ("preset", cfg.preset), ("quality", cfg.pose_variant)):
            self.settings.setValue(k, v)
        self._last_drops = 0
        self.btn_start.setEnabled(False)
        self.act_start.setEnabled(False)
        self.worker.start()

    def stop_capture(self):
        if self.worker and self.worker.isRunning():
            self.btn_stop.setEnabled(False)
            if self._recording:
                self.btn_rec.setChecked(False)
            self.worker.stop()

    def calibrate(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_calibration()

    def _face_cmd(self, what):
        if not (self.worker and self.worker.isRunning()):
            return
        if what == "reg" and not self.chk_markers.isChecked():
            self.chk_markers.setChecked(True)
        self.worker.request(what)
        self.log(t({"neutral": "hint.neutral", "rom": "hint.rom", "reg": "hint.markers"}[what]))

    def _apply_params(self, *_):
        if self.worker:
            self.worker.set_params(self.sp_beta.value(), self.sp_fmin.value(), self.sp_focal.value())

    def _on_stream_fmt_changed(self, _):
        fmt = self.cb_fmt.currentText()
        if fmt == 'ue5_livelink':
            self.sp_port.setValue(11111)
        elif fmt == 'vmc':
            self.sp_port.setValue(39539)
        else:
            self.sp_port.setValue(9000)

    def _toggle_stream(self, on):
        if self.worker:
            sm = self.worker.streamer
            sm.configure("127.0.0.1", self.sp_port.value(), self.cb_fmt.currentText())
            sm.enabled = on
        self.log(t("log.stream_on", "127.0.0.1", self.sp_port.value(), self.cb_fmt.currentText()) if on
                 else t("log.stream_off"))

    def change_record_hotkey(self):
        key, ok = QtWidgets.QInputDialog.getText(
            self, t("dlg.set_key_title"),
            f"{t('dlg.set_key_prompt')}\n(Mevcut: {self._rec_shortcut})",
            text=self._rec_shortcut
        )
        if ok and key.strip():
            key = key.strip()
            self._rec_shortcut = key
            self.settings.setValue("rec_shortcut", key)
            self.act_rec_toggle.setShortcut(key)
            self.btn_rec.setText(f"{t('btn.record')} ({key})")
            self.log(f"Kayıt kısayolu güncellendi: {key}")

    def trigger_record_with_countdown(self):
        if self._recording:
            self._finish_recording()
            return
        if self._countdown_count > 0:
            self._countdown_timer.stop()
            self._countdown_count = 0
            self.stage.countdown_text = None
            self.stage.update()
            self.btn_rec.setText(f"{t('btn.record')} ({self._rec_shortcut})")
            self.btn_rec.setStyleSheet("")
            self.btn_rec.setChecked(False)
            self.log("Geri sayım iptal edildi.")
            return

        if self.worker is None or not self.worker.isRunning():
            self.start_capture()

        if self.project is None:
            def_setting = self.settings.value("default_project_dir")
            default_dir = (Path(str(def_setting)) / "DefaultProject") if def_setting else (Path.home() / "Kinetra_Captures")
            self.project = Project.create(default_dir, "DefaultProject")
            self.lbl_project.setText(self.project.meta["name"])
            self._refresh_takes()

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Kayıt Kapsamı / Recording Scope")
            box.setText("Yüz ve vücut animasyonları hareketleri birlikte kaydedilsin mi?")
            active_mode_name = "Yüz Animasyonu" if self.tabs.currentIndex() == 1 else "Vücut Hareketi"
            box.setInformativeText(f"Evet: Hem yüz hem vücut aynı anda kaydedilir.\nHayır: Sadece şu an aktif olan sekme kaydedilir ({active_mode_name}).")
            box.setIcon(QtWidgets.QMessageBox.Question)
            btn_yes = box.addButton("Evet (Birlikte Kaydet)", QtWidgets.QMessageBox.YesRole)
            btn_no = box.addButton("Hayır (Sadece Aktif Sekme)", QtWidgets.QMessageBox.NoRole)
            btn_cancel = box.addButton("İptal", QtWidgets.QMessageBox.RejectRole)
            box.setStyleSheet("""
                QMessageBox { background-color: #121824; color: #E2E8F0; }
                QLabel { color: #E2E8F0; font-size: 13px; }
                QPushButton { background-color: #1E293B; color: #00E5FF; border: 1px solid #00E5FF; border-radius: 5px; padding: 6px 14px; font-weight: bold; }
                QPushButton:hover { background-color: #00E5FF; color: #0B0E14; }
            """)
            box.exec()
            clicked = box.clickedButton()
            if clicked == btn_cancel:
                self.btn_rec.setChecked(False)
                return
            elif clicked == btn_yes:
                self._rec_scope = "both"
                self.log("Kayıt kapsamı: Yüz ve Vücut Birlikte (Full Sync)")
            else:
                self._rec_scope = "face" if self.tabs.currentIndex() == 1 else "body"
                scope_str = "Sadece Yüz Animasyonu" if self._rec_scope == "face" else "Sadece Vücut Hareketi"
                self.log(f"Kayıt kapsamı: {scope_str}")
        else:
            self._rec_scope = "both"

        if os.environ.get("PYTEST_CURRENT_TEST"):
            self._start_actual_recording()
            return

        self._countdown_count = 3
        self.stage.countdown_text = "3"
        self.stage.update()
        self.btn_rec.setText("Geri Sayım: 3")
        self.btn_rec.setStyleSheet("background-color: #E65100; color: white; font-weight: bold;")
        self._countdown_timer.start(1000)

    def _start_actual_recording(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.recorder.start()
            if self.chk_video.isChecked() and self.project:
                tmp = self.project.path / "takes" / "_video_tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True, exist_ok=True)
                self.worker.start_video(tmp)
            self._recording = True
            self._rec_start_time = time.time()
            self.stage.is_recording = True
            self.stage.rec_time_str = "00:00.0"
            self.btn_rec.setChecked(True)
            self.btn_rec.setText("[REC] KAYDI DURDUR")
            self.btn_rec.setStyleSheet("background-color: #D50000; color: white; font-weight: bold;")
            self.log("Kayıt başladı (Recording active)!")

    def _countdown_tick(self):
        self._countdown_count -= 1
        if self._countdown_count > 0:
            self.stage.countdown_text = str(self._countdown_count)
            self.btn_rec.setText(f"Geri Sayım: {self._countdown_count}")
            self.stage.update()
        elif self._countdown_count == 0:
            self.stage.countdown_text = "REC!"
            self.stage.update()
            self._start_actual_recording()
        else:
            self._countdown_timer.stop()
            self._countdown_count = 0
            self.stage.countdown_text = None
            self.stage.update()

    def _toggle_record(self, on):
        if on:
            if not self._recording and self._countdown_count == 0:
                self.trigger_record_with_countdown()
        else:
            if self._recording or self._countdown_count > 0:
                self.trigger_record_with_countdown()

    def _finish_recording(self):
        if not self._recording or self.worker is None:
            return
        self._recording = False
        self.stage.is_recording = False
        self.stage.countdown_text = None
        self.stage.update()
        self.btn_rec.setChecked(False)
        self.btn_rec.setText(f"{t('btn.record')} ({self._rec_shortcut})")
        self.btn_rec.setStyleSheet("")
        self.worker.stop_video()
        data = self.worker.recorder.stop()
        if hasattr(self, "_rec_scope") and data:
            if self._rec_scope == "body":
                for fr in data.get("frames", []):
                    fr.pop("faces", None)
                    fr.pop("face_mesh", None)
            elif self._rec_scope == "face":
                for fr in data.get("frames", []):
                    fr.pop("joints", None)
                    fr.pop("quats", None)
                    fr.pop("quats_full", None)
        cfg = self.worker.cfg
        settings = {"beta": self.sp_beta.value(), "f_min": self.sp_fmin.value(), "sources": [str(s) for s in cfg.sources]}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        d = None
        try:
            if self.project:
                d = save_take(self.project, data, mode=cfg.mode, settings=settings, face_rig=self.face_rig)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        tmp = self.project.path / "takes" / "_video_tmp" if self.project else None
        if d is not None and tmp and tmp.exists() and any(tmp.iterdir()):
            shutil.move(str(tmp), str(d / "video"))
        elif tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        self.log(t("log.take_saved", d.name) if d else t("log.take_empty"))
        self._refresh_takes()
        if d is not None and not os.environ.get("PYTEST_CURRENT_TEST"):
            dlg = TakeExportDialog(d, self)
            dlg.exec()

    # ============================================================================== worker callbacks
    def _on_message(self, key, args):
        if key == "log.face_event":
            code, tid, val = args
            self.log(t("fe." + code, tid, val))
        elif key == "log.raw":
            self.log(str(args[0]))
        else:
            self.log(t(key, *args))

    def _on_state(self, s):
        self._set_running(s == "RUNNING")
        self._state = {"RUNNING": "status.running", "ERROR": "status.error", "STOPPED": "status.stopped"}.get(s, "status.standby")
        self._on_state_text()

    def _on_state_text(self):
        color = {"status.running": C_GREEN, "status.error": C_AMBER}.get(self._state, C_TEXT)
        self.lbl_state.setText(f"{t('status.engine')}  {t(self._state)}")
        self.lbl_state.setStyleSheet(f"color:{color}; font-weight:bold;")
        key, col, extra = self._track_text
        self.lbl_track.setText(f"{t('status.tracking')}  {t(key)}{extra}")
        self.lbl_track.setStyleSheet(f"color:{col}; font-weight:bold;")

    def _set_running(self, run):
        for b in (self.btn_stop, self.btn_calib, self.btn_neutral, self.btn_rom, self.btn_markers, self.btn_stream):
            b.setEnabled(run)
        self.btn_rec.setEnabled(True)
        self.btn_start.setEnabled(not run)
        self.act_start.setEnabled(not run)
        self.act_stop.setEnabled(run)

    def _on_finished(self):
        if self._recording:
            self._finish_recording()
        if self._countdown_count > 0:
            self._countdown_timer.stop()
            self._countdown_count = 0
        self.stage.countdown_text = None
        self.stage.is_recording = False
        self._set_running(False)
        self.btn_rec.setText(f"{t('btn.record')} ({self._rec_shortcut})")
        self.btn_rec.setStyleSheet("")
        self.btn_stream.setChecked(False)
        self._latest, self._fresh = None, False
        self.cam_grid.clear()
        self.face_grid.clear()
        self.stage.clear()
        self.view3d.set_items([])
        self.log(t("log.stopped"))
        self._state = "status.stopped"
        self._on_state_text()

    def _on_frame(self, res):
        self._latest, self._fresh = res, True
        self._frames_sec += 1

    # ======================================================================================== timers
    def _tick_ui(self):
        if self._recording and hasattr(self, "_rec_start_time"):
            elapsed = time.time() - self._rec_start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            frac = int((elapsed - int(elapsed)) * 10)
            time_str = f"{mins:02d}:{secs:02d}.{frac}"
            self.stage.rec_time_str = time_str
            self.btn_rec.setText(f"[REC] {time_str}")

        if not self._fresh or self._latest is None:
            return
        res, self._fresh = self._latest, False
        self._n += 1
        tab = self.tabs.currentIndex()
        fc = min(self.worker.cfg.face_cam, len(res["frames"]) - 1) if self.worker else 0
        if tab == 0:
            self.cam_grid.set_data(res["frames"], res["body_img"], res["faces"], fc)
            if getattr(self.stage, "multi_actor", False):
                if len(res.get("faces", [])) > 1:
                    if res.get("joints") is not None:
                        sec_j = res["joints"].copy()
                        sec_j[:, 0] += 1.3
                        self.stage.secondary_joints = sec_j
                elif res.get("secondary_joints") is not None:
                    self.stage.secondary_joints = res["secondary_joints"]
                else:
                    self.stage.secondary_joints = None
            else:
                self.stage.secondary_joints = None
            self.stage.update_result(res)
        elif tab == 1:
            self.face_grid.only = fc
            self.face_grid.set_data(res["frames"], [None] * len(res["frames"]), res["faces"], fc)
            faces = res["faces"]
            self.view3d.set_items(self.facescene.items(faces, self.cb_faceview.currentIndex()))
        if self._n % 2 == 0 and res["faces"]:
            sel = res["faces"][0]
            if self.rtabs.currentIndex() == 0:
                self.bs_list.set_values(sel["bs"])
            elif self.rtabs.currentIndex() == 1:
                self.mu_list.set_values(sel["muscles"])
            elif self.rtabs.currentIndex() == 2 and sel.get("personal") is not None:
                self.pe_list.set_values(sel["personal"])
        for key, field in (("a", "a_us"), ("b", "b_us"), ("t", "total_us")):
            r = self._roll[key]
            r.append(res[field])
            del r[:-30]
            val, sp = self.lat[key]
            val.setText(f"{sum(r) / len(r) / 1000:.1f} ms")
            sp.push(res[field] / 1000)
        if res["joints"] is None:
            self._track_text = ("status.no_subject", C_AMBER, "")
        else:
            low = res["vis_mean"] < 0.6 or res["foot_vis"] < 0.5
            extra = f"  {res['vis_mean'] * 100:.0f}%" + (f"  x{res['tri_used']:.1f}" if res["mode"] == "multi" else "")
            self._track_text = ("status.low" if low else "status.nominal", C_AMBER if low else C_GREEN, extra)
            self.lbl_lock.setText(f"{t('status.locks')}  " + " ".join(s if l else "--" for s, l in zip("LR", res["locks"])))
        hs = res.get("hands", {})
        ai_txt = " ".join(f"{k} {h['ai']['confidence'] * 100:.0f}%" for k, h in hs.items() if h.get("ai"))
        self.lbl_ai.setText(f"{t('status.ai')}  {ai_txt}" + (f"  [{len(res['limit_hits'])}]" if res.get("limit_hits") else ""))
        self.lbl_sync.setText(f"{t('status.sync')}  {res['sync_us'] / 1000:.1f} ms   {t('status.faces')} {len(res['faces'])}")
        self._on_state_text()
        self._drops_now = res["drops"]

    def _tick_stats(self):
        fps, self._frames_sec = self._frames_sec, 0
        self.lbl_fps.setText(f"{fps:.1f} FPS")
        self.lbl_fps.setStyleSheet(f"color:{C_GREEN if fps >= 24 else C_AMBER}; font-size:18px; font-weight:bold;")
        total = getattr(self, "_drops_now", 0)
        self._drops_ps, self._last_drops = max(total - self._last_drops, 0), total
        self.lbl_drop.setText(f"{t('status.drops')}  {self._drops_ps}/s  ({total})")
        self.lbl_drop.setStyleSheet(f"color:{C_AMBER if self._drops_ps > 2 else C_TEXT};")
        cpu, gpu, used, vt = self.probe.read()
        self.m_cpu.set(cpu / 100, f"{cpu:.0f}%")
        if gpu is None:
            self.m_gpu.set(0, "N/A")
            self.m_vram.set(0, "N/A")
        else:
            self.m_gpu.set(gpu / 100, f"{gpu:.0f}%")
            self.m_vram.set(used / max(vt, 1), f"{used:,.0f}MB")

    # ==================================================================================== tools
    def open_wizard(self):
        if self.worker and self.worker.isRunning():
            self.stop_capture()
        rig_path = self.project.rig_path if self.project else None
        dlg = CalibrationDialog(self._parse_sources(), self.cb_preset.currentText(), rig_path, self)
        dlg.rig_saved.connect(lambda p: (self._load_rig(p), self.cb_mode.setCurrentIndex(1)))
        dlg.exec()

    def open_face_scan(self):
        if self.worker and self.worker.isRunning():
            self.stop_capture()
        path = (self.project.path if self.project else DATA_DIR) / "face_rig.json"
        dlg = FaceScanDialog(self._parse_sources()[0], self.cb_preset.currentText(), path, self)
        dlg.rig_ready.connect(self._set_face_rig)
        dlg.exec()

    def open_ai_trainer(self):
        dlg = AITrainerDialog(str(self.project.path) if self.project else None, self)
        dlg.exec()

    def open_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("dlg.video"), "", "Video (*.mp4 *.mov *.avi *.mkv *.m4v)")
        if p:
            self.ed_sources.setText(p)
            self.cb_mode.setCurrentIndex(0)

    def photo_analysis(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, t("dlg.photos"), "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not paths:
            return
        from ..capture.photo import build_profile
        out = (self.project.path if self.project else DATA_DIR) / "profiles"
        self.log(t("log.photos", len(paths)))
        job = _Job(lambda: build_profile(paths, out, lambda m: None))
        job.done.connect(self._profile_ready_tuple)
        job.failed.connect(lambda m: self.log(t("log.photo_fail", m)))
        self._start_job(job)

    def _profile_ready_tuple(self, res):
        prof, report = res
        self.report.setPlainText(report)
        self.rtabs.setCurrentIndex(3)
        self._use_profile(prof)

    def _use_profile(self, prof):
        self.profile = prof
        self.facescene.identity = prof.get("identity_L")
        if self.worker:
            self.worker.set_profile_bias(prof["bias"])
        self.log(t("log.profile_ready", prof["name"]))

    def load_profile(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("dlg.profile"), str(DATA_DIR / "profiles"), "JSON (*.json)")
        if p:
            try:
                prof = json.loads(Path(p).read_text())
                self._use_profile(prof)
                self.report.setPlainText(json.dumps(prof["metrics"], indent=2))
            except Exception as exc:
                self.log(t("log.photo_fail", exc))

    def load_body_model(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("dlg.model"), self.settings.value("model_dir", ""),
                                                     "3D Model (*.glb *.vrm *.gltf *.obj);;glTF (*.glb *.gltf);;Wavefront OBJ (*.obj);;VRM (*.vrm);;All files (*.*)")
        if not p:
            return
        self.set_body_model(p)

    def set_body_model(self, path):
        try:
            from ..character_skinned import SkinnedCharacter
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                ch = SkinnedCharacter(path)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as exc:
            self.log(t("log.model_fail", exc))
            return False
        self.stage.set_character(ch)
        self.settings.setValue("model_path", str(path))
        self.settings.setValue("model_dir", str(Path(path).parent))
        self.log(t("log.model_loaded", ch.name, len(ch.bone_map), ch.V))
        return True

    def set_model_style(self, style):
        self.stage.model_style = style
        self.stage.set_character(None)
        self.settings.setValue("model_style", style)
        self.settings.remove("model_path")

    def reset_body_model(self):
        default_model = ASSETS / "models" / "default_character.glb"
        if default_model.exists():
            self.set_body_model(str(default_model))
            self.settings.setValue("model_path", str(default_model))
        else:
            self.stage.set_character(None)
            self.settings.remove("model_path")

    def load_character(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("dlg.glb"), "", "glTF binary (*.glb)")
        if not p:
            return
        try:
            from ..character import MorphCharacter
            ch = MorphCharacter(p)
        except Exception as exc:
            self.log(t("log.char_fail", exc))
            return
        self.facescene.character = ch
        self.log(t("log.char_loaded", ch.name, ch.V, ch.matched))
        self.cb_faceview.setCurrentIndex(1)

    def about(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(t("act.about"))
        box.setIconPixmap(QtGui.QPixmap(str(ASSETS / "icon.png")).scaled(96, 96, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        box.setText(t("about.text", APP_VERSION))
        box.exec()

    # ======================================================================================== misc
    def log(self, msg):
        self.term.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        if self.worker and self.worker.isRunning():
            self.stop_capture()
        for j in list(self._jobs):
            j.wait(2000)
        super().closeEvent(e)
