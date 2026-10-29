"""3D virtual stage (OpenGL fixed pipeline for broad driver compatibility): infinite grid, camera frustums,
the android mannequin and a Blender-style bone overlay."""
import math
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets
from OpenGL import GL as gl
from OpenGL import GLU as glu

from ..core.mannequin import make_model
from ..core.constants import HIP_MID, SHO_MID, EAR_MID, LEGS
from ..core.mathx import unit, q_from_vectors, q_mul, q_axis_angle, q_rotate
from ..core.physics_props import PhysicsWorld
from ..i18n import t
from .theme import C_BG, C_CYAN, C_AMBER, C_GRID


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


@dataclass
class StuntProp:
    id: int
    kind: str             # "cube", "box", "cylinder", "ramp", "platform", "sphere", "step"
    pos: np.ndarray       # shape (3,) [x, y, z]
    yaw: float = 0.0      # rotation in degrees
    size: tuple = (1.0, 1.0, 1.0)
    color: tuple = (0.2, 0.5, 0.8, 1.0)
    selected: bool = False


class StageOverlayButton(QtWidgets.QPushButton):
    def __init__(self, kind="props", parent=None):
        super().__init__(parent)
        self.kind = kind
        self.is_maximized = False
        self.setFixedSize(38, 38)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        hover = self.underMouse()
        bg_col = QtGui.QColor(24, 32, 47, 235) if hover else QtGui.QColor(15, 22, 35, 210)
        border_col = QtGui.QColor("#00E5FF") if hover else QtGui.QColor(51, 65, 85, 200)
        p.setBrush(QtGui.QBrush(bg_col))
        p.setPen(QtGui.QPen(border_col, 1.2))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

        cx, cy = self.width() / 2.0, self.height() / 2.0

        if self.kind == "maximize":
            pen = QtGui.QPen(QtGui.QColor("#00E5FF" if hover else "#38BDF8"), 2.0)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            s = 6.0
            if not self.is_maximized:
                d = 8.5
                p.drawLine(QtCore.QPointF(cx - d, cy - d + s), QtCore.QPointF(cx - d, cy - d))
                p.drawLine(QtCore.QPointF(cx - d, cy - d), QtCore.QPointF(cx - d + s, cy - d))
                p.drawLine(QtCore.QPointF(cx + d - s, cy - d), QtCore.QPointF(cx + d, cy - d))
                p.drawLine(QtCore.QPointF(cx + d, cy - d), QtCore.QPointF(cx + d, cy - d + s))
                p.drawLine(QtCore.QPointF(cx - d, cy + d - s), QtCore.QPointF(cx - d, cy + d))
                p.drawLine(QtCore.QPointF(cx - d, cy + d), QtCore.QPointF(cx - d + s, cy + d))
                p.drawLine(QtCore.QPointF(cx + d - s, cy + d), QtCore.QPointF(cx + d, cy + d))
                p.drawLine(QtCore.QPointF(cx + d, cy + d), QtCore.QPointF(cx + d, cy + d - s))
            else:
                d = 4.0
                p.drawLine(QtCore.QPointF(cx - d - s, cy - d), QtCore.QPointF(cx - d, cy - d))
                p.drawLine(QtCore.QPointF(cx - d, cy - d), QtCore.QPointF(cx - d, cy - d - s))
                p.drawLine(QtCore.QPointF(cx + d + s, cy - d), QtCore.QPointF(cx + d, cy - d))
                p.drawLine(QtCore.QPointF(cx + d, cy - d), QtCore.QPointF(cx + d, cy - d - s))
                p.drawLine(QtCore.QPointF(cx - d - s, cy + d), QtCore.QPointF(cx - d, cy + d))
                p.drawLine(QtCore.QPointF(cx - d, cy + d), QtCore.QPointF(cx - d, cy + d + s))
                p.drawLine(QtCore.QPointF(cx + d + s, cy + d), QtCore.QPointF(cx + d, cy + d))
                p.drawLine(QtCore.QPointF(cx + d, cy + d), QtCore.QPointF(cx + d, cy + d + s))

        elif self.kind == "props":
            # 3D Geometric Badge: Golden apex pyramid/triangle on top + Isometric cube below
            tri = QtGui.QPolygonF([
                QtCore.QPointF(cx, cy - 10.0),
                QtCore.QPointF(cx - 7.5, cy - 2.0),
                QtCore.QPointF(cx + 7.5, cy - 2.0),
            ])
            p.setPen(QtGui.QPen(QtGui.QColor("#F59E0B" if not hover else "#FBBF24"), 1.2))
            p.setBrush(QtGui.QColor(245, 158, 11, 220 if hover else 170))
            p.drawPolygon(tri)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1.0))
            p.drawLine(QtCore.QPointF(cx, cy - 10.0), QtCore.QPointF(cx, cy - 2.0))

            cube_top = QtGui.QPolygonF([
                QtCore.QPointF(cx, cy - 1.0),
                QtCore.QPointF(cx + 8.0, cy + 2.5),
                QtCore.QPointF(cx, cy + 6.0),
                QtCore.QPointF(cx - 8.0, cy + 2.5),
            ])
            p.setPen(QtGui.QPen(QtGui.QColor("#00E5FF" if hover else "#0284C7"), 1.0))
            p.setBrush(QtGui.QColor(2, 132, 199, 200 if hover else 150))
            p.drawPolygon(cube_top)

            cube_left = QtGui.QPolygonF([
                QtCore.QPointF(cx - 8.0, cy + 2.5),
                QtCore.QPointF(cx, cy + 6.0),
                QtCore.QPointF(cx, cy + 11.0),
                QtCore.QPointF(cx - 8.0, cy + 7.5),
            ])
            p.setBrush(QtGui.QColor(3, 105, 161, 200 if hover else 140))
            p.drawPolygon(cube_left)

            cube_right = QtGui.QPolygonF([
                QtCore.QPointF(cx, cy + 6.0),
                QtCore.QPointF(cx + 8.0, cy + 2.5),
                QtCore.QPointF(cx + 8.0, cy + 7.5),
                QtCore.QPointF(cx, cy + 11.0),
            ])
            p.setBrush(QtGui.QColor(14, 165, 233, 230 if hover else 170))
            p.drawPolygon(cube_right)


class StageViewport(QtOpenGLWidgets.QOpenGLWidget):
    # (a, b, radius_at_a, radius_at_b): tapered limb segments like a posed Blender/Houdini rig proxy
    DRAW_BONES = ((11, 13, .042, .030), (13, 15, .030, .022), (12, 14, .042, .030), (14, 16, .030, .022),
                  (23, 25, .070, .045), (25, 27, .045, .030), (24, 26, .070, .045), (26, 28, .045, .030),
                  (27, 31, .032, .022), (28, 32, .032, .022), (27, 29, .028, .020), (28, 30, .028, .020),
                  (15, 19, .024, .012), (16, 20, .024, .012), (11, 12, .040, .040), (23, 24, .055, .055),
                  (SHO_MID, EAR_MID, .050, .038))
    NODES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    RIG_BONES = ((HIP_MID, SHO_MID), (SHO_MID, EAR_MID), (11, 13), (13, 15), (12, 14), (14, 16),
                 (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32))

    maximize_requested = QtCore.Signal()

    PROP_CONFIGS = {
        "cube": {"size": (0.9, 0.9, 0.9), "color": (0.96, 0.55, 0.10, 1.0)},        # Studio Amber/Orange
        "box": {"size": (1.3, 0.65, 0.85), "color": (0.12, 0.53, 0.90, 1.0)},       # Studio Teal Blue
        "cylinder": {"size": (0.42, 0.90, 0.42), "color": (0.22, 0.32, 0.82, 1.0)}, # Studio Cobalt
        "ramp": {"size": (1.3, 0.65, 1.6), "color": (0.20, 0.65, 0.35, 1.0)},        # Studio Emerald Green
        "step": {"size": (1.2, 0.60, 1.2), "color": (0.58, 0.18, 0.70, 1.0)},        # Studio Amethyst
        "platform": {"size": (2.2, 0.30, 1.6), "color": (0.35, 0.44, 0.52, 1.0)},    # Studio Titanium Gray
        "sphere": {"size": (0.48, 0.48, 0.48), "color": (0.88, 0.22, 0.24, 1.0)},    # Studio Crimson Red
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.joints, self.locks, self.rig = None, np.zeros(2, dtype=bool), None
        self._focus = np.zeros(2)
        self.yaw, self.pitch, self.dist = math.radians(28), math.radians(14), 4.2
        self._mouse, self._quad = None, None
        self._model, self._model_sig, self._fk, self._skel = None, None, None, None
        self.show_rig, self.model_style = False, "bjd"
        self.character, self._char_v, self._char_n, self._tex = None, None, None, {}
        self.countdown_text = None
        self.is_recording = False
        self.rec_time_str = "00:00.0"
        self.props = []
        self.selected_prop = None
        self._prop_id_counter = 0
        self.multi_actor = False
        self.secondary_joints = None
        self.actor_profiles = []
        self.prop_res = None
        self.physics = PhysicsWorld()
        self.audio_lipsync_active = False
        self.stream_info = None
        self._last_proj, self._last_mv, self._last_vp = None, None, None
        self.setMinimumSize(320, 220)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._init_overlay_ui()

    def _init_overlay_ui(self):
        self.btn_maximize = StageOverlayButton("maximize", self)
        self.btn_maximize.setToolTip("Görünümü Büyüt / Küçült (Tam Ekran Modu - Esc)")
        self.btn_maximize.clicked.connect(self._on_maximize_clicked)

        self.btn_props = StageOverlayButton("props", self)
        self.btn_props.setToolTip("Sahne Cisimleri / Stunt Props (Kutu, Küp, Rampa, Podyum, Silindir)")
        self.btn_props.clicked.connect(self._show_props_menu)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w = self.width()
        if hasattr(self, "btn_maximize") and hasattr(self, "btn_props"):
            self.btn_maximize.move(w - 46, 12)
            self.btn_props.move(w - 90, 12)

    def _on_maximize_clicked(self):
        self.maximize_requested.emit()

    def _show_props_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #121824;
                color: #E2E8F0;
                border: 1px solid #00E5FF;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 7px 24px 7px 12px;
                border-radius: 4px;
                font-weight: 500;
            }
            QMenu::item:selected {
                background-color: rgba(0, 229, 255, 0.2);
                color: #00E5FF;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2D3A52;
                margin: 4px 6px;
            }
        """)
        items = [
            ("🧊  Küp (Studio Amber Cube)", "cube"),
            ("📦  Stunt Kutusu (Beveled Box)", "box"),
            ("🛢️  Silindir Kolon (Barrel / Column)", "cylinder"),
            ("📐  Stunt Rampası (Wedge Ramp)", "ramp"),
            ("🪜  Akrobasi Merdiveni (Stairs / Step)", "step"),
            ("🏢  Yükseltilmiş Podyum (Platform)", "platform"),
            ("⚪  Denge Küresi (Stunt Sphere)", "sphere"),
        ]
        for label, kind in items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, k=kind: self.spawn_prop(k))

        menu.addSeparator()
        clear_act = menu.addAction("🗑️  Sahneyi Temizle (Tüm Cisimleri Sil)")
        clear_act.triggered.connect(self.clear_props)
        menu.exec(self.btn_props.mapToGlobal(QtCore.QPoint(0, self.btn_props.height() + 4)))

    def spawn_prop(self, kind="box"):
        cfg = self.PROP_CONFIGS.get(kind, self.PROP_CONFIGS["box"])
        self._prop_id_counter += 1
        y_pos = cfg["size"][1] / 2.0 if kind != "ramp" else 0.0
        prop = StuntProp(
            id=self._prop_id_counter,
            kind=kind,
            pos=np.array([float(self._focus[0]), y_pos, float(self._focus[1]) + 0.8], dtype=float),
            yaw=0.0,
            size=cfg["size"],
            color=cfg["color"],
            selected=True
        )
        for p in self.props:
            p.selected = False
        self.props.append(prop)
        self.selected_prop = prop
        self.update()
        return prop

    def clear_props(self):
        self.props.clear()
        self.selected_prop = None
        self.update()

    def set_character(self, ch):
        """A SkinnedCharacter replaces the built-in procedural model (None = back to the built-in one)."""
        self.character, self._char_v, self._tex = ch, None, {}
        self.update()

    def set_rig(self, rig):
        self.rig = rig
        self.update()

    def update_data(self, joints, locks):
        if joints is not None:
            self.joints, self.locks = joints, locks
            self._focus = 0.9 * self._focus + 0.1 * joints[HIP_MID][[0, 2]]
        self.update()

    def update_result(self, res):
        """Preferred entry point: uses the 52-joint pose and the detailed model when available."""
        self.update_data(res["joints"], res["locks"])
        self.actor_profiles = res.get("actors", [])
        self.prop_res = res.get("prop")
        self.audio_lipsync_active = res.get("audio_lipsync_active", False)
        if self.character is not None and res.get("quats_full") is not None:
            bs = res["faces"][0]["bs"] if res.get("faces") else None
            self._char_v, self._char_n = self.character.pose(res["quats_full"], res["joints"][HIP_MID], bs)
        if res.get("fk") is not None and res.get("skel") is not None:
            skel = res["skel"]
            sig = (self.model_style,) + tuple(np.round(np.array([skel.offsets[j] for j in skel.order], float).ravel(), 3))
            if sig != self._model_sig:                       # rebuild only when the measured proportions change
                self._model, self._model_sig = make_model(skel, self.model_style), sig
            self._skel, self._fk = skel, res["fk"]
        else:
            self._fk = None

    def clear(self):
        self.joints = None
        self.update()

    # -- GL ---------------------------------------------------------------------------------------
    def initializeGL(self):
        gl.glClearColor(*_rgb(C_BG), 1.0)
        self._quad = glu.gluNewQuadric()
        glu.gluQuadricNormals(self._quad, glu.GLU_SMOOTH)

    def paintGL(self):
        dpr = self.devicePixelRatioF()
        w, h = max(self.width(), 1), max(self.height(), 1)
        gl.glViewport(0, 0, int(w * dpr), int(h * dpr))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(50.0, w / h, 0.05, 200.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        tgt = (float(self._focus[0]), 0.9, float(self._focus[1]))
        eye = (tgt[0] + self.dist * math.cos(self.pitch) * math.sin(self.yaw), tgt[1] + self.dist * math.sin(self.pitch),
               tgt[2] + self.dist * math.cos(self.pitch) * math.cos(self.yaw))
        glu.gluLookAt(*eye, *tgt, 0.0, 1.0, 0.0)
        self._last_proj = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX), dtype=np.float64)
        self._last_mv = np.array(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX), dtype=np.float64)
        self._last_vp = np.array(gl.glGetIntegerv(gl.GL_VIEWPORT), dtype=np.int32)
        gl.glDisable(gl.GL_LIGHTING)
        self._grid()
        self._frustums()
        self.physics.update(self.props, dt=0.016)
        self.physics.test_and_react(self.props, self.joints, self.prop_res)
        self._draw_props()
        self._draw_particles()
        if self.selected_prop is not None:
            self._draw_gizmo(self.selected_prop.pos, self.selected_prop.yaw)
        if self.joints is not None:
            J_draw = self.joints.copy()
            min_jy = float(J_draw[:, 1].min())
            if min_jy < 0.0:
                J_draw[:, 1] -= min_jy

            if self.character is not None and self._char_v is not None:
                min_cv = float(self._char_v[:, 1].min())
                if min_cv < 0.0:
                    self._char_v[:, 1] -= min_cv
                self._draw_character()
                self._feet_rings(J_draw)
                if self.show_rig and self._fk is not None:
                    self._skeleton_lines()
            elif self._model is not None and self._fk is not None:
                self._draw_model()
                self._feet_rings(J_draw)
                if self.show_rig:
                    self._skeleton_lines()
            else:
                self._mannequin(J_draw)
                self._rig_overlay(J_draw)

        if self.multi_actor:
            sec_joints = self.secondary_joints
            theme_color = (0.95, 0.55, 0.12, 1.0)
            if len(self.actor_profiles) > 1 and self.actor_profiles[1].joints_3d is not None:
                sec_joints = self.actor_profiles[1].joints_3d[:, :3] * np.array([1.0, -1.0, -1.0])
                theme_color = (*self.actor_profiles[1].theme_rgb, 1.0)
            if sec_joints is not None:
                J_sec = sec_joints.copy()
                min_sy = float(J_sec[:, 1].min())
                if min_sy < 0.0:
                    J_sec[:, 1] -= min_sy
                gl.glPushMatrix()
                gl.glTranslatef(1.3, 0.0, 0.0)
                self._mannequin(J_sec, color_override=theme_color)
                gl.glPopMatrix()

        if self.prop_res is not None:
            self._draw_prop_weapon(self.prop_res)

        self._hud()

    def _grid(self):
        ext = 40
        gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        for i in range(-ext, ext + 1):
            major = (i % 5 == 0)
            base = _rgb("#2D3A52") if major else _rgb(C_GRID)
            a = (1.0 - abs(i) / ext) * (1.0 if major else 0.9)
            for x0, z0, x1, z1 in ((i, -ext, i, 0), (i, 0, i, ext), (-ext, i, 0, i), (0, i, ext, i)):
                far_first = (z0 == -ext) or (x0 == -ext)
                gl.glColor4f(*base, 0.0 if far_first else a)
                gl.glVertex3f(x0, 0, z0)
                gl.glColor4f(*base, a if far_first else 0.0)
                gl.glVertex3f(x1, 0, z1)
        gl.glEnd()
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(*_rgb(C_CYAN), 0.35)
        gl.glVertex3f(-1.5, 0.002, 0); gl.glVertex3f(1.5, 0.002, 0)
        gl.glVertex3f(0, 0.002, -1.5); gl.glVertex3f(0, 0.002, 1.5)
        gl.glEnd()

    def _frustum(self, apex, fwd, right, up, hw, hh, d):
        far = [apex + fwd * d + right * sx * hw + up * sy * hh for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        gl.glLineWidth(1.5)
        gl.glColor4f(*_rgb(C_CYAN), 0.5)
        gl.glBegin(gl.GL_LINES)
        for f in far:
            gl.glVertex3f(*map(float, apex)); gl.glVertex3f(*map(float, f))
        gl.glEnd()
        gl.glBegin(gl.GL_LINE_LOOP)
        for f in far:
            gl.glVertex3f(*map(float, f))
        gl.glEnd()

    def _frustums(self):
        if self.rig is not None and len(self.rig) >= 1:
            for cam in self.rig.cameras:
                R = cam.R
                right, up, fwd = R.T[:, 0], -R.T[:, 1], R.T[:, 2]
                w, h = cam.size
                d = 0.6
                self._frustum(cam.center, fwd, right, up, d * (w / 2) / cam.focal, d * (h / 2) / cam.focal, d)
            return
        self._frustum(np.array([0.0, 1.0, 3.4]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]), 1.2, 0.68, 3.2)

    # -- mannequin --------------------------------------------------------------------------------
    def _cylinder(self, a, b, r, r2=None):
        v = np.asarray(b) - np.asarray(a)
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            return
        d = v / L
        gl.glPushMatrix()
        gl.glTranslatef(*[float(x) for x in a])
        axis = np.cross([0.0, 0.0, 1.0], d)
        s = float(np.linalg.norm(axis))
        if s > 1e-6:
            gl.glRotatef(math.degrees(math.atan2(s, float(d[2]))), *[float(x) for x in axis])
        elif d[2] < 0:
            gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        glu.gluCylinder(self._quad, r, r if r2 is None else r2, L, 14, 1)
        gl.glPopMatrix()

    def _sphere(self, p, r):
        gl.glPushMatrix()
        gl.glTranslatef(*[float(x) for x in p])
        glu.gluSphere(self._quad, r, 14, 10)
        gl.glPopMatrix()

    def _ellipsoid(self, c, up, side, sx, sy, sz):
        d = unit(np.asarray(up, dtype=float))
        q = q_from_vectors(np.array([0.0, 1.0, 0.0]), d)
        tt = unit(side - d * float(np.dot(side, d)), [1.0, 0.0, 0.0])
        x1 = q_rotate(q, np.array([1.0, 0.0, 0.0]))
        roll = math.atan2(float(np.dot(np.cross(x1, tt), d)), float(np.dot(x1, tt)))
        q = q_mul(q_axis_angle(d, roll), q)
        w, x, y, z = q
        M = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0], [0, 0, 0, 1]])
        gl.glPushMatrix()
        gl.glTranslatef(*[float(v) for v in c])
        gl.glMultMatrixf(np.ascontiguousarray(M.T, dtype=np.float32))
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glScalef(float(sx), float(sy), float(sz))
        glu.gluSphere(self._quad, 1.0, 20, 14)
        gl.glPopMatrix()

    def _mannequin(self, J, color_override=None):
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (2.0, 4.0, 3.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.35, 0.38, 0.45, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.9, 0.95, 1.0, 1.0))
        gl.glColor4f(*(color_override or (0.42, 0.50, 0.62, 1.0)))
        for a, b, r, r2 in self.DRAW_BONES:
            self._cylinder(J[a], J[b], r, r2)
        self._sphere(J[EAR_MID] + np.array([0.0, 0.07, 0.0]), 0.105)
        torso, side = J[SHO_MID] - J[HIP_MID], J[11] - J[12]
        hw, sw = float(np.linalg.norm(J[23] - J[24])) / 2, float(np.linalg.norm(side)) / 2
        self._ellipsoid(J[HIP_MID] + torso * 0.06, torso, side, hw * 1.25, 0.10, 0.11)
        self._ellipsoid(J[HIP_MID] + torso * 0.66, torso, side, sw * 1.05, float(np.linalg.norm(torso)) * 0.36, 0.13)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor4f(*_rgb(C_CYAN), 0.16)
        for i in self.NODES:
            self._sphere(J[i], 0.06)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glColor4f(*_rgb(C_CYAN), 1.0)
        for i in self.NODES:
            self._sphere(J[i], 0.027)
        gl.glLineWidth(2.0)
        gl.glColor4f(*_rgb(C_AMBER), 0.95)                       # foot-lock rings
        for k, s in enumerate("LR"):
            if self.locks[k]:
                ank = LEGS[s][2]
                gl.glBegin(gl.GL_LINE_LOOP)
                for tt in range(24):
                    ang = tt * math.pi / 12
                    gl.glVertex3f(float(J[ank, 0]) + 0.09 * math.cos(ang), 0.004, float(J[ank, 2]) + 0.09 * math.sin(ang))
                gl.glEnd()

    def _draw_model(self):
        pos, rot = self._fk
        V, N = self._model.pose(pos, rot)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (2.0, 4.0, 3.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.38, 0.40, 0.46, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.9, 0.92, 0.95, 1.0))
        for arr in (gl.GL_VERTEX_ARRAY, gl.GL_NORMAL_ARRAY, gl.GL_COLOR_ARRAY):
            gl.glEnableClientState(arr)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, V)
        gl.glNormalPointer(gl.GL_FLOAT, 0, N)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self._model.colors)
        gl.glDrawElements(gl.GL_TRIANGLES, self._model.tris.size, gl.GL_UNSIGNED_INT, self._model.tris)

        # Wireframe topology lines overlay (manken üstü çizgiler)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
        gl.glPolygonOffset(-1.0, -1.0)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(1.1)
        gl.glColor4f(0.25, 0.28, 0.35, 0.35)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDrawElements(gl.GL_TRIANGLES, self._model.tris.size, gl.GL_UNSIGNED_INT, self._model.tris)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)

        for arr in (gl.GL_VERTEX_ARRAY, gl.GL_NORMAL_ARRAY):
            gl.glDisableClientState(arr)
        gl.glDisable(gl.GL_LIGHTING)

    def _texture(self, pr):
        img = pr["image"]
        if img is None:
            return None
        key = id(img)
        if key not in self._tex:
            tid = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
            for k, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                         (gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, k, v)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.shape[1], img.shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                            np.ascontiguousarray(img))
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self._tex[key] = tid
        return self._tex[key]

    def _draw_character(self):
        ch = self.character
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (2.0, 4.0, 3.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.45, 0.46, 0.5, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.85, 0.85, 0.85, 1.0))
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self._char_v)
        gl.glNormalPointer(gl.GL_FLOAT, 0, self._char_n)
        for pr in ch.prims:
            tid = self._texture(pr)
            gl.glColor4f(*[float(c) for c in pr["color"]])
            if tid is not None:
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
                gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
                gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
                gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, ch.uv)
            if pr["alpha"] == "MASK":
                gl.glEnable(gl.GL_ALPHA_TEST)
                gl.glAlphaFunc(gl.GL_GREATER, float(pr["cutoff"]))
            blend = pr["alpha"] == "BLEND"
            if blend:
                gl.glDepthMask(gl.GL_FALSE)
            tris = pr.setdefault("_tris32", np.ascontiguousarray(pr["tris"], dtype=np.uint32))
            gl.glDrawElements(gl.GL_TRIANGLES, tris.size, gl.GL_UNSIGNED_INT, tris)
            if blend:
                gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_ALPHA_TEST)
            if tid is not None:
                gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
                gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)

    def _skeleton_lines(self):
        pos = self._fk[0]
        idx = {j: i for i, j in enumerate(self._skel.order)}
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(1.6)
        gl.glColor4f(*_rgb(C_CYAN), 0.9)
        gl.glBegin(gl.GL_LINES)
        for j, p in self._skel.parent.items():
            gl.glVertex3f(*map(float, pos[idx[p]]))
            gl.glVertex3f(*map(float, pos[idx[j]]))
        gl.glEnd()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _feet_rings(self, J):
        gl.glLineWidth(2.0)
        gl.glColor4f(*_rgb(C_AMBER), 0.95)
        for k, s in enumerate("LR"):
            if self.locks[k]:
                ank = LEGS[s][2]
                gl.glBegin(gl.GL_LINE_LOOP)
                for tt in range(24):
                    ang = tt * math.pi / 12
                    gl.glVertex3f(float(J[ank, 0]) + 0.09 * math.cos(ang), 0.004, float(J[ank, 2]) + 0.09 * math.sin(ang))
                gl.glEnd()

    def _rig_overlay(self, J):
        """Blender-style octahedral bones drawn over the shell."""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(1.4)
        gl.glColor4f(*_rgb(C_CYAN), 0.85)
        V = lambda p: gl.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        gl.glBegin(gl.GL_LINES)
        for a, b in self.RIG_BONES:
            p0, p1 = J[a], J[b]
            L = float(np.linalg.norm(p1 - p0))
            if L < 1e-6:
                continue
            d = (p1 - p0) / L
            u = unit(np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 0.0, 1.0]))
            v = np.cross(d, u)
            c, wd = p0 + d * 0.2 * L, 0.11 * L
            ring = [c + u * wd, c + v * wd, c - u * wd, c - v * wd]
            for i, r in enumerate(ring):
                V(p0); V(r); V(p1); V(r); V(r); V(ring[(i + 1) % 4])
        gl.glEnd()
        gl.glEnable(gl.GL_DEPTH_TEST)

    # -- stunt props rendering -------------------------------------------------------------------
    def _draw_props(self):
        if not self.props:
            return
        for prop in self.props:
            gl.glPushMatrix()
            gl.glTranslatef(float(prop.pos[0]), float(prop.pos[1]), float(prop.pos[2]))
            gl.glRotatef(float(prop.yaw), 0.0, 1.0, 0.0)
            if prop.kind in ("cube", "box", "platform"):
                self._draw_beveled_box(prop.size[0], prop.size[1], prop.size[2], prop.color, prop.selected)
            elif prop.kind == "cylinder":
                self._draw_cylinder_prop(prop.size[0], prop.size[1], prop.color, prop.selected)
            elif prop.kind == "ramp":
                self._draw_ramp_prop(prop.size[0], prop.size[1], prop.size[2], prop.color, prop.selected)
            elif prop.kind == "sphere":
                self._draw_sphere_prop(prop.size[0], prop.color, prop.selected)
            elif prop.kind == "step":
                self._draw_steps_prop(prop.size[0], prop.size[1], prop.size[2], prop.color, prop.selected)
            gl.glPopMatrix()

    def _draw_beveled_box(self, sx, sy, sz, color, selected=False):
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_QUADS)
        # Top (+Y)
        gl.glNormal3f(0.0, 1.0, 0.0)
        gl.glVertex3f(-hx, hy, hz); gl.glVertex3f(hx, hy, hz); gl.glVertex3f(hx, hy, -hz); gl.glVertex3f(-hx, hy, -hz)
        # Bottom (-Y)
        gl.glNormal3f(0.0, -1.0, 0.0)
        gl.glVertex3f(-hx, -hy, -hz); gl.glVertex3f(hx, -hy, -hz); gl.glVertex3f(hx, -hy, hz); gl.glVertex3f(-hx, -hy, hz)
        # Front (+Z)
        gl.glNormal3f(0.0, 0.0, 1.0)
        gl.glVertex3f(-hx, -hy, hz); gl.glVertex3f(hx, -hy, hz); gl.glVertex3f(hx, hy, hz); gl.glVertex3f(-hx, hy, hz)
        # Back (-Z)
        gl.glNormal3f(0.0, 0.0, -1.0)
        gl.glVertex3f(hx, -hy, -hz); gl.glVertex3f(-hx, -hy, -hz); gl.glVertex3f(-hx, hy, -hz); gl.glVertex3f(hx, hy, -hz)
        # Right (+X)
        gl.glNormal3f(1.0, 0.0, 0.0)
        gl.glVertex3f(hx, -hy, hz); gl.glVertex3f(hx, -hy, -hz); gl.glVertex3f(hx, hy, -hz); gl.glVertex3f(hx, hy, hz)
        # Left (-X)
        gl.glNormal3f(-1.0, 0.0, 0.0)
        gl.glVertex3f(-hx, -hy, -hz); gl.glVertex3f(-hx, -hy, hz); gl.glVertex3f(-hx, hy, hz); gl.glVertex3f(-hx, hy, -hz)
        gl.glEnd()
        gl.glDisable(gl.GL_LIGHTING)

        # Wireframe Bevel Edges
        gl.glLineWidth(2.5 if selected else 1.2)
        edge_col = (0.0, 0.95, 1.0, 0.95) if selected else (color[0] * 0.45, color[1] * 0.45, color[2] * 0.45, 0.75)
        gl.glColor4f(*edge_col)
        gl.glBegin(gl.GL_LINES)
        for p1, p2 in [
            ((-hx, -hy, -hz), (hx, -hy, -hz)), ((hx, -hy, -hz), (hx, -hy, hz)), ((hx, -hy, hz), (-hx, -hy, hz)), ((-hx, -hy, hz), (-hx, -hy, -hz)),
            ((-hx, hy, -hz), (hx, hy, -hz)), ((hx, hy, -hz), (hx, hy, hz)), ((hx, hy, hz), (-hx, hy, hz)), ((-hx, hy, hz), (-hx, hy, -hz)),
            ((-hx, -hy, -hz), (-hx, hy, -hz)), ((hx, -hy, -hz), (hx, hy, -hz)), ((hx, -hy, hz), (hx, hy, hz)), ((-hx, -hy, hz), (-hx, hy, hz))
        ]:
            gl.glVertex3f(*p1); gl.glVertex3f(*p2)
        gl.glEnd()

    def _draw_cylinder_prop(self, radius, height, color, selected=False):
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glColor4f(*color)
        gl.glPushMatrix()
        gl.glTranslatef(0.0, -height / 2.0, 0.0)
        gl.glRotatef(-90.0, 1.0, 0.0, 0.0)
        glu.gluCylinder(self._quad, radius, radius, height, 28, 2)
        gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        glu.gluDisk(self._quad, 0.0, radius, 28, 1)
        gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        gl.glTranslatef(0.0, 0.0, height)
        glu.gluDisk(self._quad, 0.0, radius, 28, 1)
        gl.glPopMatrix()
        gl.glDisable(gl.GL_LIGHTING)

        if selected:
            gl.glLineWidth(2.5)
            gl.glColor4f(0.0, 0.95, 1.0, 0.95)
            for y_off in (-height / 2.0, height / 2.0):
                gl.glBegin(gl.GL_LINE_LOOP)
                for i in range(24):
                    ang = i * 2.0 * math.pi / 24.0
                    gl.glVertex3f(radius * math.cos(ang), y_off, radius * math.sin(ang))
                gl.glEnd()

    def _draw_ramp_prop(self, sx, sy, sz, color, selected=False):
        hx, hz = sx / 2.0, sz / 2.0
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_TRIANGLES)
        # Left wall (-X)
        gl.glNormal3f(-1.0, 0.0, 0.0)
        gl.glVertex3f(-hx, 0.0, -hz); gl.glVertex3f(-hx, sy, hz); gl.glVertex3f(-hx, 0.0, hz)
        # Right wall (+X)
        gl.glNormal3f(1.0, 0.0, 0.0)
        gl.glVertex3f(hx, 0.0, -hz); gl.glVertex3f(hx, 0.0, hz); gl.glVertex3f(hx, sy, hz)
        gl.glEnd()
        gl.glBegin(gl.GL_QUADS)
        # Bottom (+Y=0)
        gl.glNormal3f(0.0, -1.0, 0.0)
        gl.glVertex3f(-hx, 0.0, -hz); gl.glVertex3f(hx, 0.0, -hz); gl.glVertex3f(hx, 0.0, hz); gl.glVertex3f(-hx, 0.0, hz)
        # Back (+Z)
        gl.glNormal3f(0.0, 0.0, 1.0)
        gl.glVertex3f(-hx, 0.0, hz); gl.glVertex3f(hx, 0.0, hz); gl.glVertex3f(hx, sy, hz); gl.glVertex3f(-hx, sy, hz)
        # Sloped ramp surface
        slope_n = unit([0.0, sz, -sy])
        gl.glNormal3f(*slope_n)
        gl.glVertex3f(-hx, 0.0, -hz); gl.glVertex3f(-hx, sy, hz); gl.glVertex3f(hx, sy, hz); gl.glVertex3f(hx, 0.0, -hz)
        gl.glEnd()
        gl.glDisable(gl.GL_LIGHTING)
        if selected:
            gl.glLineWidth(2.5)
            gl.glColor4f(0.0, 0.95, 1.0, 0.95)
            gl.glBegin(gl.GL_LINES)
            for p1, p2 in [
                ((-hx, 0.0, -hz), (hx, 0.0, -hz)), ((-hx, sy, hz), (hx, sy, hz)),
                ((-hx, 0.0, hz), (hx, 0.0, hz)), ((-hx, 0.0, -hz), (-hx, sy, hz)),
                ((hx, 0.0, -hz), (hx, sy, hz)), ((-hx, 0.0, hz), (-hx, sy, hz)), ((hx, 0.0, hz), (hx, sy, hz))
            ]:
                gl.glVertex3f(*p1); gl.glVertex3f(*p2)
            gl.glEnd()

    def _draw_sphere_prop(self, radius, color, selected=False):
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glColor4f(*color)
        glu.gluSphere(self._quad, radius, 26, 18)
        gl.glDisable(gl.GL_LIGHTING)
        if selected:
            gl.glLineWidth(2.5)
            gl.glColor4f(0.0, 0.95, 1.0, 0.95)
            for ax, ay, az in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                gl.glBegin(gl.GL_LINE_LOOP)
                for i in range(24):
                    ang = i * 2.0 * math.pi / 24.0
                    c, s = radius * math.cos(ang), radius * math.sin(ang)
                    if ax: gl.glVertex3f(0.0, c, s)
                    elif ay: gl.glVertex3f(c, 0.0, s)
                    else: gl.glVertex3f(c, s, 0.0)
                gl.glEnd()

    def _draw_steps_prop(self, sx, sy, sz, color, selected=False):
        n_steps = 3
        step_h = sy / n_steps
        step_d = sz / n_steps
        for i in range(n_steps):
            gl.glPushMatrix()
            h = step_h * (i + 1)
            z = -sz / 2.0 + (i + 0.5) * step_d
            y = -sy / 2.0 + h / 2.0
            gl.glTranslatef(0.0, y, z)
            self._draw_beveled_box(sx, h, step_d, color, selected and i == (n_steps - 1))
            gl.glPopMatrix()

    def _draw_gizmo(self, pos, yaw=0.0):
        # Blender-style 3D Transform Gizmo (Red +X, Green +Y, Blue +Z)
        gl.glPushMatrix()
        gl.glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
        gl.glRotatef(float(yaw), 0.0, 1.0, 0.0)

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        L = 0.52
        stem_r = 0.015
        cone_r = 0.042
        cone_h = 0.12

        # White center pivot
        gl.glColor4f(0.95, 0.95, 0.95, 1.0)
        glu.gluSphere(self._quad, 0.032, 16, 12)

        # X Axis (Red)
        gl.glColor4f(0.95, 0.22, 0.22, 1.0)
        gl.glPushMatrix()
        gl.glRotatef(90.0, 0.0, 1.0, 0.0)
        glu.gluCylinder(self._quad, stem_r, stem_r, L - cone_h, 16, 1)
        gl.glTranslatef(0.0, 0.0, L - cone_h)
        glu.gluCylinder(self._quad, cone_r, 0.0, cone_h, 16, 1)
        gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        glu.gluDisk(self._quad, 0.0, cone_r, 16, 1)
        gl.glPopMatrix()

        # Y Axis (Green)
        gl.glColor4f(0.22, 0.90, 0.25, 1.0)
        gl.glPushMatrix()
        gl.glRotatef(-90.0, 1.0, 0.0, 0.0)
        glu.gluCylinder(self._quad, stem_r, stem_r, L - cone_h, 16, 1)
        gl.glTranslatef(0.0, 0.0, L - cone_h)
        glu.gluCylinder(self._quad, cone_r, 0.0, cone_h, 16, 1)
        gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        glu.gluDisk(self._quad, 0.0, cone_r, 16, 1)
        gl.glPopMatrix()

        # Z Axis (Blue)
        gl.glColor4f(0.20, 0.45, 0.98, 1.0)
        gl.glPushMatrix()
        glu.gluCylinder(self._quad, stem_r, stem_r, L - cone_h, 16, 1)
        gl.glTranslatef(0.0, 0.0, L - cone_h)
        glu.gluCylinder(self._quad, cone_r, 0.0, cone_h, 16, 1)
        gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        glu.gluDisk(self._quad, 0.0, cone_r, 16, 1)
        gl.glPopMatrix()

        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPopMatrix()

    def _project_point(self, pt):
        # High precision pure CPU camera ray & frustum projection: independent of GL context state
        w, h = max(self.width(), 1), max(self.height(), 1)
        tgt = np.array([float(self._focus[0]), 0.9, float(self._focus[1])])
        eye = np.array([
            tgt[0] + self.dist * math.cos(self.pitch) * math.sin(self.yaw),
            tgt[1] + self.dist * math.sin(self.pitch),
            tgt[2] + self.dist * math.cos(self.pitch) * math.cos(self.yaw)
        ])
        z_axis = eye - tgt
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            return None
        z_axis /= z_norm

        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)

        p_rel = np.asarray(pt, float) - eye
        cx = float(np.dot(p_rel, x_axis))
        cy = float(np.dot(p_rel, y_axis))
        cz = float(np.dot(p_rel, z_axis))

        depth = -cz
        if depth <= 0.05:
            return None

        tan_half = math.tan(math.radians(25.0))
        aspect = w / h

        ndc_x = (cx / depth) / (tan_half * aspect)
        ndc_y = (cy / depth) / tan_half

        sx = (ndc_x + 1.0) * 0.5 * w
        sy = (1.0 - ndc_y) * 0.5 * h
        return QtCore.QPointF(sx, sy)

    def _prop_screen_bounds(self, prop):
        hx, hy, hz = prop.size[0] / 2.0, prop.size[1] / 2.0, prop.size[2] / 2.0
        yaw_rad = math.radians(prop.yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        corners = []
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for dz in (-hz, hz):
                    rx = dx * cos_y + dz * sin_y
                    rz = -dx * sin_y + dz * cos_y
                    corners.append([prop.pos[0] + rx, prop.pos[1] + dy, prop.pos[2] + rz])
        corners.append([prop.pos[0], prop.pos[1], prop.pos[2]])
        pts_2d = [self._project_point(c) for c in corners]
        pts_2d = [p for p in pts_2d if p is not None]
        if not pts_2d:
            return None
        min_x = min(p.x() for p in pts_2d)
        max_x = max(p.x() for p in pts_2d)
        min_y = min(p.y() for p in pts_2d)
        max_y = max(p.y() for p in pts_2d)
        return QtCore.QRectF(min_x - 10.0, min_y - 10.0, (max_x - min_x) + 20.0, (max_y - min_y) + 20.0)

    def _find_clicked_prop(self, mouse_pos):
        if not self.props:
            return None
        for prop in reversed(self.props):
            rect = self._prop_screen_bounds(prop)
            if rect is not None and rect.contains(mouse_pos):
                return prop
        return None

    def _hud(self):
        # Full OpenGL clean state reset to prevent QPainter font glyph rasterization corruption
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_ALPHA_TEST)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        for client_state in (gl.GL_VERTEX_ARRAY, gl.GL_NORMAL_ARRAY, gl.GL_COLOR_ARRAY, gl.GL_TEXTURE_COORD_ARRAY):
            try:
                gl.glDisableClientState(client_state)
            except Exception:
                pass

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        p.setPen(QtGui.QColor(C_CYAN))
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        p.drawText(10, 18, t("view.stage"))
        if self.multi_actor:
            badge_x = 110
            profs = self.actor_profiles if self.actor_profiles else [
                type("MockP", (), {"label": "Aktör 1", "color_hex": "#00E5FF", "color_name": "Mavi/Cyan", "theme_rgb": (0.0, 0.9, 1.0)})(),
                type("MockP", (), {"label": "Aktör 2", "color_hex": "#FFB300", "color_name": "Turuncu/Amber", "theme_rgb": (0.98, 0.55, 0.12)})(),
            ]
            for pr in profs:
                pill_rect = QtCore.QRect(badge_x, 6, 142, 20)
                p.setPen(QtGui.QPen(QtGui.QColor(pr.color_hex), 1.2))
                p.setBrush(QtGui.QColor(18, 24, 36, 210))
                p.drawRoundedRect(pill_rect, 10, 10)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(pr.color_hex))
                p.drawEllipse(QtCore.QPointF(badge_x + 10, 16), 4.5, 4.5)
                p.setPen(QtGui.QColor("#E2E8F0"))
                bf = p.font()
                bf.setPointSize(8)
                bf.setBold(True)
                p.setFont(bf)
                p.drawText(QtCore.QRect(badge_x + 20, 6, 118, 20), QtCore.Qt.AlignVCenter, f"{pr.label}: {pr.color_name}")
                badge_x += 150
        if self.joints is None and self.countdown_text is None:
            p.setPen(QtGui.QColor(C_AMBER))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, t("nosignal"))

        # Large Visual Countdown Overlay
        if self.countdown_text is not None:
            p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 150))
            cx, cy = self.width() / 2.0, self.height() / 2.0
            r = min(self.width(), self.height()) * 0.22
            is_go = (str(self.countdown_text) in ("0", "REC!", "GO!"))
            ring_color = QtGui.QColor("#FF1744" if is_go else C_CYAN)

            # Circular target ring
            p.setPen(QtGui.QPen(ring_color, 6))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(cx, cy), r, r)

            # Massive bold countdown number
            cf = p.font()
            cf.setPointSize(max(32, int(min(self.width(), self.height()) * 0.18)))
            cf.setBold(True)
            p.setFont(cf)
            p.setPen(ring_color)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, str(self.countdown_text))

            # Subtitle
            sub_f = p.font()
            sub_f.setPointSize(11)
            sub_f.setBold(True)
            p.setFont(sub_f)
            p.setPen(QtGui.QColor("#FFFFFF"))
            p.setBrush(QtCore.Qt.NoBrush)
            sub_rect = QtCore.QRect(0, int(cy + r + 18), self.width(), 32)
            p.drawText(sub_rect, QtCore.Qt.AlignCenter, "HAZIRLANIN / GET READY")

        # Studio Active Recording Indicator
        elif self.is_recording:
            p.setPen(QtGui.QPen(QtGui.QColor("#FF1744"), 4))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))

            badge_w, badge_h = 160, 30
            bx, by = self.width() - badge_w - 96, 12
            badge_rect = QtCore.QRect(bx, by, badge_w, badge_h)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(220, 20, 60, 220))
            p.drawRoundedRect(badge_rect, 6, 6)

            p.setPen(QtCore.Qt.white)
            bf = p.font()
            bf.setPointSize(10)
            bf.setBold(True)
            p.setFont(bf)
            p.drawText(badge_rect, QtCore.Qt.AlignCenter, f"REC  {self.rec_time_str}")

        # Live Streaming Status Badge
        if getattr(self, 'stream_info', None):
            st_rect = QtCore.QRect(10, self.height() - 64, 260, 24)
            p.setPen(QtGui.QPen(QtGui.QColor("#00E5FF"), 1.2))
            p.setBrush(QtGui.QColor(10, 25, 38, 220))
            p.drawRoundedRect(st_rect, 10, 10)
            p.setPen(QtGui.QColor("#E0F2FE"))
            sf = p.font(); sf.setPointSize(8); sf.setBold(True); p.setFont(sf)
            p.drawText(st_rect, QtCore.Qt.AlignCenter, f"📡 {self.stream_info.upper()}")

        # Live Audio Lip-Sync Badge
        if getattr(self, "audio_lipsync_active", False):
            ab_rect = QtCore.QRect(self.width() - 280, 12, 175, 24)
            p.setPen(QtGui.QPen(QtGui.QColor("#A855F7"), 1.2))
            p.setBrush(QtGui.QColor(20, 15, 32, 210))
            p.drawRoundedRect(ab_rect, 10, 10)
            p.setPen(QtGui.QColor("#D8B4FE"))
            af = p.font(); af.setPointSize(8); af.setBold(True); p.setFont(af)
            p.drawText(ab_rect, QtCore.Qt.AlignCenter, "🎙️ SESLI DUDAK SENK (AKTIF)")

        # Weapon / Prop Tracking Badge
        if self.prop_res is not None and getattr(self.prop_res, "detected", False):
            wb_rect = QtCore.QRect(self.width() - 480, 12, 192, 24)
            p.setPen(QtGui.QPen(QtGui.QColor("#10B981"), 1.2))
            p.setBrush(QtGui.QColor(12, 28, 22, 210))
            p.drawRoundedRect(wb_rect, 10, 10)
            p.setPen(QtGui.QColor("#6EE7B7"))
            wf = p.font(); wf.setPointSize(8); wf.setBold(True); p.setFont(wf)
            p.drawText(wb_rect, QtCore.Qt.AlignCenter, f"⚔️ {self.prop_res.kind.upper()} ({self.prop_res.color_name} BANT)")

        # Dublor Carpisma / Hit Reaction Banner
        if hasattr(self, "physics") and self.physics.hit_events:
            last_hit = self.physics.hit_events[-1]
            hit_rect = QtCore.QRect(int(self.width() / 2 - 120), 44, 240, 28)
            p.setPen(QtGui.QPen(QtGui.QColor("#F59E0B"), 2.0))
            p.setBrush(QtGui.QColor(45, 25, 10, 220))
            p.drawRoundedRect(hit_rect, 12, 12)
            p.setPen(QtGui.QColor("#FDE68A"))
            hf = p.font(); hf.setPointSize(9); hf.setBold(True); p.setFont(hf)
            p.drawText(hit_rect, QtCore.Qt.AlignCenter, f"💥 DARBE! ({last_hit.source.upper()})")

        # Prop manipulation instructions HUD
        if self.selected_prop is not None:
            hud_bar = QtCore.QRect(12, self.height() - 34, self.width() - 24, 26)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(15, 20, 32, 220))
            p.drawRoundedRect(hud_bar, 6, 6)
            p.setPen(QtGui.QColor("#00E5FF"))
            hf = p.font()
            hf.setPointSize(9)
            hf.setBold(True)
            p.setFont(hf)
            info = f"Cisim: {self.selected_prop.kind.upper()} | Sol Tık: Zeminde Taşı | Sağ Tık: Çevir | Fn/Shift/Alt + Sol Tık: Uçur/Yükseklik | Del: Sil"
            p.drawText(hud_bar, QtCore.Qt.AlignCenter, info)

        p.end()



    def _draw_particles(self):
        if not self.physics.particles:
            return
        gl.glDisable(gl.GL_LIGHTING)
        gl.glPointSize(4.5)
        gl.glBegin(gl.GL_POINTS)
        for pt in self.physics.particles:
            gl.glColor4f(float(pt.color[0]), float(pt.color[1]), float(pt.color[2]), float(pt.life))
            gl.glVertex3f(float(pt.pos[0]), float(pt.pos[1]), float(pt.pos[2]))
        gl.glEnd()

    def _draw_prop_weapon(self, prop):
        if prop is None or prop.base_3d is None:
            return
        base = prop.base_3d
        tip = prop.tip_3d
        gl.glPushMatrix()
        gl.glTranslatef(float(base[0]), float(base[1]), float(base[2]))
        # Transform along wrist-to-tip direction
        v = tip - base
        L = float(np.linalg.norm(v))
        if L > 1e-4:
            d = v / L
            axis = np.cross([0.0, 0.0, 1.0], d)
            s = float(np.linalg.norm(axis))
            if s > 1e-4:
                gl.glRotatef(math.degrees(math.atan2(s, float(d[2]))), *[float(x) for x in axis])
            elif d[2] < 0:
                gl.glRotatef(180.0, 1.0, 0.0, 0.0)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        if prop.kind == 'sword':
            # Hilt (Kabza) - Dark Steel
            gl.glColor4f(0.22, 0.22, 0.25, 1.0)
            glu.gluCylinder(self._quad, 0.018, 0.018, 0.14, 14, 1)
            # Guard (Tsuba / Koruma)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.14)
            gl.glColor4f(0.85, 0.70, 0.20, 1.0)
            glu.gluDisk(self._quad, 0.0, 0.055, 16, 1)
            # Blade (Celik Namlu)
            gl.glColor4f(0.92, 0.94, 0.98, 1.0)
            glu.gluCylinder(self._quad, 0.022, 0.006, L - 0.14, 14, 1)
            gl.glPopMatrix()
        elif prop.kind == 'knife':
            gl.glColor4f(0.18, 0.18, 0.20, 1.0)
            glu.gluCylinder(self._quad, 0.016, 0.016, 0.09, 12, 1)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.09)
            gl.glColor4f(0.88, 0.90, 0.95, 1.0)
            glu.gluCylinder(self._quad, 0.018, 0.004, L - 0.09, 12, 1)
            gl.glPopMatrix()
        elif prop.kind == 'pistol':
            gl.glColor4f(0.15, 0.16, 0.18, 1.0)
            glu.gluCylinder(self._quad, 0.022, 0.020, L, 12, 1)
        elif prop.kind == 'shield':
            gl.glColor4f(0.20, 0.50, 0.85, 1.0)
            glu.gluDisk(self._quad, 0.0, 0.32, 22, 2)
        else:
            # Staff (Bo-Staff) - Polished Wood
            gl.glColor4f(0.65, 0.38, 0.18, 1.0)
            glu.gluCylinder(self._quad, 0.018, 0.018, L, 14, 1)
        gl.glDisable(gl.GL_LIGHTING)
        # Color tape marker highlight at tip
        if prop.detected:
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, L)
            gl.glColor4f(0.0, 1.0, 0.35, 0.95)
            glu.gluSphere(self._quad, 0.026, 12, 10)
            gl.glPopMatrix()
        gl.glPopMatrix()

    def mousePressEvent(self, e):
        self._mouse = e.position()
        if e.button() == QtCore.Qt.LeftButton:
            clicked = self._find_clicked_prop(e.position())
            for p in self.props:
                p.selected = False
            if clicked is not None:
                clicked.selected = True
                self.selected_prop = clicked
                self.update()
            else:
                if self.selected_prop is not None:
                    self.selected_prop = None
                    self.update()

    def mouseMoveEvent(self, e):
        if self._mouse is not None:
            d = e.position() - self._mouse
            if self.selected_prop is not None:
                if e.buttons() & QtCore.Qt.LeftButton:
                    if e.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier | QtCore.Qt.ControlModifier):
                        # Height elevation (uçurma / yüksekte konumlama: Fn/Shift/Alt/Ctrl + Sol tık)
                        self.selected_prop.pos[1] = max(0.0, float(self.selected_prop.pos[1] - d.y() * 0.015))
                    else:
                        # Ground XZ movement
                        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
                        sens = 0.008 * (self.dist / 4.2)
                        self.selected_prop.pos[0] += (d.x() * cos_y + d.y() * sin_y) * sens
                        self.selected_prop.pos[2] += (-d.x() * sin_y + d.y() * cos_y) * sens
                    self.update()
                elif e.buttons() & QtCore.Qt.RightButton:
                    # Yaw rotation
                    self.selected_prop.yaw += d.x() * 0.8
                    self.update()
            elif e.buttons() & QtCore.Qt.LeftButton:
                self.yaw -= d.x() * 0.008
                self.pitch = float(np.clip(self.pitch + d.y() * 0.008, -0.2, 1.4))
                self.update()
        self._mouse = e.position()

    def wheelEvent(self, e):
        self.dist = float(np.clip(self.dist * (0.999 ** e.angleDelta().y()), 1.2, 15.0))
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self.selected_prop is not None and self.selected_prop in self.props:
                self.props.remove(self.selected_prop)
                self.selected_prop = None
                self.update()
        super().keyPressEvent(e)
