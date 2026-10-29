"""3D face viewport (GLSL skin shader) and the scene builder that turns face packets into drawable items."""
import math
import sys

import numpy as np
from PySide6 import QtCore, QtOpenGLWidgets
from OpenGL import GL as gl
from OpenGL import GLU as glu
from OpenGL.GL import shaders as gl_shaders

from ..character import MeshBuilder
from ..core.bodymesh import ellipsoid
from ..core.face import heat_rgba

VS = """#version 120
attribute vec3 aPos; attribute vec3 aNrm; attribute vec4 aCol;
varying vec3 vN; varying vec3 vP; varying vec4 vC;
void main(){ vec4 e = gl_ModelViewMatrix * vec4(aPos,1.0); vP = e.xyz; vN = gl_NormalMatrix * aNrm;
             vC = aCol; gl_Position = gl_ProjectionMatrix * e; }"""
FS = """#version 120
uniform vec3 uSkin; uniform float uHeat;
varying vec3 vN; varying vec3 vP; varying vec4 vC;
void main(){
  vec3 N = normalize(vN); vec3 V = normalize(-vP); if (dot(N,V) < 0.0) N = -N;
  vec3 L1 = normalize(vec3(0.5,0.8,1.0)); vec3 L2 = normalize(vec3(-0.8,0.2,0.6));
  float w1 = clamp((dot(N,L1)+0.45)/1.45,0.0,1.0); float w2 = clamp((dot(N,L2)+0.45)/1.45,0.0,1.0);
  vec3 base = mix(uSkin, vC.rgb, vC.a*uHeat);
  vec3 bleed = vec3(0.60,0.14,0.07) * pow(clamp(1.0-abs(dot(N,L1)),0.0,1.0),2.0) * 0.22;
  vec3 col = base*(0.16+0.85*w1+0.32*w2) + base*bleed;
  float spec = pow(max(dot(N,normalize(L1+V)),0.0),52.0)*0.20;
  float rim = pow(1.0-max(dot(N,V),0.0),3.0)*0.18;
  gl_FragColor = vec4(col + spec + vec3(0.0,0.7,0.8)*rim, 1.0); }"""


class FaceViewport(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items, self.wire, self.heat, self.paints = [], False, True, 0
        self.yaw, self.pitch, self.dist = 0.0, 0.05, 7.5
        self._prog, self._mouse = None, None
        self.setMinimumSize(380, 300)

    def set_items(self, items):
        self.items = items
        self.update()

    def initializeGL(self):
        gl.glClearColor(0.027, 0.035, 0.055, 1.0)
        try:
            self._prog = gl_shaders.compileProgram(gl_shaders.compileShader(VS, gl.GL_VERTEX_SHADER),
                                                   gl_shaders.compileShader(FS, gl.GL_FRAGMENT_SHADER))
        except Exception as exc:
            print("shader compile failed, wireframe fallback:", exc, file=sys.stderr)
            self._prog = None

    def paintGL(self):
        self.paints += 1
        dpr = self.devicePixelRatioF()
        w, h = max(self.width(), 1), max(self.height(), 1)
        gl.glViewport(0, 0, int(w * dpr), int(h * dpr))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(35.0, w / h, 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        tg = (0.0, -0.7, 0.0)
        eye = (tg[0] + self.dist * math.cos(self.pitch) * math.sin(self.yaw), tg[1] + self.dist * math.sin(self.pitch),
               tg[2] + self.dist * math.cos(self.pitch) * math.cos(self.yaw))
        glu.gluLookAt(*eye, *tg, 0.0, 1.0, 0.0)
        self._floor()
        for it in self.items:
            self._draw(it)

    def _floor(self):
        gl.glUseProgram(0)
        gl.glLineWidth(1.0)
        gl.glColor4f(0.10, 0.13, 0.19, 0.8)
        gl.glBegin(gl.GL_LINES)
        for i in range(-12, 13):
            gl.glVertex3f(i, -2.6, -12); gl.glVertex3f(i, -2.6, 12)
            gl.glVertex3f(-12, -2.6, i); gl.glVertex3f(12, -2.6, i)
        gl.glEnd()

    def _draw(self, it):
        gl.glPushMatrix()
        gl.glTranslatef(float(it["offset"]), 0.0, 0.0)
        if it.get("rot") is not None:
            M = np.eye(4)
            M[:3, :3] = it["rot"]
            gl.glMultMatrixf(np.ascontiguousarray(M.T, dtype=np.float32))
        pos, nrm, col, idx = it["pos"], it["nrm"], it["col"], it["idx"]
        if self._prog is not None:
            gl.glUseProgram(self._prog)
            locs = [gl.glGetAttribLocation(self._prog, n) for n in ("aPos", "aNrm", "aCol")]
            for loc, arr, size in zip(locs, (pos, nrm, col), (3, 3, 4)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, 0, arr)
            gl.glUniform3f(gl.glGetUniformLocation(self._prog, "uSkin"), *it["skin"])
            gl.glUniform1f(gl.glGetUniformLocation(self._prog, "uHeat"),
                           1.0 if (it.get("force_color") or (self.heat and it["heat"])) else 0.0)
            gl.glDrawElements(gl.GL_TRIANGLES, idx.size, gl.GL_UNSIGNED_INT, idx)
            for loc in locs:
                gl.glDisableVertexAttribArray(loc)
            gl.glUseProgram(0)
        if self.wire or self._prog is None:
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, pos)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glColor4f(0.0, 0.94, 1.0, 0.35)
            gl.glDrawElements(gl.GL_TRIANGLES, idx.size, gl.GL_UNSIGNED_INT, idx)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glPopMatrix()

    def mousePressEvent(self, e):
        self._mouse = e.position()

    def mouseMoveEvent(self, e):
        if self._mouse is not None and e.buttons() & QtCore.Qt.LeftButton:
            d = e.position() - self._mouse
            self.yaw -= d.x() * 0.008
            self.pitch = float(np.clip(self.pitch + d.y() * 0.008, -1.2, 1.2))
            self.update()
        self._mouse = e.position()

    def wheelEvent(self, e):
        self.dist = float(np.clip(self.dist * (0.999 ** e.angleDelta().y()), 2.0, 30.0))
        self.update()


REGION_COLOR = {"lips": (0.76, 0.32, 0.35), "eyes": (0.24, 0.16, 0.12), "brows": (0.20, 0.13, 0.09), "nose": (0.87, 0.67, 0.57)}


def _sphere_item(center, radius, color, tk, F):
    v, t = ellipsoid((radius,) * 3, 14, 9)
    pos = ((v + center) * F).astype(np.float32)
    col = np.tile(np.array([*color, 1.0], np.float32), (len(v), 1))
    return {"pos": pos, "nrm": MeshBuilder.normals(pos, t), "col": col, "idx": t.astype(np.uint32), "rot": tk["S"],
            "skin": (1, 1, 1), "heat": True, "force_color": True}


class FaceScene:
    """Turns face packets into drawable items: live mesh, character (.glb), both, or only the performance features
    (eyes, brows, nose, lips) of the scanned actor."""
    SKIN = (0.86, 0.66, 0.56)

    def __init__(self):
        self.mesh, self.character, self.identity, self.rig = MeshBuilder(), None, None, None
        self._nrm, self._n = {}, 0

    def _features(self, tk, F):
        rig, subs = self.rig, []
        L = tk["L"]
        base = rig.identity + (L - tk["neutral_L"]) if tk.get("neutral_L") is not None else L
        pos_full = (base * F)
        idx = {int(v): i for i, v in enumerate(rig.feat)}
        tris = np.array([[idx[a] for a in t] for t in rig.tris if all(int(a) in idx for a in t)], dtype=np.uint32).reshape(-1, 3)
        pos = pos_full[rig.feat].astype(np.float32)
        col = np.tile(np.array([*self.SKIN, 1.0], np.float32), (len(pos), 1))
        for name, members in rig.regions.items():
            for v in members:
                if v in idx:
                    col[idx[v]] = [*REGION_COLOR.get(name, self.SKIN), 1.0]
        if len(tris):
            subs.append({"pos": pos, "nrm": MeshBuilder.normals(pos, tris.astype(np.int64)), "col": col, "idx": tris,
                         "rot": tk["S"], "skin": self.SKIN, "heat": True, "force_color": True})
        for eye_idx, iris in (([33, 133, 159, 145], 468), ([263, 362, 386, 374], 473)):
            c = L[eye_idx].mean(axis=0) + np.array([0.0, 0.0, 0.07])          # eyeball sits behind the lids (face z points inward)
            subs.append(_sphere_item(c, 0.15, (0.95, 0.95, 0.93), tk, F))
            ic = L[iris] + np.array([0.0, 0.0, -0.02])
            subs.append(_sphere_item(ic, 0.075, (0.16, 0.10, 0.07), tk, F))
        return subs

    def items(self, tracks, mode):
        """mode: 0 live mesh, 1 character, 2 both, 3 performance features of the scanned face"""
        self._n += 1
        F = np.array([1.0, -1.0, -1.0])
        subs = []
        if mode == 3 and self.rig is not None:
            for tk in tracks:
                subs += self._features(tk, F)
            for s in subs:
                s["offset"] = 0.0
            return subs
        mode = min(mode, 2)
        for tk in tracks:
            if mode in (0, 2):
                L = tk["L"][:468]
                base = np.array(self.identity)[:468] if self.identity is not None else tk["neutral_L"][:468]
                pos = ((base + (L - tk["neutral_L"][:468])) * F).astype(np.float32)
                if not self.mesh.oriented:
                    self.mesh.orient(pos)
                subs.append({"pos": pos, "nrm": MeshBuilder.normals(pos, self.mesh.tris),
                             "col": heat_rgba(tk["heat_W"] @ tk["muscles"]), "idx": self.mesh.tris,
                             "rot": tk["S"], "skin": self.SKIN, "heat": True})
            if mode in (1, 2) and self.character is not None:
                ch = self.character
                pos = np.ascontiguousarray(ch.pose(tk["bs"]), dtype=np.float32)
                nrm = self._nrm.get(tk["slot"])
                if nrm is None or len(nrm) != len(pos) or len(ch.tris) < 20000 or self._n % 4 == 0:
                    nrm = self._nrm[tk["slot"]] = MeshBuilder.normals(pos, ch.tris)
                subs.append({"pos": pos, "nrm": nrm, "col": ch.col0, "idx": ch.idx, "rot": tk["S"],
                             "skin": self.SKIN, "heat": False})
        for i, s in enumerate(subs):
            s["offset"] = (i - (len(subs) - 1) / 2) * 3.4
        return subs
