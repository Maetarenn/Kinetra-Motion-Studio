"""Professional icon factory for Kinetra UI.
Generates crisp vector icons using QPainter. No external files needed.
Icons are cached after first creation."""
import math
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QIcon, QPainterPath, QBrush

_C: dict[str, QIcon] = {}

def _mk(key, fn, sz=20):
    if key in _C: return _C[key]
    px = QPixmap(sz, sz); px.fill(Qt.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.Antialiasing); fn(p, sz); p.end()
    ic = QIcon(px); _C[key] = ic; return ic

def camera(c="#00F0FF"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.6)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(2,5,12,10,2,2)
        p.drawPolygon([QPointF(14,7),QPointF(18,4),QPointF(18,16),QPointF(14,13)])
    return _mk(f"cam{c}",_p)

def add(c="#14B8A6"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),2)); m=s//2
        p.drawEllipse(2,2,s-4,s-4); p.drawLine(m,5,m,s-5); p.drawLine(5,m,s-5,m)
    return _mk(f"add{c}",_p)

def refresh(c="#38BDF8"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),2)); p.setBrush(Qt.NoBrush)
        pa=QPainterPath(); pa.arcMoveTo(3,3,14,14,60); pa.arcTo(3,3,14,14,60,240); p.drawPath(pa)
        e=pa.currentPosition(); p.setBrush(QBrush(QColor(c)))
        p.drawPolygon([e, e+QPointF(3,-3), e+QPointF(0,4)])
    return _mk(f"ref{c}",_p)

def remove(c="#F87171"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),2)); p.drawLine(5,5,s-5,s-5); p.drawLine(s-5,5,5,s-5)
    return _mk(f"rem{c}",_p)

def preview(c="#FBBF24"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); p.setBrush(Qt.NoBrush); m=s//2
        pa=QPainterPath(); pa.moveTo(1,m); pa.cubicTo(4,4,s-4,4,s-1,m)
        pa.cubicTo(s-4,s-4,4,s-4,1,m); p.drawPath(pa); p.drawEllipse(m-3,m-3,6,6)
    return _mk(f"prev{c}",_p)

def scan(c="#38BDF8"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),2)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(2,2,11,11); p.drawLine(11,11,17,17)
    return _mk(f"scn{c}",_p)

def setup(c="#00F0FF"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); p.setBrush(Qt.NoBrush); m=s//2
        p.drawEllipse(m-4,m-4,8,8)
        for a in range(0,360,45):
            r=math.radians(a)
            p.drawLine(QPointF(m+5*math.cos(r),m+5*math.sin(r)),QPointF(m+8*math.cos(r),m+8*math.sin(r)))
    return _mk(f"set{c}",_p)

def play(c="#22C55E"):
    def _p(p,s):
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c)))
        p.drawPolygon([QPointF(5,3),QPointF(17,10),QPointF(5,17)])
    return _mk(f"ply{c}",_p)

def stop(c="#EF4444"):
    def _p(p,s):
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c))); p.drawRoundedRect(4,4,12,12,2,2)
    return _mk(f"stp{c}",_p)

def record(c="#EF4444"):
    def _p(p,s):
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c))); p.drawEllipse(3,3,14,14)
    return _mk(f"rec{c}",_p)

def crosshair(c="#00F0FF"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); p.setBrush(Qt.NoBrush); m=s//2
        p.drawEllipse(4,4,12,12)
        p.drawLine(m,1,m,6); p.drawLine(m,14,m,19); p.drawLine(1,m,6,m); p.drawLine(14,m,19,m)
    return _mk(f"xh{c}",_p)

def face(c="#FBBF24"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(3,2,14,16)
        p.setBrush(QBrush(QColor(c))); p.drawEllipse(7,7,2,2); p.drawEllipse(11,7,2,2)
        p.setBrush(Qt.NoBrush)
        pa=QPainterPath(); pa.moveTo(7,13); pa.cubicTo(8,15,12,15,13,13); p.drawPath(pa)
    return _mk(f"fac{c}",_p)

def wave(c="#A78BFA"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),2))
        for r in [3,6,9]:
            pa=QPainterPath(); pa.arcMoveTo(10-r,10-r,r*2,r*2,30)
            pa.arcTo(10-r,10-r,r*2,r*2,30,120); p.drawPath(pa)
    return _mk(f"wav{c}",_p)

def dots(c="#00F0FF"):
    def _p(p,s):
        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c)))
        for x,y in [(5,5),(10,5),(15,5),(5,10),(10,10),(15,10),(5,15),(10,15),(15,15)]:
            p.drawEllipse(x-1,y-1,3,3)
    return _mk(f"dot{c}",_p)

def folder(c="#FBBF24"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); p.setBrush(Qt.NoBrush)
        p.drawRect(2,7,16,10); p.drawLine(2,7,2,5); p.drawLine(2,5,8,5); p.drawLine(8,5,10,7)
    return _mk(f"fld{c}",_p)

def arrows(c="#00F0FF"):
    def _p(p,s):
        p.setPen(QPen(QColor(c),1.5)); m=s//2
        p.drawLine(m,2,m,18); p.drawLine(m-4,6,m,2); p.drawLine(m+4,6,m,2)
        p.drawLine(2,m,18,m); p.drawLine(14,m-4,18,m); p.drawLine(14,m+4,18,m)
    return _mk(f"arr{c}",_p)
