"""Render the SVG logo into PNG/ICO files:  python packaging/make_icons.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

ASSETS = Path(__file__).resolve().parent.parent / "kinetra" / "assets"


def render(svg, size, out):
    r = QtSvg.QSvgRenderer(str(ASSETS / svg))
    vb = r.viewBoxF()
    w, h = (size, int(size * vb.height() / vb.width()))
    img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    r.render(p)
    p.end()
    img.save(str(ASSETS / out))
    return img


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    render("icon.svg", 512, "icon.png")
    render("icon.svg", 256, "icon.ico")
    render("logo.svg", 1640, "logo.png")
    print("icons written to", ASSETS)
